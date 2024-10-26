import os
import json
import PyPDF2
from pdf2image import convert_from_path
from google.cloud import vision
from google.cloud import storage
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.auth import default
from tqdm import tqdm
import time
from datetime import datetime
import tempfile
import logging
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import io
import gc
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pdf_processing.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
SOURCE_FOLDER_ID = "1JHMoXCcjgPUcDQ8rjvlyxkvoEMtgbhhn"
CREDENTIALS_PATH = "google_cloud_credentials.json"
OUTPUT_BASE = "output"
MAX_WORKERS = 3  # Reduced for stability
BATCH_SIZE = 3   # Reduced batch size
PDF_DPI = 300
COMPRESSION_QUALITY = 95
CHUNK_SIZE = 5 * 1024 * 1024  # 5MB chunks
MAX_RETRIES = 5
PROCESS_TIMEOUT = 600  # 10 minutes timeout for processing

SCOPES = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.file',
    'https://www.googleapis.com/auth/drive.readonly',
    'https://www.googleapis.com/auth/drive.metadata.readonly'
]

class DriveConfig:
    def __init__(self):
        self.FOLDER_ID = SOURCE_FOLDER_ID
        self.OUTPUT_FOLDER = OUTPUT_BASE
        self.CREDENTIALS_PATH = CREDENTIALS_PATH
        self.create_directories()
        self.drive_service = self.initialize_drive_service()

    def create_directories(self):
        """Create necessary output directories"""
        os.makedirs(self.OUTPUT_FOLDER, exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_FOLDER, "text"), exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_FOLDER, "metadata"), exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_FOLDER, "temp"), exist_ok=True)

    def initialize_drive_service(self):
        """Initialize Google Drive service with retry mechanism"""
        try:
            credentials = service_account.Credentials.from_service_account_file(
                self.CREDENTIALS_PATH,
                scopes=SCOPES
            )
            service = build('drive', 'v3', credentials=credentials, cache_discovery=False)
            
            try:
                about = service.about().get(fields="user").execute()
                logger.info(f"Connected to Drive as: {about.get('user', {}).get('emailAddress', 'Unknown')}")
            except Exception as e:
                logger.error(f"Drive connection test failed: {e}")
            
            return service
        except Exception as e:
            logger.error(f"Failed to initialize Drive service: {e}")
            raise

def download_file_safely(service, file_id, destination):
    """Download a file from Drive with retries"""
    for attempt in range(MAX_RETRIES):
        try:
            request = service.files().get_media(fileId=file_id)
            with open(destination, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request, chunksize=CHUNK_SIZE)
                done = False
                while not done:
                    try:
                        status, done = downloader.next_chunk()
                        if status:
                            logger.info(f"Download progress: {int(status.progress() * 100)}%")
                    except Exception as e:
                        if attempt == MAX_RETRIES - 1:
                            raise
                        logger.warning(f"Chunk download failed: {e}, retrying...")
                        time.sleep(2)
                        break  # Break inner loop to retry whole download
            return True
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                logger.error(f"Download failed after {MAX_RETRIES} attempts: {e}")
                return False
            logger.warning(f"Download attempt {attempt + 1} failed, retrying...")
            time.sleep(2 ** attempt)  # Exponential backoff
    return False

class PDFProcessor:
    def __init__(self, config):
        self.config = config
        self.vision_api = self.init_vision_api()

    def init_vision_api(self):
        """Initialize Vision API client"""
        try:
            credentials = service_account.Credentials.from_service_account_file(
                self.config.CREDENTIALS_PATH)
            return vision.ImageAnnotatorClient(credentials=credentials)
        except Exception as e:
            logger.error(f"Failed to initialize Vision API: {e}")
            raise

    def process_single_image(self, image_path):
        """Process a single image with Vision API"""
        try:
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            image = vision.Image(content=content)
            response = self.vision_api.document_text_detection(
                image=image,
                image_context={"language_hints": ["en"]}
            )
            
            if response.error.message:
                raise Exception(response.error.message)
                
            return {
                'text': response.full_text_annotation.text if response.full_text_annotation else "",
                'confidence': response.text_annotations[0].confidence if response.text_annotations else 0,
                'word_count': len(response.full_text_annotation.text.split()) if response.full_text_annotation else 0
            }
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return None

    def process_pdf(self, file_info):
        """Process a single PDF file"""
        temp_pdf_path = None
        try:
            temp_pdf_path = os.path.join(self.config.OUTPUT_FOLDER, "temp", f"{file_info['id']}.pdf")
            
            # Download PDF using safe download
            logger.info(f"Processing: {file_info['name']}")
            if not download_file_safely(self.config.drive_service, file_info['id'], temp_pdf_path):
                raise Exception("Failed to download file")

            results = []
            # Convert and process PDF in small batches
            with tempfile.TemporaryDirectory() as temp_dir:
                # Get total pages
                with open(temp_pdf_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    total_pages = len(pdf_reader.pages)

                # Process in small batches
                for start_page in range(1, total_pages + 1, BATCH_SIZE):
                    end_page = min(start_page + BATCH_SIZE - 1, total_pages)
                    
                    # Convert batch to images
                    images = convert_from_path(
                        temp_pdf_path,
                        dpi=PDF_DPI,
                        thread_count=2,
                        grayscale=False,
                        use_cropbox=True,
                        first_page=start_page,
                        last_page=end_page
                    )

                    # Process each image
                    for idx, image in enumerate(images):
                        page_num = start_page + idx
                        temp_path = os.path.join(temp_dir, f'page_{page_num}.png')
                        
                        try:
                            # Save image
                            image = image.convert('RGB')
                            image.save(temp_path, 'PNG', optimize=True, quality=COMPRESSION_QUALITY)
                            
                            # Process with Vision API
                            result = self.process_single_image(temp_path)
                            if result:
                                result['page_number'] = page_num
                                results.append(result)
                            
                            # Clean up image immediately
                            os.remove(temp_path)
                            
                        except Exception as e:
                            logger.error(f"Error processing page {page_num}: {e}")
                            continue
                        
                        finally:
                            gc.collect()

            # Save results
            if results:
                doc_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + file_info['id']
                text_folder = os.path.join(self.config.OUTPUT_FOLDER, "text", doc_id)
                os.makedirs(text_folder, exist_ok=True)
                
                for result in results:
                    if result:
                        text_path = os.path.join(text_folder, f"page_{result['page_number']}.txt")
                        with open(text_path, 'w', encoding='utf-8') as f:
                            f.write(result['text'])
                
                metadata = {
                    'document_id': doc_id,
                    'drive_file_id': file_info['id'],
                    'original_filename': file_info['name'],
                    'processing_date': datetime.now().isoformat(),
                    'page_count': len(results),
                    'pages': results
                }
                
                metadata_path = os.path.join(self.config.OUTPUT_FOLDER, "metadata", f"{doc_id}_metadata.json")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"Successfully processed {file_info['name']}")
                return {'status': 'success', 'doc_id': doc_id, 'pages': len(results)}
            
            raise Exception("No results generated")
            
        except Exception as e:
            logger.error(f"Error processing {file_info['name']}: {e}")
            return {'status': 'failed', 'error': str(e), 'file': file_info['name']}
        finally:
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                try:
                    os.remove(temp_pdf_path)
                except:
                    pass
            gc.collect()

def main():
    """Main processing function"""
    try:
        # Initialize configuration
        config = DriveConfig()
        processor = PDFProcessor(config)
        
        logger.info(f"Searching for PDFs in folder ID: {SOURCE_FOLDER_ID}")
        
        try:
            # List all files
            results = []
            page_token = None
            
            while True:
                query = f"'{SOURCE_FOLDER_ID}' in parents"
                response = config.drive_service.files().list(
                    q=query,
                    spaces='drive',
                    fields='nextPageToken, files(id, name, mimeType)',
                    pageToken=page_token
                ).execute()
                
                files = response.get('files', [])
                logger.info(f"Found {len(files)} files in folder")
                
                # Filter PDF files
                pdf_files = [f for f in files if f['mimeType'] == 'application/pdf']
                results.extend(pdf_files)
                
                page_token = response.get('nextPageToken', None)
                if page_token is None:
                    break
            
            if not results:
                logger.warning("No PDF files found in the specified folder!")
                return
            
            logger.info(f"Found {len(results)} PDF files in Google Drive folder")
            
            # Process files with parallel processing
            success_count = 0
            failed_files = []
            
            # Process files in smaller batches
            batch_size = 3  # Process only 3 files at a time
            for i in range(0, len(results), batch_size):
                batch = results[i:i + batch_size]
                
                with ThreadPoolExecutor(max_workers=batch_size) as executor:
                    futures = {executor.submit(processor.process_pdf, file_info): file_info 
                              for file_info in batch}
                    
                    batch_number = i//batch_size + 1
                    total_batches = (len(results) + batch_size - 1)//batch_size
                    
                    for future in tqdm(futures, 
                                     total=len(batch), 
                                     desc=f"Processing batch {batch_number}/{total_batches}"):
                        try:
                            result = future.result()
                            if result['status'] == 'success':
                                success_count += 1
                            else:
                                failed_files.append(result['file'])
                        except Exception as e:
                            logger.error(f"Task failed: {e}")
                            failed_files.append(futures[future]['name'])
                
                # Force cleanup between batches
                gc.collect()
                time.sleep(1)  # Small delay between batches
            
            # Print summary
            logger.info("\n📊 Processing Summary")
            logger.info("=" * 50)
            logger.info(f"Total PDFs processed: {len(results)}")
            logger.info(f"Successfully processed: {success_count}")
            logger.info(f"Failed: {len(failed_files)}")
            
            if failed_files:
                logger.info("\nFailed Files:")
                for file in failed_files:
                    logger.info(f"- {file}")
            
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"Error accessing folder: {e}")
            return
            
    except Exception as e:
        logger.error(f"Error during processing: {e}")
    finally:
        gc.collect()

if __name__ == "__main__":
    main()
