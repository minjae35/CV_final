from pathlib import Path
from google.cloud import documentai
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ====== Configuration ======
PROJECT_ID = "banner-data-labelling"
LOCATION = "us"
PROCESSOR_ID = "331ae0287e6eaa30"  # home-data-ocr processor ID
MAX_WORKERS = 10  # Number of concurrent threads (considering API rate limit)
# ======================

# Input/Output directories
INPUT_DIR = Path("~/data/raw_panel_images_small").expanduser()
OUTPUT_DIR = Path("~/data/small_ocr_result").expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Thread-local storage for client
thread_local = threading.local()

def get_client_and_name():
    """Get thread-local client and processor name"""
    if not hasattr(thread_local, 'client'):
        thread_local.client = documentai.DocumentProcessorServiceClient()
    client = thread_local.client
    name = client.processor_path(PROJECT_ID, LOCATION, PROCESSOR_ID)
    return client, name


def process_one_image(img_path: Path):
    """Process single image with OCR"""
    # Extract comic_no from parent directory name
    # Path structure: ~/data/raw_panel_images_small/{comic_no}/{page_no}_{panel_idx}.jpg
    comic_no = img_path.parent.name
    base = img_path.stem  # {page_no}_{panel_idx}
    
    # Create unique filename: {comic_no}_{page_no}_{panel_idx}.txt
    out_path = OUTPUT_DIR / f"{comic_no}_{base}.txt"
    if out_path.exists():
        return True, img_path.name
    
    try:
        # Get thread-local client
        client, name = get_client_and_name()
        
        # Read image
        with img_path.open("rb") as f:
            content = f.read()
        
        # Determine MIME type
        ext = img_path.suffix.lower()
        mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
        
        # Process with Document AI
        raw_document = documentai.RawDocument(content=content, mime_type=mime_type)
        request = documentai.ProcessRequest(name=name, raw_document=raw_document)
        result = client.process_document(request=request)
        doc = result.document
        
        # Save result
        text = doc.text or ""
        out_path.write_text(text, encoding="utf-8")
        
        return True, img_path.name
    except Exception as e:
        return False, f"{img_path.name}: {str(e)}"


def main():
    # Find all image files
    patterns = ["*.png", "*.jpg", "*.jpeg"]
    img_files = []
    for p in patterns:
        # Recursively search in subdirectories
        img_files.extend(sorted(INPUT_DIR.rglob(p)))

    if not img_files:
        print(f"No images found: {INPUT_DIR}")
        return

    # Filter out already processed files
    remaining_files = []
    for img_path in img_files:
        comic_no = img_path.parent.name
        base = img_path.stem
        out_path = OUTPUT_DIR / f"{comic_no}_{base}.txt"
        if not out_path.exists():
            remaining_files.append(img_path)
    
    total = len(img_files)
    remaining = len(remaining_files)
    already_done = total - remaining
    
    print(f"Already processed {already_done} out of {total} images")
    print(f"Starting OCR for {remaining} images (parallel processing: {MAX_WORKERS} threads)...")
    
    if remaining == 0:
        print("🎉 All images already processed")
        return

    # Process with ThreadPoolExecutor
    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_img = {executor.submit(process_one_image, img_path): img_path 
                        for img_path in remaining_files}
        
        # Process completed tasks with progress bar
        with tqdm(total=remaining, desc="Document AI OCR Progress", unit="image") as pbar:
            for future in as_completed(future_to_img):
                success, result = future.result()
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    tqdm.write(f"❌ Error: {result}")
                pbar.update(1)

    print("\n" + "="*80)
    print("OCR Complete")
    print("="*80)
    print(f"Success: {success_count}")
    print(f"Error: {error_count}")
    print(f"Total processed: {success_count + error_count}")
    print("🎉 OCR complete for ~/data/raw_panel_images_small images")


if __name__ == "__main__":
    main()
