from pathlib import Path
from google.cloud import documentai
from tqdm import tqdm

# ====== Configuration ======
PROJECT_ID = "banner-data-labelling"
LOCATION = "us"  # Same LOCATION as used in curl
PROCESSOR_ID = "94ab8c7501715dd1"  # Document AI OCR Processor ID
# ======================

# Input/Output directories
INPUT_DIR = Path(
    "~/CV_final_project/data/processed_444_pages/cropped_panels"
).expanduser()

OUTPUT_DIR = Path(
    "~/CV_final_project/data/processed_444_pages/ocr_result"
).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_client_and_name():
    """Create Document AI client and processor resource name"""
    client = documentai.DocumentProcessorServiceClient()
    name = client.processor_path(PROJECT_ID, LOCATION, PROCESSOR_ID)
    return client, name


def process_one_image(client, name, img_path: Path, mime_type: str):
    """Send one image file to Document AI OCR and save the result"""
    with img_path.open("rb") as f:
        content = f.read()

    raw_document = documentai.RawDocument(content=content, mime_type=mime_type)
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)
    result = client.process_document(request=request)
    doc = result.document

    text = doc.text or ""
    base = img_path.stem

    # Save text only (.txt)
    out_path = OUTPUT_DIR / f"{base}.txt"
    out_path.write_text(text, encoding="utf-8")

    # Can disable this print if too verbose
    # print(f"✅ {img_path.name} OCR complete")


def main():
    client, name = get_client_and_name()

    patterns = ["*.png", "*.jpg", "*.jpeg"]
    img_files = []
    for p in patterns:
        img_files.extend(sorted(INPUT_DIR.glob(p)))

    if not img_files:
        print(f"No images found: {INPUT_DIR}")
        return

    print(f"Starting OCR for {len(img_files)} images...")

    for img_path in tqdm(img_files, desc="Document AI OCR Progress", unit="image"):
        base = img_path.stem
        out_path = OUTPUT_DIR / f"{base}.txt"

        # Skip if result already exists
        if out_path.exists():
            continue

        ext = img_path.suffix.lower()
        if ext in [".jpg", ".jpeg"]:
            mime = "image/jpeg"
        else:
            mime = "image/png"

        process_one_image(client, name, img_path, mime)

    print("🎉 OCR complete for all images")



if __name__ == "__main__":
    main()
