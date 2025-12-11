#!/usr/bin/env python3
"""
Create HTML file to view images and their OCR results side by side
"""
from pathlib import Path
import html
import base64

INPUT_DIR = Path("~/data/raw_panel_images_small").expanduser()
OCR_DIR = Path("~/data/small_ocr_result").expanduser()
OUTPUT_HTML = Path("~/data/ocr_viewer.html").expanduser()

def create_html_viewer(num_samples=50):
    """Create HTML file with images and OCR results"""
    
    # Find processed images
    img_files = []
    for img_path in sorted(INPUT_DIR.rglob("*.jpg")):
        base = img_path.stem
        ocr_file = OCR_DIR / f"{base}.txt"
        if ocr_file.exists():
            img_files.append((img_path, ocr_file))
            if len(img_files) >= num_samples:
                break
    
    if not img_files:
        print("No processed images found")
        return
    
    # Create HTML
    total_count = len(img_files)
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>OCR Results Viewer</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .panel {{
            background: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .panel-header {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }}
        .content {{
            display: flex;
            gap: 20px;
        }}
        .image-section {{
            flex: 1;
            min-width: 300px;
        }}
        .image-section img {{
            max-width: 100%;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 4px;
        }}
        .ocr-section {{
            flex: 1;
            min-width: 300px;
        }}
        .ocr-text {{
            background: #f9f9f9;
            padding: 15px;
            border-left: 4px solid #4CAF50;
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.6;
            border-radius: 4px;
        }}
        .file-path {{
            font-size: 12px;
            color: #666;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>OCR Results Viewer</h1>
        <p>Total: {total_count} images with OCR results</p>
"""
    
    # Add each image-OCR pair
    for idx, (img_path, ocr_file) in enumerate(img_files, 1):
        # Read OCR text
        ocr_text = ocr_file.read_text(encoding='utf-8').strip()
        ocr_text_escaped = html.escape(ocr_text)
        
        # Read image and encode as base64
        try:
            img_data = img_path.read_bytes()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            img_ext = img_path.suffix.lower()
            if img_ext in ['.jpg', '.jpeg']:
                img_mime = 'image/jpeg'
            else:
                img_mime = 'image/png'
            img_src = f"data:{img_mime};base64,{img_base64}"
        except Exception as e:
            img_src = f"file://{img_path}"
            print(f"Warning: Could not encode {img_path.name}: {e}")
        
        html_content += f"""
        <div class="panel">
            <div class="panel-header">Image {idx}: {img_path.name}</div>
            <div class="content">
                <div class="image-section">
                    <img src="{img_src}" alt="{img_path.name}" />
                    <div class="file-path">{img_path}</div>
                </div>
                <div class="ocr-section">
                    <h3>OCR Result:</h3>
                    <div class="ocr-text">{ocr_text_escaped}</div>
                </div>
            </div>
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    # Write HTML file
    OUTPUT_HTML.write_text(html_content, encoding='utf-8')
    print(f"✅ HTML viewer created: {OUTPUT_HTML}")
    print(f"   Open in browser to view {len(img_files)} images with OCR results")

if __name__ == "__main__":
    create_html_viewer(num_samples=50)

