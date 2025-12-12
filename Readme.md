Handwritten Medical OCR + PII Extraction
========================================

End‑to‑end pipeline for extracting text and detecting PII from **handwritten medical documents**, with an **EasyOCR backend** and a simple **Flask web UI**.

> **Input JPEG/PNG → Pre‑processing → EasyOCR → Text Cleaning → PII Detection → Optional Redacted Image → Web UI**

This project was built as part of an **“OCR Pipeline – Handwritten document PII Extraction”** assignment and focuses on real‑world, messy doctor / clinic notes.

Features
--------

*   Handles scanned **handwritten medical notes and forms** (JPEG/PNG).
    
*   Robust pre‑processing: resize, deskew, denoise, contrast enhancement, multi‑binarization.
    
*   **Deep‑learning OCR** via EasyOCR for better performance on handwriting.
    
*   **PII detection** using regex + keyword labels for:
    
    *   Phone numbers
        
    *   Dates
        
    *   UHID / IPD‑style identifiers
        
    *   Patient / doctor related fields (patient name, age, sex, mobile, dept, bed no, doctor name, etc.)
        
*   Generates an optional **redacted image** with PII masked by black boxes.
    
*   **Double‑page support**: automatically splits wide scans into left/right pages.
    
*   **Flask web app** to upload an image, view:
    
    *   Recognized text
        
    *   Detected PII items
        
    *   Original vs redacted image previews
        

Project Structure
-----------------

OCR_Project/
  app.py                     # Flask backend
  ocr_pipeline_easyocr.py    # Main OCR + PII + redaction pipeline
  Code_File/                 # (if you keep this name, adjust paths in app.py)
    assets/                  # Sample images / static assets
    templates/
      index.html             # Web UI for upload + results
    uploads/                 # Uploaded images (created at runtime)
    redacted/                # Redacted output images (created at runtime)
  .venv/                     # Optional virtualenv (not committed)
  README.md                  # This file


Only these are strictly required to run the assignment solution:

*   ocr\_pipeline\_easyocr.py
    
*   app.py
    
*   templates/index.html
    

If you still keep an older ocr\_pipeline\_tesseract.py, it is just for reference.

How It Works (High Level)
-------------------------

1.  **Pre‑processing**
    
    *   Load the image, resize to a target range (≈2000–4000 px longest side).
        
    *   Convert to grayscale and **deskew** using HoughLines / minAreaRect.
        
    *   Apply CLAHE contrast enhancement, denoising, sharpening.
        
    *   Create several **binarized variants** (Gaussian, mean adaptive, Otsu) plus raw grayscale.
        
    *   Optionally run simple morphological open/close to clean noise.
        
2.  **Multi‑variant OCR with EasyOCR**
    
    *   Run easyocr.Reader(\['en'\]) on each preprocessed variant.
        
    *   Collect word‑level boxes (text, x, y, w, h, confidence).
        
    *   Filter low‑confidence words and **group words into lines** based on Y‑overlap.
        
    *   For each line store merged text and average confidence.
        
3.  **Line Merging / Fusion**
    
    *   Lines from different variants are compared using **Jaccard similarity** over word sets.
        
    *   Highly similar lines (> 0.8) are treated as duplicates; the line with higher confidence is kept.
        
    *   Final lines\_map is sorted top‑to‑bottom and indexed from 0..N-1.
        
4.  **PII Detection**
    
    *   **Regex PII**: phone numbers, dates, UHID/IPD‑like identifiers using compiled regex patterns.
        
    *   **Label‑based PII**: search for keywords such as patient name, age:, sex, mobile, dept, bed no, dr., etc., and extract the value that follows within the same line.
        
    *   Each detected value is mapped back to its word boxes so it can be masked later.
        
    *   json{ "type": "patient\_name", "value": "John Doe", "line": 0, "page": "left"}
        
5.  **Redaction & Double‑Page Logic**
    
    *   If the image width is much larger than height (w > 1.4 \* h), treat it as a **double page** and split into left/right halves.
        
    *   OCR + PII runs independently on each half; coordinates for the right half are shifted back onto the full image.
        
    *   Redaction draws black rectangles with a small padding over all PII boxes and saves the redacted image.
        
6.  **Flask Web App**
    
    *   GET / renders index.html with an upload form.
        
    *   POST /upload:
        
        *   Saves the uploaded image to uploads/.
            
        *   Calls run\_pipeline(image\_path, redacted\_output\_path) from ocr\_pipeline\_easyocr.py.
            
        *   Returns JSON with:
            
            *   full\_text
                
            *   pii\_items
                
            *   uploaded\_image\_url
                
            *   redacted\_image\_url (if redaction generated)
                
    *   GET /files// serves original and redacted images.
        

The front‑end (plain HTML + JavaScript) calls /upload via fetch, then updates:

*   Original image preview
    
*   Redacted image preview
    
*   PII results table
    
*   Text area with full OCR text
    

Installation
------------

1\. Clone the Repository
------------------------
git clone https://github.com/RohitMakaniProfile/OCR_Project.git
cd OCR_Project


2\. Create and Activate Virtualenv (recommended)
------------------------------------------------
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1


3\. Install Dependencies
------------------------
pip install --upgrade pip
pip install flask easyocr opencv-python numpy


On macOS using the official python.org installer, run the Install Certificates.command script once if EasyOCR fails to download models due to SSL errors.

Running the Web App
-------------------

From the project root (where app.py lives):

python app.py

You should see something like:
* Serving Flask app 'app'
* Debug mode: on
* Running on http://127.0.0.1:5000/


Then:

1.  Open http://127.0.0.1:5000/ in your browser.
    
2.  Upload a handwritten medical document image (.jpg, .jpeg, .png).
    
3.  Click the button to run OCR + PII.
    
4.  Wait for processing to finish and inspect:
    
    *   Original image
        
    *   Redacted image
        
    *   Detected PII items (type, value, line, page)
        
    *   Full recognized text
        
    

Assignment Alignment
--------------------

This project was designed to satisfy the **“OCR Pipeline – Handwritten document PII Extraction”** assignment requirements:

*   **Input**: JPEG handwritten medical documents (doctor‑style notes and forms).
    
*   **Robustness**: Supports **tilted** images and multiple handwriting styles via advanced pre‑processing plus EasyOCR.
    
*   **PII Coverage**:
    
    *   Phone numbers, dates, UHID/IPD identifiers via regex.
        
    *   Patient / doctor‑related labels via keyword‑based extraction.
        
*   **Outputs**:
    
    *   Machine‑readable text.
        
    *   Structured list of PII items.
        
    *   Optional redacted image with PII masked.
        
    *   Simple web interface for interactive testing and demonstration.
        

Known Limitations & Future Work
-------------------------------

*   OCR quality still degrades for extremely messy handwriting or very low‑resolution scans.
    
*   PII detection is primarily regex + keyword based; complex contexts can produce false positives / negatives.
    

Possible improvements:

*   Add medical NER models for more semantic PII detection.
    
*   Support more languages via EasyOCR.
    
*   Confidence‑based thresholds and human‑in‑the‑loop review for low‑confidence lines.
    
*   Highlight overlays instead of hard black boxes for redaction previews.
    


