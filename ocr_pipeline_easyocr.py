#!/usr/bin/env python3
"""
EasyOCR-based OCR + PII Extraction Pipeline for Handwritten Medical Documents

End-to-end:
Input (handwritten JPEG) → Pre-processing → OCR (EasyOCR) → Text Cleaning
→ PII Detection → (Optional) Redacted Image

Optimized for:
- Slightly tilted images
- Different handwriting styles
- Clinic / doctor-style notes or forms
"""

import cv2
import numpy as np
import easyocr
import re
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# ----------------------------
# Regexes & Keyword Definitions
# ----------------------------

PHONE_RE = re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
DATE_RE = re.compile(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}')
UHID_RE = re.compile(r'\d{8,12}')
IPD_RE = re.compile(r'\d{6,10}')

KEYWORDS = {
    "patient_name": ["patient name", "pt name", "patient:", "name:", "patient name:"],
    "age": ["age", "age:", "age "],
    "sex": ["sex", "gender", "sex:", "gender:"],
    "ipd_no": ["ipd no", "ip no", "ipd no.", "ipd no:"],
    "uhid": ["uhid", "uhid no", "uhid no.", "uhid no:"],
    "mobile": ["mobile", "phone", "tel", "mobile no", "mobile no."],
    "bed_no": ["bed no", "bed no.", "bed no:"],
    "dept": ["dept", "dept.", "dept:", "department"],
    "date": ["date", "date:", "date & time", "date&time"],
    "doctor_name": ["dr.", "dr ", "doctor", "consultant", "resident"],
}

ALL_KEYWORDS_FLAT = sorted(
    {kw for kws in KEYWORDS.values() for kw in kws},
    key=len,
    reverse=True,
)

# Global EasyOCR reader (GPU can be set True if available)
READER = easyocr.Reader(['en'], gpu=False)


# ----------------------------
# Pre-processing / Deskew
# ----------------------------

def detect_skew_angle(image: np.ndarray) -> float:
    """Detect skew angle using edges + minAreaRect as fallback."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                             minLineLength=100, maxLineGap=10)

    if lines is not None and len(lines) > 0:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if -45 < angle < 45:
                    angles.append(angle)
        if angles:
            return float(np.median(angles))

    coords = np.column_stack(np.where(gray < 200))
    if coords.size > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        return float(angle)

    return 0.0


def deskew(image: np.ndarray) -> np.ndarray:
    """Rotate image to correct skew."""
    angle = detect_skew_angle(image)
    if abs(angle) < 0.5:
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """CLAHE for better text contrast."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def remove_noise(image: np.ndarray) -> np.ndarray:
    """Denoise while preserving edges."""
    denoised = cv2.fastNlMeansDenoising(
        image, None, h=10, templateWindowSize=7, searchWindowSize=21
    )
    return denoised


def sharpen_image(image: np.ndarray) -> np.ndarray:
    """Sharpen to highlight text strokes."""
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


def adaptive_binarization(gray: np.ndarray) -> np.ndarray:
    """Combine several binarization strategies."""
    binary1 = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15
    )

    binary2 = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        25, 12
    )

    _, otsu = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    combined = cv2.bitwise_and(binary1, binary2)
    combined = cv2.bitwise_and(combined, otsu)
    return combined


def morphological_cleanup(binary: np.ndarray) -> np.ndarray:
    """Morphological open/close to clean noise."""
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

    return closed


def preprocess_image_advanced(image_path: str) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Advanced preprocessing optimized for handwritten docs.

    Returns:
        original_color_img, [list of preprocessed grayscale/binary images]
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at path: {image_path}")

    h, w = img.shape[:2]
    target_size = 2000
    if max(h, w) < target_size:
        scale = target_size / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    elif max(h, w) > 4000:
        scale = 4000 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = deskew(gray)

    if gray.shape[:2] != img.shape[:2]:
        img = cv2.resize(img, (gray.shape[1], gray.shape[0]))

    preprocessed_versions: List[np.ndarray] = []

    # Version 1: enhanced + binarized + morph cleanup
    enhanced = enhance_contrast(gray)
    binary1 = adaptive_binarization(enhanced)
    binary1 = morphological_cleanup(binary1)
    preprocessed_versions.append(binary1)

    # Version 2: denoised + adaptive threshold
    denoised = remove_noise(gray)
    binary2 = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21, 10
    )
    preprocessed_versions.append(binary2)

    # Version 3: sharpened + contrast + otsu
    sharpened = sharpen_image(gray)
    enhanced_sharp = enhance_contrast(sharpened)
    _, binary3 = cv2.threshold(
        enhanced_sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    preprocessed_versions.append(binary3)

    # Version 4: raw grayscale (EasyOCR handles quite a lot itself)
    preprocessed_versions.append(gray)

    return img, preprocessed_versions


# ----------------------------
# OCR & Line Grouping (EasyOCR)
# ----------------------------

def build_line_data(words: List[Dict]) -> Dict:
    """Build a line dict from word boxes."""
    if not words:
        return {"text": "", "words": [], "conf": 0.0}

    words_sorted = sorted(words, key=lambda w: w["x"])
    text = " ".join(w["text"] for w in words_sorted)
    conf = float(sum(w["conf"] for w in words_sorted) / len(words_sorted))
    return {
        "text": text,
        "words": words_sorted,
        "conf": conf,
    }


def group_words_into_lines(words: List[Dict], img_idx: int) -> Dict:
    """
    Group word boxes into rough text lines based on Y overlap.
    Returns dict: key=(img_idx, line_index) -> line_data
    """
    if not words:
        return {}

    # Sort by vertical position
    words_sorted = sorted(words, key=lambda w: w["y"])
    lines: Dict[Tuple[int, int], Dict] = {}

    current_line_words: List[Dict] = [words_sorted[0]]
    current_center_y = words_sorted[0]["y"] + words_sorted[0]["h"] / 2
    current_height = words_sorted[0]["h"]
    line_index = 0

    for w in words_sorted[1:]:
        center_y = w["y"] + w["h"] / 2
        line_threshold = max(current_height, w["h"]) * 0.6

        if abs(center_y - current_center_y) <= line_threshold:
            current_line_words.append(w)
            # update average center & height
            current_center_y = (current_center_y + center_y) / 2
            current_height = max(current_height, w["h"])
        else:
            key = (img_idx, line_index)
            lines[key] = build_line_data(current_line_words)
            line_index += 1
            current_line_words = [w]
            current_center_y = center_y
            current_height = w["h"]

    key = (img_idx, line_index)
    lines[key] = build_line_data(current_line_words)

    return lines


def text_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity between two texts."""
    if not text1 or not text2:
        return 0.0

    words1 = set(text1.split())
    words2 = set(text2.split())
    if not words1 or not words2:
        return 0.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union)


def merge_duplicate_lines(all_lines: Dict) -> Dict:
    """
    Merge similar lines from different preprocessing variants.
    Keeps the line with higher confidence when texts are similar.
    """
    unique_lines: Dict[Any, Dict] = {}

    for key, data in all_lines.items():
        text_normalized = data["text"].lower().strip()
        found_similar = False

        for existing_key, existing_data in list(unique_lines.items()):
            existing_text = existing_data["text"].lower().strip()
            if text_similarity(text_normalized, existing_text) > 0.8:
                found_similar = True
                if data["conf"] > existing_data["conf"]:
                    unique_lines[existing_key] = data
                break

        if not found_similar:
            unique_lines[key] = data

    return unique_lines


def run_ocr_multi_easyocr(preprocessed_images: List[np.ndarray]) -> Dict[int, Dict]:
    """
    Run EasyOCR on multiple preprocessed versions and merge line results.

    Returns:
        lines_map: {0: {"text": ..., "words": [...], "conf": ...}, 1: ...}
    """
    all_lines: Dict[Any, Dict] = {}

    for img_idx, img in enumerate(preprocessed_images):
        try:
            results = READER.readtext(img, detail=1, paragraph=False)
        except Exception as e:
            print(f"EasyOCR failed on image {img_idx}: {e}", file=sys.stderr)
            continue

        words: List[Dict] = []
        for res in results:
            if len(res) == 3:
                bbox, text, conf = res
            else:
                # Safety in case of different format
                bbox, text, conf = res[0], res[1], res[2]

            text = text.strip()
            if not text:
                continue

            # EasyOCR conf is ~0–1; filter low confidence words
            if conf is not None and conf < 0.3:
                continue

            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x_min, y_min = min(xs), min(ys)
            w_box, h_box = max(xs) - x_min, max(ys) - y_min

            words.append({
                "text": text,
                "x": int(x_min),
                "y": int(y_min),
                "w": int(w_box),
                "h": int(h_box),
                "conf": float(conf * 100.0 if conf is not None else 50.0),
            })

        line_dict = group_words_into_lines(words, img_idx)
        all_lines.update(line_dict)

    merged_lines = merge_duplicate_lines(all_lines)

    # Normalize keys to 0..N in reading order (top to bottom)
    normalized_lines: Dict[int, Dict] = {}
    sorted_items = sorted(
        merged_lines.items(),
        key=lambda kv: min(
            (w["y"] for w in kv[1].get("words", [])),
            default=0
        ),
    )

    for idx, (_, data) in enumerate(sorted_items):
        normalized_lines[idx] = data

    return normalized_lines


# ----------------------------
# Text Cleaning & PII Helpers
# ----------------------------

def clean_ocr_text(text: str) -> str:
    """Basic clean-up: collapse spaces, strip punctuation edges."""
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[^\w]+|[^\w]+$", "", text)
    return text.strip()


def cut_at_next_keyword(line_lower: str, start_idx: int) -> Optional[int]:
    """
    Return position of next keyword in line after start_idx,
    used to cut values like: 'Patient Name: John Doe Age: 32'
    """
    next_positions = []
    for kw in ALL_KEYWORDS_FLAT:
        pos = line_lower.find(kw, start_idx)
        if pos != -1:
            next_positions.append(pos)
    if not next_positions:
        return None
    return min(next_positions)


def extract_value_coordinates(line_data: Dict, value_text: str) -> List[Dict]:
    """
    Find coordinates of value_text within line words,
    returning the list of word-box dicts that correspond to the value.
    """
    coords_list: List[Dict] = []
    value_words = [w for w in value_text.split() if w]
    if not value_words:
        return coords_list

    source_words = line_data["words"]

    for i in range(len(source_words)):
        match = True
        for j, val_word in enumerate(value_words):
            if i + j >= len(source_words):
                match = False
                break
            source_text = source_words[i + j]["text"].lower()
            val_lower = val_word.lower()
            if val_lower not in source_text and source_text not in val_lower:
                match = False
                break
        if match:
            for k in range(min(len(value_words), len(source_words) - i)):
                coords_list.append(source_words[i + k])
            break

    return coords_list


def detect_pii(lines_map: Dict[int, Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Detect PII from OCR lines.

    Returns:
        pii_log: list of dicts {type, value, line}
        redaction_boxes: list of word-box dicts for redaction
    """
    redaction_boxes: List[Dict] = []
    pii_log: List[Dict] = []

    # 1) Regex-based PII (phone, date, UHID/IPD)
    for line_num, data in lines_map.items():
        full_text = data["text"]
        line_lower = full_text.lower()

        # Run regexes on full line to catch spaced numbers
        for pattern, pii_type in [
            (PHONE_RE, "phone"),
            (DATE_RE, "date"),
            (UHID_RE, "uhid_number"),
            (IPD_RE, "ipd_number"),
        ]:
            for match in pattern.finditer(full_text):
                value = match.group()
                coords = extract_value_coordinates(data, value)
                if coords:
                    redaction_boxes.extend(coords)
                pii_log.append({
                    "type": pii_type,
                    "value": value,
                    "line": line_num,
                })

        # 2) Label-based fields (patient_name, age, etc.)
        for field, labels in KEYWORDS.items():
            for label in labels:
                idx = line_lower.find(label)
                if idx == -1:
                    continue

                start_val = idx + len(label)
                end_val = cut_at_next_keyword(line_lower, start_val)
                if end_val is None:
                    end_val = len(line_lower)

                raw_val = full_text[start_val:end_val]
                value_clean = clean_ocr_text(raw_val)
                if not value_clean:
                    continue

                coords = extract_value_coordinates(data, value_clean)
                if coords:
                    redaction_boxes.extend(coords)

                pii_log.append({
                    "type": field,
                    "value": value_clean,
                    "line": line_num,
                })
                # Only use first matching label per line for this field
                break

    return pii_log, redaction_boxes


# ----------------------------
# Redaction / Double-page Handling
# ----------------------------

def redact_image(img: np.ndarray, boxes: List[Dict]) -> np.ndarray:
    """Draw black rectangles over detected PII boxes."""
    redacted_img = img.copy()
    for b in boxes:
        x, y, w, h = b['x'], b['y'], b['w'], b['h']
        padding = 3
        cv2.rectangle(
            redacted_img,
            (max(0, x - padding), max(0, y - padding)),
            (x + w + padding, y + h + padding),
            (0, 0, 0),
            -1
        )
    return redacted_img


def split_double_page(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split double-page scan into left and right pages (vertical half)."""
    h, w = img.shape[:2]
    mid = w // 2
    left = img[:, :mid]
    right = img[:, mid:]
    return left, right


# ----------------------------
# Main Pipeline
# ----------------------------

def run_pipeline(image_path: str, output_redacted_path: Optional[str] = None) -> Dict[str, Any]:
    """Run full OCR + PII extraction pipeline on a single JPEG."""
    img, preprocessed_list = preprocess_image_advanced(image_path)

    h, w = img.shape[:2]
    mid = w // 2
    is_double_page = w > 1.4 * h  # heuristic

    pages_to_process = []
    if is_double_page:
        left, right = split_double_page(img)
        left_pre = [p[:, :p.shape[1] // 2] for p in preprocessed_list]
        right_pre = [p[:, p.shape[1] // 2:] for p in preprocessed_list]
        pages_to_process.append(("left", left, left_pre, 0))
        pages_to_process.append(("right", right, right_pre, mid))
    else:
        pages_to_process.append(("single", img, preprocessed_list, 0))

    all_pii: List[Dict] = []
    all_boxes: List[Dict] = []
    all_text_parts: List[str] = []

    for page_name, page_img, page_preprocessed, x_offset in pages_to_process:
        lines_map = run_ocr_multi_easyocr(page_preprocessed)

        pii_found, boxes = detect_pii(lines_map)

        for pii in pii_found:
            pii["page"] = page_name

        all_pii.extend(pii_found)

        # Adjust boxes for double-page right side
        for b in boxes:
            b_adj = dict(b)
            b_adj["x"] = b["x"] + x_offset
            all_boxes.append(b_adj)

        page_text = "\n".join(
            [v["text"].strip() for v in lines_map.values() if v["text"].strip()]
        )
        all_text_parts.append(f"=== {page_name.upper()} PAGE ===\n{page_text}")

    redacted_path = None
    if output_redacted_path and all_boxes:
        final_img = redact_image(img, all_boxes)
        cv2.imwrite(output_redacted_path, final_img)
        redacted_path = output_redacted_path

    # Deduplicate PII items by (type, value)
    unique_pii: List[Dict] = []
    seen_values = set()
    for pii in all_pii:
        key = (pii["type"], pii["value"].lower())
        if key not in seen_values:
            seen_values.add(key)
            unique_pii.append(pii)

    return {
        "full_text": "\n\n".join(all_text_parts),
        "pii_items": unique_pii,
        "redacted_path": redacted_path,
        "is_double_page": is_double_page,
    }


# ----------------------------
# CLI Entrypoint
# ----------------------------

def main():
    if len(sys.argv) < 2:
        print(json.dumps({
            "error": "Usage: python ocr_pipeline_easyocr.py <image_path> [output_redacted_path]"
        }))
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        result = run_pipeline(image_path, output_path)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
