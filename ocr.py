# ocr.py

import logging
import cv2
import numpy as np
import pytesseract
import fitz  # PyMuPDF
from PIL import Image

from normalizer import normalize_scientific_text

# 400 DPI (up from 300) gives small sub/superscript glyphs (e.g. "10−3", "Li+") enough pixel
# height for Tesseract's LSTM engine to resolve, at the cost of slower rendering.
_OCR_DPI = 400


class ImagePreprocessor:
    """Handles OpenCV image enhancements to maximize OCR accuracy on scanned scientific papers."""

    @staticmethod
    def process(pixmap: fitz.Pixmap) -> np.ndarray:
        # 1. Convert PyMuPDF Pixmap to OpenCV image array via PNG bytes
        img_bytes = pixmap.tobytes("png")
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # 2. Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. Denoising (Removes grain/noise from older scanned papers)
        denoised = cv2.fastNlMeansDenoising(gray, h=30)

        # 4. Mild unsharp mask: sharpens fine strokes (subscripts/superscripts/Greek letters)
        # that denoising can soften, without amplifying background grain.
        blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=3)
        sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)

        # 5. Adaptive Thresholding (Handles uneven lighting/shadows in scanned pages)
        # Binarizes the image: text becomes perfectly black, background perfectly white
        thresh = cv2.adaptiveThreshold(
            sharpened, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 4
        )

        return thresh


def run_ocr(fitz_page: fitz.Page) -> str:
    """
    Renders a PDF page to an image, preprocesses it, and extracts text using Tesseract.
    Returns "NONE" if extraction fails, adhering to the strict default rule.

    Note (schema section 13, known limitation): this is a best-effort improvement (higher DPI,
    sharpening, unicode normalization shared with the rest of the pipeline) — true equation/Greek
    OCR accuracy is inherently limited without installing Tesseract's dedicated 'equ' trained data,
    which this pipeline does not assume is present.
    """
    try:
        mat = fitz.Matrix(_OCR_DPI / 72, _OCR_DPI / 72)
        pix = fitz_page.get_pixmap(matrix=mat, alpha=False)

        # Apply OpenCV preprocessing pipeline
        processed_img_array = ImagePreprocessor.process(pix)

        # Convert OpenCV array back to PIL Image for Tesseract
        pil_img = Image.fromarray(processed_img_array)

        # Tesseract Configuration for Scientific Papers:
        # --oem 3 : Use Default (LSTM) Neural Net OCR Engine
        # --psm 3 : Fully automatic page segmentation (handles multi-column papers well)
        # -l eng  : English language (Add '+equ' if the equations trained-data pack is installed)
        custom_config = r'--oem 3 --psm 3 -l eng'

        text = pytesseract.image_to_string(pil_img, config=custom_config)
        text = normalize_scientific_text(text).strip()

        return text if text else "NONE"

    except Exception as e:
        logging.warning(f"OCR Pipeline failed on page {fitz_page.number}: {str(e)}")
        return "NONE"