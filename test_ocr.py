"""
Test script for OCR functionality.
Run this file to test the OCR engine with your chemical label image.
Place this file in the same directory as ocr_engine.py for now.
"""
import cv2
import pytesseract
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional
import os


class OCREngine:
    """Handles OCR processing for chemical label images."""
    
    def __init__(self, tesseract_path: Optional[str] = None):
        """Initialize OCR engine."""
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Test if tesseract is available
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            raise RuntimeError(f"Tesseract not found. Please install tesseract-ocr: {e}")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image to improve OCR accuracy."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Apply threshold to get black text on white background
        _, threshold = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return threshold
    
    def extract_text(self, image_path: str, preprocess: bool = True) -> Dict[str, any]:
        """Extract text from image using OCR."""
        try:
            if preprocess:
                processed_image = self.preprocess_image(image_path)
                # Convert back to PIL Image for tesseract
                pil_image = Image.fromarray(processed_image)
            else:
                pil_image = Image.open(image_path)
            
            # Configure tesseract for better chemical text recognition
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-.,()[]%°'
            
            # Extract text with confidence scores
            ocr_data = pytesseract.image_to_data(pil_image, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Extract raw text
            raw_text = pytesseract.image_to_string(pil_image, config=custom_config)
            
            # Calculate average confidence
            confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Extract words with high confidence
            high_confidence_words = []
            for i, word in enumerate(ocr_data['text']):
                if word.strip() and int(ocr_data['conf'][i]) > 30:  # Confidence threshold
                    high_confidence_words.append({
                        'text': word.strip(),
                        'confidence': int(ocr_data['conf'][i]),
                        'bbox': {
                            'x': ocr_data['left'][i],
                            'y': ocr_data['top'][i],
                            'width': ocr_data['width'][i],
                            'height': ocr_data['height'][i]
                        }
                    })
            
            return {
                'raw_text': raw_text.strip(),
                'average_confidence': round(avg_confidence, 2),
                'word_count': len([w for w in raw_text.split() if w.strip()]),
                'high_confidence_words': high_confidence_words,
                'preprocessing_applied': preprocess,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'raw_text': '',
                'average_confidence': 0,
                'word_count': 0,
                'high_confidence_words': [],
                'preprocessing_applied': preprocess,
                'success': False,
                'error': str(e)
            }


def test_ocr():
    """Test the OCR engine with a sample image."""
    
    # Initialize OCR engine
    try:
        ocr = OCREngine()
        print("✅ OCR Engine initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing OCR: {e}")
        print("Make sure Tesseract is installed:")
        print("Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print("Mac: brew install tesseract")
        print("Linux: sudo apt install tesseract-ocr")
        return
    
    # Test image path - update this to your actual image
    image_path = r"C:\Users\GabeShavitt\Downloads\LabChemicalsHPLCRoom\PXL_20250824_081230419.jpg"
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("Please update the image_path variable with the correct path to your image.")
        return
    
    print(f"🔍 Processing image: {image_path}")
    
    # Extract text from image
    result = ocr.extract_text(image_path)
    
    if result['success']:
        print("✅ OCR completed successfully!")
        print(f"📊 Average confidence: {result['average_confidence']}%")
        print(f"📝 Word count: {result['word_count']}")
        print("\n📄 Extracted text:")
        print("-" * 50)
        print(result['raw_text'])
        print("-" * 50)
        
        if result['high_confidence_words']:
            print(f"\n🎯 High confidence words ({len(result['high_confidence_words'])}):")
            for word_data in result['high_confidence_words'][:10]:  # Show first 10
                print(f"  '{word_data['text']}' (confidence: {word_data['confidence']}%)")
    else:
        print(f"❌ OCR failed: {result['error']}")


if __name__ == "__main__":
    test_ocr()