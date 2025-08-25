"""
OCR engine for extracting text from chemical labels.
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
        """
        Initialize OCR engine.
        
        Args:
            tesseract_path: Path to tesseract executable (if not in PATH)
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Test if tesseract is available
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            raise RuntimeError(f"Tesseract not found. Please install tesseract-ocr: {e}")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image to improve OCR accuracy.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array
        """
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
        """
        Extract text from image using OCR.
        
        Args:
            image_path: Path to the image file
            preprocess: Whether to apply image preprocessing
            
        Returns:
            Dictionary containing extracted text and metadata
        """
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
    
    def save_debug_image(self, image_path: str, output_path: str):
        """Save preprocessed image for debugging purposes."""
        processed = self.preprocess_image(image_path)
        cv2.imwrite(output_path, processed)


# Example usage and testing
if __name__ == "__main__":
    # Initialize OCR engine
    ocr = OCREngine()
    
    # Test with a sample image (you'll need to provide an actual image path)
    # result = ocr.extract_text("path/to/chemical_label.jpg")
    # print(f"OCR Result: {result}")
    
    print("OCR Engine initialized successfully!")
    print("To test, provide an image path in the main section.")