"""
Test script for OCR functionality.
Run this file to test the OCR engine with your chemical label image.
"""
import sys
import os

# Add src directory to path so we can import our modules
sys.path.append('src')

from image_processor.ocr_engine import OCREngine


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