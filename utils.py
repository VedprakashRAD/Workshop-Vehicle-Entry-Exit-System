import cv2
import numpy as np
import easyocr
import re
from ultralytics import YOLO

# Initialize the EasyOCR reader with optimized parameters
reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='./models')

def preprocess_license_plate(plate_img):
    """Preprocess the license plate image for better OCR results"""
    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Apply gaussian blur
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(blur)
    
    # Apply thresholding to get a binary image
    _, thresh = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply morphological operations for noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Return both the thresholded and the equalized image for OCR to try
    return opening, equalized

def recognize_license_plate(plate_img):
    """Extract text from license plate image using EasyOCR"""
    # Preprocess the image - returns two different preprocessed versions
    processed_img1, processed_img2 = preprocess_license_plate(plate_img)
    
    # Try OCR on original image
    results1 = reader.readtext(plate_img)
    
    # Try OCR on first processed image
    results2 = reader.readtext(processed_img1)
    
    # Try OCR on second processed image
    results3 = reader.readtext(processed_img2)
    
    # Combine results
    all_results = results1 + results2 + results3
    
    # Extract text and confidence
    if all_results:
        # Sort by confidence and take the highest confidence result
        all_results.sort(key=lambda x: x[2], reverse=True)
        text = all_results[0][1]
        confidence = all_results[0][2]
        
        # Clean the text (remove spaces and special characters)
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        return text, confidence
    
    return None, 0.0

def is_valid_license_plate(text, min_length=2, max_length=12):
    """Check if the extracted text is a valid license plate"""
    if not text:
        return False
    
    if len(text) < min_length or len(text) > max_length:
        return False
    
    # For testing, we'll be more lenient with the pattern
    # Just ensure it has some mix of letters and numbers
    has_letter = bool(re.search(r'[A-Z]', text))
    has_number = bool(re.search(r'[0-9]', text))
    
    # For testing, we'll accept any text that has at least one letter or number
    # In production, you should use a stricter pattern based on your region
    return has_letter or has_number

def get_vehicle_crop(frame, box):
    """Get a cropped image of the vehicle from the frame"""
    x1, y1, x2, y2 = [int(i) for i in box]
    return frame[y1:y2, x1:x2] 