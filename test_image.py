import cv2
import os
import sys
import numpy as np
from vehicle_detector import VehicleDetector
from rto_api import RTOApiService

def test_with_image(image_path):
    """Test the vehicle detection and license plate recognition with a single image"""
    # Initialize the detector
    detector = VehicleDetector()
    
    # Initialize the RTO API service
    rto_api = RTOApiService()
    
    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return
    
    # Open the image file
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open image file '{image_path}'.")
        return
    
    print(f"Image loaded: {image.shape[1]}x{image.shape[0]}")
    
    # Detect vehicles and license plates
    vehicle_detections, plate_detections = detector.detect_vehicles_and_plates(image)
    
    # Draw detections on the image
    processed_image = detector.draw_detections(image, vehicle_detections, plate_detections)
    
    # Print detection results
    print(f"Detected {len(vehicle_detections)} vehicles:")
    for i, vehicle in enumerate(vehicle_detections):
        print(f"  Vehicle {i+1}: {vehicle['class']} (Confidence: {vehicle['confidence']:.2f})")
    
    print(f"\nDetected {len(plate_detections)} license plates:")
    for i, plate in enumerate(plate_detections):
        print(f"  Plate {i+1}: {plate['text']} (Confidence: {plate['text_confidence']:.2f})")
        
        # Get vehicle details (mock data for demo)
        vehicle_details = rto_api.get_mock_vehicle_details(plate['text'])
        print(f"    Vehicle: {vehicle_details['manufacturer']} {vehicle_details['model']}")
        print(f"    Registration: {vehicle_details['registration']['status']}")
    
    # Save output image
    output_path = 'output_' + os.path.basename(image_path)
    cv2.imwrite(output_path, processed_image)
    print(f"\nOutput saved to '{output_path}'")
    
    # Display the image
    cv2.imshow('Vehicle Detection', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Use command line argument as image path or default to a sample image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "sample_image.jpg"
        print(f"No image file specified. Using default: {image_path}")
        print("Usage: python test_image.py path/to/image.jpg")
    
    test_with_image(image_path) 