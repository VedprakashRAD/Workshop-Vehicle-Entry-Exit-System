import os
import requests
import shutil
import urllib.request

def download_test_image():
    """Download a test image with a license plate for testing"""
    print("Downloading test image with license plate...")
    
    # URL for a sample license plate image
    test_image_url = "https://commons.wikimedia.org/wiki/Special:FilePath/KL-01-AQ-1.jpg"
    
    # Local path to save the image
    test_image_path = "sample_license_plate.jpg"
    
    try:
        # Create a directory for test data if it doesn't exist
        os.makedirs('test_data', exist_ok=True)
        
        # Download the image
        urllib.request.urlretrieve(test_image_url, os.path.join('test_data', test_image_path))
        
        print(f"Test image downloaded successfully to: test_data/{test_image_path}")
        print("\nTo test license plate detection, use:")
        print(f"  python test_image.py test_data/{test_image_path}")
        
    except Exception as e:
        print(f"Error downloading test image: {e}")
        print("You can try manually downloading an image with a license plate for testing.")

if __name__ == "__main__":
    download_test_image() 