import cv2
import numpy as np
import os

def generate_license_plate(text="AB12CD3456"):
    """Generate a simple license plate image with the given text"""
    # Create a blank image (white background)
    width, height = 400, 150
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw a black border
    cv2.rectangle(image, (10, 10), (width - 10, height - 10), (0, 0, 0), 2)
    
    # Add the license plate text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 3
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    
    # Calculate position to center the text
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    
    # Draw the text
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)
    
    return image

def generate_car_with_plate(plate_text="AB12CD3456"):
    """Generate a simple car image with a license plate"""
    # Create a larger image (car)
    width, height = 800, 600
    car_image = np.ones((height, width, 3), dtype=np.uint8) * 200
    
    # Draw a car shape (simplified)
    car_color = (120, 120, 200)  # Light red
    cv2.rectangle(car_image, (100, 200), (700, 500), car_color, -1)
    cv2.rectangle(car_image, (150, 100), (650, 200), car_color, -1)
    
    # Add wheels
    cv2.circle(car_image, (200, 500), 50, (50, 50, 50), -1)
    cv2.circle(car_image, (600, 500), 50, (50, 50, 50), -1)
    
    # Generate license plate
    plate_img = generate_license_plate(plate_text)
    plate_height, plate_width = plate_img.shape[:2]
    
    # Position for the plate on the car
    plate_x = (width - plate_width) // 2
    plate_y = 400
    
    # Place the plate on the car
    car_image[plate_y:plate_y + plate_height, plate_x:plate_x + plate_width] = plate_img
    
    return car_image

def main():
    """Generate test images with license plates"""
    os.makedirs('test_data', exist_ok=True)
    
    # Generate a few test images with different license plates
    plate_texts = ["KL01AQ1", "MH02AB1234", "DL3CAB5678", "TN07CD9012"]
    
    for idx, text in enumerate(plate_texts):
        # Generate car with plate
        car_image = generate_car_with_plate(text)
        
        # Save the image
        output_path = os.path.join('test_data', f"test_car_{idx+1}.jpg")
        cv2.imwrite(output_path, car_image)
        print(f"Generated test image: {output_path} with plate {text}")
    
    print("\nTo test license plate detection, use:")
    print("  python test_image.py test_data/test_car_1.jpg")

if __name__ == "__main__":
    main() 