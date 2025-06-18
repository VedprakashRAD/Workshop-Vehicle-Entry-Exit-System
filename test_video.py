import os
import cv2
import time
import sys
from vehicle_detector import VehicleDetector
from rto_api import RTOApiService

def test_with_video(video_path):
    """Test the vehicle detection and license plate recognition with a video file"""
    # Initialize the detector
    detector = VehicleDetector()
    
    # Initialize the RTO API service
    rto_api = RTOApiService()
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video loaded: {width}x{height} at {fps} FPS")
    
    # Create output video writer
    output_path = 'output_' + os.path.basename(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process the video
    frame_count = 0
    detected_plates = set()
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 5th frame to speed up testing
        if frame_count % 5 == 0:
            # Detect vehicles and license plates
            vehicle_detections, plate_detections = detector.detect_vehicles_and_plates(frame)
            
            # Draw detections on the frame
            processed_frame = detector.draw_detections(frame, vehicle_detections, plate_detections)
            
            # Collect detected plates
            for plate in plate_detections:
                plate_text = plate['text']
                if plate_text and plate_text not in detected_plates:
                    detected_plates.add(plate_text)
                    print(f"New plate detected: {plate_text} (Confidence: {plate['text_confidence']:.2f})")
                    
                    # Get vehicle details (mock data for demo)
                    vehicle_details = rto_api.get_mock_vehicle_details(plate_text)
                    print(f"  Vehicle: {vehicle_details['manufacturer']} {vehicle_details['model']}")
                    print(f"  Registration: {vehicle_details['registration']['status']}")
                    print()
        else:
            processed_frame = frame
        
        # Display the frame (resize if too large)
        display_height = 600
        if height > display_height:
            display_width = int(width * (display_height / height))
            display_frame = cv2.resize(processed_frame, (display_width, display_height))
        else:
            display_frame = processed_frame
            
        cv2.imshow('Vehicle Detection', display_frame)
        
        # Write the processed frame to output video
        out.write(processed_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_count += 1
        
        # Print progress every 100 frames
        if frame_count % 100 == 0:
            elapsed_time = time.time() - start_time
            fps_processed = frame_count / elapsed_time if elapsed_time > 0 else 0
            print(f"Processed {frame_count} frames ({fps_processed:.2f} FPS)")
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Video processing completed. Output saved to '{output_path}'")
    print(f"Detected {len(detected_plates)} unique license plates:")
    for plate in detected_plates:
        print(f"  - {plate}")

if __name__ == "__main__":
    # Use command line argument as video path or default to a sample video
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "sample_video.mp4"
        print(f"No video file specified. Using default: {video_path}")
        print("Usage: python test_video.py path/to/video.mp4")
    
    test_with_video(video_path) 