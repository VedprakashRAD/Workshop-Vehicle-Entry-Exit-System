import os
import cv2
import time
import json
from flask import Flask, render_template, Response, request, jsonify
import threading
from dotenv import load_dotenv
from vehicle_detector import VehicleDetector
from rto_api import RTOApiService

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize the vehicle detector
detector = VehicleDetector()

# Initialize the RTO API service
rto_api = RTOApiService()

# Global variables for video stream
# Examples for video_source:
# - 0: Use webcam
# - 'rtsp://username:password@192.168.1.64:554/stream': Use IP camera with RTSP
# - 'http://192.168.1.64/video': Use IP camera with HTTP stream
# - 'video_file.mp4': Use video file
video_source = 0  # Change this to your camera URL or video file path
cap = None
frame_lock = threading.Lock()
current_frame = None
processing = False
last_processed = None
process_interval = 0.5  # Process every 0.5 seconds to reduce CPU usage

def initialize_camera():
    """Initialize the video capture device"""
    global cap
    try:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return False
        return True
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return False

def read_frames():
    """Read frames from the video source continuously"""
    global cap, current_frame, processing, last_processed
    
    if not initialize_camera():
        return
    
    while True:
        try:
            ret, frame = cap.read()
            
            if not ret:
                print("End of video stream or error reading frame")
                # Try to reinitialize camera
                if not initialize_camera():
                    time.sleep(1)  # Wait before trying again
                    continue
                else:
                    continue  # Try reading again with reinitialized camera
            
            # Update the current frame
            with frame_lock:
                current_frame = frame.copy()
            
            # Process frame for vehicle detection periodically
            current_time = time.time()
            if not processing and (last_processed is None or current_time - last_processed > process_interval):
                processing = True
                threading.Thread(target=process_frame, args=(frame.copy(),)).start()
                last_processed = current_time
                
            # Sleep to reduce CPU usage
            time.sleep(0.03)  # ~30 FPS
            
        except Exception as e:
            print(f"Error reading video frame: {e}")
            time.sleep(1)  # Wait before trying again

def process_frame(frame):
    """Process a frame to detect vehicles and license plates"""
    global processing
    
    try:
        # Detect vehicles and plates
        vehicle_detections, plate_detections = detector.detect_vehicles_and_plates(frame)
        
        # Draw the detections on the frame
        processed_frame = detector.draw_detections(frame, vehicle_detections, plate_detections)
        
        # Update the current frame with the processed one
        with frame_lock:
            current_frame = processed_frame
            
    except Exception as e:
        print(f"Error processing frame: {e}")
    finally:
        processing = False

def generate_frames():
    """Generate frames for video streaming"""
    while True:
        # Wait until a frame is available
        if current_frame is None:
            time.sleep(0.1)
            continue
            
        # Get the current frame with a lock
        with frame_lock:
            frame_to_send = current_frame.copy()
        
        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame_to_send)
        
        if not ret:
            continue
            
        # Convert to bytes and yield
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/detected_plates')
def detected_plates():
    """API endpoint to get detected license plates"""
    plates = detector.get_recent_plate_detections()
    return jsonify(plates)

@app.route('/api/vehicle_details')
def vehicle_details():
    """API endpoint to get vehicle details by license plate"""
    plate = request.args.get('plate', '')
    
    if not plate:
        return jsonify({"error": "No license plate provided"}), 400
    
    # Get vehicle details from RTO API (using mock data for demo)
    details = rto_api.get_mock_vehicle_details(plate)
    
    if details:
        return jsonify(details)
    else:
        return jsonify({"error": "Vehicle details not found"}), 404

def create_templates_folder():
    """Create templates folder and HTML files"""
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)

if __name__ == '__main__':
    # Create necessary folders
    create_templates_folder()
    
    # Start the frame reading thread
    video_thread = threading.Thread(target=read_frames)
    video_thread.daemon = True
    video_thread.start()
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=9000, debug=False) 