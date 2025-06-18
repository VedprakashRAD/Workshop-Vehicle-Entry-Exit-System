import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
from utils import recognize_license_plate, is_valid_license_plate, get_vehicle_crop

class VehicleDetector:
    def __init__(self, model_dir='models'):
        """Initialize the vehicle detector with YOLOv8 models"""
        # Create models directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Load vehicle detection model - using a more efficient approach
        print("Loading YOLOv8 model for vehicle-only detection...")
        self.vehicle_model = YOLO('yolov8n.pt')  # Using YOLOv8 nano model
        
        # Load license plate detection model
        license_plate_model_path = os.path.join(model_dir, 'license_plate_detector.pt')
        
        # Use the specialized license plate detector model
        if os.path.exists(license_plate_model_path):
            try:
                self.license_plate_model = YOLO(license_plate_model_path)
                print("Using specialized license plate detector model")
            except Exception as e:
                print(f"Error loading license plate model: {e}")
                self.license_plate_model = self.vehicle_model
                print("Falling back to general model for license plate detection")
        else:
            self.license_plate_model = self.vehicle_model
            print("License plate model not found, using general model")
            
        # Vehicle classes of interest (from COCO dataset)
        # Focusing only on actual vehicles (removing person, boat, traffic light)
        self.vehicle_classes = {
            2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'
        }
        
        # Confidence thresholds - increase vehicle confidence for faster detection
        self.vehicle_conf = 0.35  # Higher threshold for more confident and faster detection
        self.plate_conf = 0.25  # Increased threshold for license plates to reduce false positives
        
        # Track detected vehicles and plates
        self.detected_vehicles = {}
        self.detected_plates = {}
        self.last_cleanup = time.time()
        self.cleanup_interval = 30  # 30 seconds
    
    def detect_vehicles_and_plates(self, frame):
        """Detect vehicles and license plates in a frame, with focus on plates"""
        if frame is None:
            return [], []
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Only detect vehicles once with optimized settings
        vehicle_detections = []
        
        try:
            # Use faster inference settings and filter for vehicle classes only (car, truck, bus, motorcycle)
            # Classes 2 (car), 3 (motorcycle), 5 (bus), 7 (truck) only
            vehicle_classes_filter = [2, 3, 5, 7]  # Only vehicle classes for faster detection
            
            # Run YOLOv8 with class filtering to only detect vehicles, not other objects
            vehicle_results = self.vehicle_model(
                frame, 
                conf=self.vehicle_conf, 
                iou=0.5, 
                classes=vehicle_classes_filter  # Only detect vehicle classes
            )[0]
            
            # Process vehicle detections
            for box in vehicle_results.boxes:
                cls = int(box.cls.item())
                # Skip filters here since we already filtered in the YOLO call
                if cls in self.vehicle_classes:
                    x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                    confidence = box.conf.item()
                    vehicle_detections.append({
                        'box': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class': self.vehicle_classes[cls]
                    })
        except Exception as e:
            # Log error and return empty lists
            print(f"Error detecting vehicles: {e}")
            return [], []
        
        # Fast approach for license plate detection - only process if vehicles detected
        plate_detections = []
        
        # Skip license plate detection if no vehicles detected
        if not vehicle_detections:
            return vehicle_detections, plate_detections
            
        # Approach 1: Use specialized license plate detector only on regions with vehicles
        # This is much faster than scanning the whole frame
        for vehicle in vehicle_detections:
            v_x1, v_y1, v_x2, v_y2 = map(int, vehicle['box'])
            
            # Extract the vehicle region with padding
            padding = 20  # Add some padding around vehicle
            y_start = max(0, v_y1 - padding)
            y_end = min(frame.shape[0], v_y2 + padding)
            x_start = max(0, v_x1 - padding)
            x_end = min(frame.shape[1], v_x2 + padding)
            
            vehicle_region = frame[y_start:y_end, x_start:x_end]
            
            # Only process if the region is valid and big enough
            if vehicle_region.size > 1000:  # Skip tiny regions
                # Use specialized license plate model on the vehicle region
                try:
                    plate_results = self.license_plate_model(vehicle_region, conf=self.plate_conf)[0]
                    
                    # Process each detected plate
                    for box in plate_results.boxes:
                        # Get coordinates relative to vehicle region
                        rx1, ry1, rx2, ry2 = box.xyxy.cpu().numpy()[0]
                        
                        # Convert back to full frame coordinates
                        x1, y1 = rx1 + x_start, ry1 + y_start
                        x2, y2 = rx2 + x_start, ry2 + y_start
                        confidence = box.conf.item()
                        
                        # Process this region as a potential plate
                        self._process_potential_plate_region(frame, x1, y1, x2, y2, confidence, plate_detections)
                except Exception as e:
                    # Continue to the fallback approach if this fails
                    pass
                
                # Fallback: Focus on the area where license plates typically appear
                # For cars: Front/back center, for trucks/buses: front center or sides
                # We'll create 1-2 regions of interest based on vehicle class
                vehicle_class = vehicle['class']
                regions = []
                
                height, width = vehicle_region.shape[:2]
                
                if vehicle_class in ['car', 'motorcycle']:
                    # For cars - check front/back center regions
                    # Front license plate region (bottom center)
                    regions.append((
                        width//4, height*2//3,  # Top-left
                        width*3//4, height      # Bottom-right
                    ))
                    
                elif vehicle_class in ['bus', 'truck']:
                    # For larger vehicles - check multiple regions
                    # Front plate (bottom center)
                    regions.append((
                        width//4, height*2//3,
                        width*3//4, height
                    ))
                    # Side plate (middle center)
                    regions.append((
                        width//3, height//3,
                        width*2//3, height*2//3
                    ))
                
                # Process each region
                for rx1, ry1, rx2, ry2 in regions:
                    # Convert to full frame coordinates
                    x1, y1 = rx1 + x_start, ry1 + y_start
                    x2, y2 = rx2 + x_start, ry2 + y_start
                    
                    # Process this region as a potential plate
                    self._process_potential_plate_region(frame, x1, y1, x2, y2, 0.5, plate_detections)
        
        # Associate plates with vehicles
        self._associate_plates_with_vehicles(vehicle_detections, plate_detections)
        
        # Cleanup old detections
        self._cleanup_old_detections()
        
        return vehicle_detections, plate_detections
    
    def _associate_plates_with_vehicles(self, vehicles, plates):
        """Associate license plates with vehicles based on position"""
        current_time = time.time()
        
        for vehicle in vehicles:
            v_x1, v_y1, v_x2, v_y2 = vehicle['box']
            vehicle_id = f"{v_x1:.0f}_{v_y1:.0f}_{v_x2:.0f}_{v_y2:.0f}"
            
            # Update vehicle tracking
            self.detected_vehicles[vehicle_id] = {
                'box': vehicle['box'],
                'class': vehicle['class'],
                'last_seen': current_time,
                'plates': []
            }
            
            # Find plates that belong to this vehicle
            for plate in plates:
                p_x1, p_y1, p_x2, p_y2 = plate['box']
                
                # Check if plate is inside or near the vehicle
                if (v_x1 <= p_x1 and p_x2 <= v_x2 and v_y1 <= p_y1 and p_y2 <= v_y2) or \
                   (abs(p_x1 - v_x1) < 0.2 * (v_x2 - v_x1) and abs(p_y1 - v_y1) < 0.2 * (v_y2 - v_y1)):
                    
                    plate_id = plate['text']
                    
                    # Update plate tracking
                    if plate_id not in self.detected_plates:
                        self.detected_plates[plate_id] = {
                            'text': plate['text'],
                            'confidence': plate['text_confidence'],
                            'last_seen': current_time,
                            'vehicle_id': vehicle_id,
                            'count': 1
                        }
                    else:
                        # Update existing plate data
                        self.detected_plates[plate_id]['last_seen'] = current_time
                        self.detected_plates[plate_id]['count'] += 1
                        
                        # Update confidence if higher
                        if plate['text_confidence'] > self.detected_plates[plate_id]['confidence']:
                            self.detected_plates[plate_id]['confidence'] = plate['text_confidence']
                    
                    # Add to vehicle's plates
                    if plate_id not in self.detected_vehicles[vehicle_id]['plates']:
                        self.detected_vehicles[vehicle_id]['plates'].append(plate_id)
    
    def _cleanup_old_detections(self):
        """Remove old vehicle and plate detections"""
        current_time = time.time()
        
        # Only run cleanup occasionally
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        self.last_cleanup = current_time
        timeout = 60  # Remove after 60 seconds of not being seen
        
        # Remove old vehicles
        for vehicle_id in list(self.detected_vehicles.keys()):
            if current_time - self.detected_vehicles[vehicle_id]['last_seen'] > timeout:
                del self.detected_vehicles[vehicle_id]
        
        # Remove old plates
        for plate_id in list(self.detected_plates.keys()):
            if current_time - self.detected_plates[plate_id]['last_seen'] > timeout:
                del self.detected_plates[plate_id]
    
    def draw_detections(self, frame, vehicles, plates):
        """Draw only license plates on the frame"""
        if frame is None:
            return frame
        
        # Create a copy of the frame to avoid modifying the original
        result = frame.copy()
        
        # Only draw license plates
        for plate in plates:
            x1, y1, x2, y2 = map(int, plate['box'])
            text = plate['text']
            confidence = plate['text_confidence']
            
            # Draw the bounding box with thicker lines
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Create a filled background for the text
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            cv2.rectangle(result, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), (0, 0, 255), -1)
            
            # Draw the license plate number in larger font
            cv2.putText(result, text, (x1 + 5, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        return result
    
    def get_recent_plate_detections(self, max_plates=5):
        """Get the most recent license plate detections"""
        # Sort plates by last_seen time (most recent first)
        sorted_plates = sorted(
            self.detected_plates.items(), 
            key=lambda item: (item[1]['last_seen'], item[1]['confidence']), 
            reverse=True
        )
        
        # Return the top plates
        return [
            {
                'plate_number': plate_id,
                'confidence': data['confidence'],
                'vehicle_type': self.detected_vehicles.get(data['vehicle_id'], {}).get('class', 'unknown'),
                'last_seen': data['last_seen']
            }
            for plate_id, data in sorted_plates[:max_plates]
        ]
    
    def _process_potential_plate_region(self, frame, x1, y1, x2, y2, confidence, plate_detections):
        """Process a region as a potential license plate - optimized for speed"""
        # Convert to int coordinates and ensure within frame bounds
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(width, int(x2)), min(height, int(y2))
        
        # Calculate width and height
        region_width = x2 - x1
        region_height = y2 - y1
        
        # Skip if region is too small or dimensions make no sense
        if region_width < 20 or region_height < 10 or x1 >= x2 or y1 >= y2:
            return
            
        # Quick aspect ratio check (license plates are typically rectangular)
        aspect_ratio = region_width / region_height if region_height > 0 else 0
        if not (1.5 < aspect_ratio < 5.0):  # More strict criteria for better performance
            return
        
        # Extract the potential license plate image
        plate_img = frame[y1:y2, x1:x2]
        if plate_img.size == 0:
            return
        
        # Fast resize if needed - always resize to a standard size for faster processing
        standard_width = 200
        try:
            plate_img = cv2.resize(plate_img, (standard_width, int(standard_width / aspect_ratio)), 
                              interpolation=cv2.INTER_AREA)  # INTER_AREA is faster and good for downsampling
        except Exception:
            return
        
        try:
            # Recognize the license plate text
            plate_text, text_confidence = recognize_license_plate(plate_img)
            
            # Skip if text recognition failed or confidence is too low
            if not plate_text or text_confidence < 0.15:
                return
            
            # Quick validation
            if is_valid_license_plate(plate_text):
                # Simple duplicate check
                for plate in plate_detections:
                    if plate['text'] == plate_text:
                        # Update if confidence is higher
                        if text_confidence > plate['text_confidence']:
                            plate['box'] = [x1, y1, x2, y2]
                            plate['confidence'] = confidence
                            plate['text_confidence'] = text_confidence
                        return  # Exit early
                
                # Add new detection
                plate_detections.append({
                    'box': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'text': plate_text,
                    'text_confidence': text_confidence
                })
        except Exception:
            # Just continue if there's an error
            pass 