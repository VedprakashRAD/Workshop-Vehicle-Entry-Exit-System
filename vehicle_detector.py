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
        
        # Load vehicle detection model
        self.vehicle_model = YOLO('yolov8n.pt')  # Using YOLOv8 nano model
        
        # Load license plate detection model
        license_plate_model_path = os.path.join(model_dir, 'license_plate_detector.pt')
        
        # Download license plate model if it doesn't exist (placeholder for actual fine-tuned model)
        if not os.path.exists(license_plate_model_path):
            # In real implementation, download or copy the fine-tuned model
            # For this example, we'll use the same general model
            self.license_plate_model = self.vehicle_model
            print("Using general model for license plate detection")
        else:
            self.license_plate_model = YOLO(license_plate_model_path)
            
        # Vehicle classes of interest (from COCO dataset)
        self.vehicle_classes = {
            2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 
            8: 'boat', 9: 'traffic light', 0: 'person'
        }
        
        # Confidence thresholds
        self.vehicle_conf = 0.25  # Lower threshold to detect more vehicles
        self.plate_conf = 0.15  # Lower threshold for license plates
        
        # Track detected vehicles and plates
        self.detected_vehicles = {}
        self.detected_plates = {}
        self.last_cleanup = time.time()
        self.cleanup_interval = 30  # 30 seconds
    
    def detect_vehicles_and_plates(self, frame):
        """Detect vehicles and license plates in a frame"""
        if frame is None:
            return [], []
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Detect vehicles
        vehicle_results = self.vehicle_model(frame, conf=self.vehicle_conf)[0]
        vehicle_detections = []
        
        # Filter for vehicle classes
        for box in vehicle_results.boxes:
            cls = int(box.cls.item())
            if cls in self.vehicle_classes:
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                confidence = box.conf.item()
                vehicle_detections.append({
                    'box': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class': self.vehicle_classes[cls]
                })
        
        # Detect license plates - two approaches
        plate_detections = []
        
        # Approach 1: Use general object detection
        plate_results = self.vehicle_model(frame, conf=self.plate_conf)[0]
        
        # Approach 2: Use direct scanning of vehicle regions
        for vehicle in vehicle_detections:
            v_x1, v_y1, v_x2, v_y2 = vehicle['box']
            
            # Extract the vehicle region
            vehicle_region = frame[int(v_y1):int(v_y2), int(v_x1):int(v_x2)]
            
            # Only process if the region is valid
            if vehicle_region.size > 0:
                # Look for license plate regions in the vehicle area
                # 1. Use standard model detection
                try:
                    vehicle_results = self.vehicle_model(vehicle_region, conf=self.plate_conf)[0]
                    
                    # Process each detected box within the vehicle
                    for box in vehicle_results.boxes:
                        # Convert coordinates to the full frame
                        rx1, ry1, rx2, ry2 = box.xyxy.cpu().numpy()[0]
                        # Adjust coordinates to the original frame
                        x1, y1 = rx1 + v_x1, ry1 + v_y1
                        x2, y2 = rx2 + v_x1, ry2 + v_y1
                        confidence = box.conf.item()
                        
                        # Process this region as a potential plate
                        self._process_potential_plate_region(frame, x1, y1, x2, y2, confidence, plate_detections)
                except Exception as e:
                    # Just continue if there's an error with this region
                    pass
                
                # 2. Use manual scanning for potential plate regions
                height, width = vehicle_region.shape[:2]
                # Typical locations for license plates (bottom center, middle, etc.)
                regions_to_check = [
                    # Bottom center
                    (width//4, height*2//3, width*3//4, height),
                    # Middle center
                    (width//4, height//3, width*3//4, height*2//3),
                    # Full region (small vehicles)
                    (0, 0, width, height)
                ]
                
                for rx1, ry1, rx2, ry2 in regions_to_check:
                    # Convert to full frame coordinates
                    x1, y1 = rx1 + v_x1, ry1 + v_y1
                    x2, y2 = rx2 + v_x1, ry2 + v_y1
                    # Process this region as a potential plate
                    self._process_potential_plate_region(frame, x1, y1, x2, y2, 0.5, plate_detections)
        
        # Process results from general object detection
        for box in plate_results.boxes:
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
            confidence = box.conf.item()
            
            # Process this region as a potential plate
            self._process_potential_plate_region(frame, x1, y1, x2, y2, confidence, plate_detections)
        
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
        """Draw bounding boxes and labels on the frame"""
        if frame is None:
            return frame
        
        # Create a copy of the frame to avoid modifying the original
        result = frame.copy()
        
        # Draw vehicles
        for vehicle in vehicles:
            x1, y1, x2, y2 = map(int, vehicle['box'])
            confidence = vehicle['confidence']
            vehicle_class = vehicle['class']
            
            # Draw the bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw the label
            label = f"{vehicle_class}: {confidence:.2f}"
            cv2.putText(result, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw license plates
        for plate in plates:
            x1, y1, x2, y2 = map(int, plate['box'])
            text = plate['text']
            confidence = plate['text_confidence']
            
            # Draw the bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw the label
            label = f"{text}: {confidence:.2f}"
            cv2.putText(result, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
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
        """Process a region as a potential license plate"""
        # Convert to int coordinates
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Ensure coordinates are within frame bounds
        height, width = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        # Calculate width and height
        region_width = x2 - x1
        region_height = y2 - y1
        
        # Skip if region is too small
        if region_width < 10 or region_height < 10:
            return
            
        # Calculate aspect ratio
        aspect_ratio = region_width / region_height if region_height > 0 else 0
        
        # License plates typically have an aspect ratio between 1:1 and 7:1
        # More relaxed criteria for testing
        if 0.5 < aspect_ratio < 10.0 and region_width < 0.8 * width and region_height < 0.5 * height:
            # Extract the potential license plate image
            plate_img = frame[y1:y2, x1:x2]
            
            # Skip if image is invalid
            if plate_img.size == 0:
                return
            
            # Apply image enhancement for better OCR
            # Resize if too small
            min_width = 150  # Minimum width for good OCR
            if region_width < min_width:
                scale = min_width / region_width
                new_width = int(region_width * scale)
                new_height = int(region_height * scale)
                try:
                    plate_img = cv2.resize(plate_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                except Exception:
                    # Skip if resize fails
                    return
            
            try:
                # Recognize the license plate text
                plate_text, text_confidence = recognize_license_plate(plate_img)
                
                # Skip if text recognition failed
                if plate_text is None:
                    return
                
                # Check if it's a valid license plate
                if is_valid_license_plate(plate_text) and text_confidence > 0.1:
                    # Check if this plate has already been detected (to avoid duplicates)
                    is_duplicate = False
                    for existing_plate in plate_detections:
                        if existing_plate['text'] == plate_text:
                            # If this detection has higher confidence, update the existing one
                            if text_confidence > existing_plate['text_confidence']:
                                existing_plate['box'] = [x1, y1, x2, y2]
                                existing_plate['confidence'] = confidence
                                existing_plate['text_confidence'] = text_confidence
                            is_duplicate = True
                            break
                    
                    # Add new detection if not a duplicate
                    if not is_duplicate:
                        plate_detections.append({
                            'box': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'text': plate_text,
                            'text_confidence': text_confidence
                        })
            except Exception as e:
                # Just continue if there's an error
                pass 