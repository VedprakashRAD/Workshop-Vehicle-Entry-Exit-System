# Workshop Vehicle Entry/Exit System

A simple and cost-effective solution for OEMs to monitor vehicles entering and exiting their workshop. This system uses automatic license plate recognition (ANPR) from video streams and displays vehicle details fetched from RTO API.

## Features

- **Real-time Vehicle Detection**: Detects vehicles in video streams using YOLOv8
- **License Plate Recognition**: Extracts and reads license plate text using deep learning and OCR
- **Split-View Interface**:
  - Left side: Live video feed with vehicle/license plate detection
  - Right side: Vehicle details from RTO API
- **Vehicle Details**: Displays comprehensive vehicle information including registration, insurance, and pollution certificate details
- **Entry/Exit Logging**: Automatically logs when vehicles enter or exit the workshop

## System Requirements

- Python 3.8+
- Webcam or IP camera
- Internet connection for RTO API access

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/workshop-vehicle-system.git
cd workshop-vehicle-system
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with the following variables:
```
RTO_API_KEY=your_api_key_here
RTO_API_URL=https://api.example.com/rto/v1
```

## Usage

1. Run the application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. The system will automatically:
   - Detect vehicles in the video feed
   - Recognize license plates
   - Fetch vehicle details from the RTO API
   - Log vehicle entries and exits

## Configuration

- You can adjust camera settings in `app.py` by changing the `video_source` variable
- Detection confidence thresholds can be modified in `vehicle_detector.py`
- The RTO API endpoint can be configured in the `.env` file

## Implementation Details

This project uses:

- **YOLOv8** for vehicle detection
- **EasyOCR** for license plate text extraction
- **Flask** for the web interface
- **OpenCV** for image processing and video handling
- **Bootstrap** for responsive UI design

## Notes

- For production use, it's recommended to:
  - Use a dedicated GPU for faster detection
  - Implement proper authentication for the web interface
  - Set up a proper database for long-term storage of entry/exit logs

## Credits

This project is based on the [Video-ANPR](https://github.com/sveyek/Video-ANPR) repository by sveyek, with additional features for workshop management.

## License

MIT License 