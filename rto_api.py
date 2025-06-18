import os
import requests
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RTOApiService:
    def __init__(self):
        """Initialize the RTO API service with API key from environment variables"""
        self.api_key = os.getenv('RTO_API_KEY', '')
        self.api_base_url = os.getenv('RTO_API_URL', 'https://api.example.com/rto/v1')
        self.cache = {}  # Simple cache to avoid redundant API calls
        self.cache_ttl = 3600  # Cache time-to-live in seconds (1 hour)

    def get_vehicle_details(self, license_plate):
        """Get vehicle details from RTO API based on license plate number"""
        if not license_plate:
            return None
        
        # Check cache first
        if license_plate in self.cache:
            cache_time, data = self.cache[license_plate]
            if time.time() - cache_time < self.cache_ttl:
                return data
        
        # If not in cache or expired, call the API
        try:
            # This is a placeholder implementation. Replace with actual API endpoint.
            url = f"{self.api_base_url}/vehicle/{license_plate}"
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                # Cache the result
                self.cache[license_plate] = (time.time(), data)
                return data
            else:
                print(f"API Error: {response.status_code}, {response.text}")
                return None
                
        except Exception as e:
            print(f"Error fetching vehicle details: {e}")
            return None
    
    def get_mock_vehicle_details(self, license_plate):
        """Mock function to generate sample vehicle details for testing"""
        # This function provides mock data for testing when actual API is not available
        vehicle_types = ["Sedan", "SUV", "Hatchback", "Truck", "Van", "Motorcycle"]
        manufacturers = ["Toyota", "Honda", "Hyundai", "Maruti Suzuki", "Tata", "Mahindra"]
        colors = ["Red", "Blue", "White", "Black", "Silver", "Grey"]
        
        # Generate consistent mock data based on the license plate number
        seed = sum(ord(c) for c in license_plate)
        vehicle_type = vehicle_types[seed % len(vehicle_types)]
        manufacturer = manufacturers[(seed // 10) % len(manufacturers)]
        color = colors[(seed // 100) % len(colors)]
        year = 2010 + (seed % 13)  # Years between 2010 and 2022
        
        # Create mock data
        mock_data = {
            "license_plate": license_plate,
            "vehicle_type": vehicle_type,
            "manufacturer": manufacturer,
            "model": f"{manufacturer} {vehicle_type} {year}",
            "year": year,
            "color": color,
            "owner": {
                "name": "John Doe",  # Anonymized for privacy
                "address": "123 Example St, City",  # Anonymized for privacy
            },
            "registration": {
                "date": f"{year}-01-01",
                "expiry": f"{year + 15}-01-01",
                "status": "Active"
            },
            "insurance": {
                "provider": "Example Insurance",
                "policy_number": f"INS{license_plate}",
                "expiry": f"{year + 1}-01-01"
            },
            "pollution": {
                "certificate_number": f"PUC{license_plate}",
                "expiry": f"{year + 1}-06-01"
            }
        }
        
        return mock_data 