document.addEventListener('DOMContentLoaded', function() {
    // Cache DOM elements
    const recentPlatesGridElement = document.getElementById('recent-plates-grid');
    const currentPlateElement = document.getElementById('current-plate');
    const vehicleDetailsElement = document.getElementById('vehicle-details');
    const entryExitLogElement = document.getElementById('entry-exit-log');
    
    // Store detected plates
    let detectedPlates = [];
    let seenPlates = new Map(); // To track entry/exit status
    let selectedPlate = null;
    
    // Fetch detected plates from the server every 2 seconds
    function fetchDetectedPlates() {
        fetch('/api/detected_plates')
            .then(response => response.json())
            .then(data => {
                // Update the detected plates
                detectedPlates = data;
                updateRecentPlatesList();
                updateEntryExitLog();
            })
            .catch(error => console.error('Error fetching detected plates:', error));
    }
    
    // Update the grid of recently detected plates
    function updateRecentPlatesList() {
        // Clear the current grid
        recentPlatesGridElement.innerHTML = '';
        
        // Add detected plates to the grid
        detectedPlates.forEach(plate => {
            const plateCard = document.createElement('div');
            plateCard.className = 'col-md-4 mb-3';
            
            // Create card style for each plate
            plateCard.innerHTML = `
                <div class="card plate-card ${selectedPlate === plate.plate_number ? 'border-primary' : ''}">
                    <div class="card-body p-2 text-center">
                        <h5 class="card-title">${plate.plate_number}</h5>
                        <p class="card-text mb-1">Confidence: ${Math.round(plate.confidence * 100)}%</p>
                        <button class="btn btn-sm btn-primary">View Details</button>
                    </div>
                </div>
            `;
            
            // Add click event to show vehicle details
            const button = plateCard.querySelector('button');
            button.addEventListener('click', () => {
                selectPlate(plate.plate_number);
            });
            
            recentPlatesGridElement.appendChild(plateCard);
        });
        
        // If no plates detected
        if (detectedPlates.length === 0) {
            const noPlatesDiv = document.createElement('div');
            noPlatesDiv.className = 'col-12 text-center text-muted p-4';
            noPlatesDiv.innerHTML = '<h5>No license plates detected</h5>';
            recentPlatesGridElement.appendChild(noPlatesDiv);
        }
    }
    
    // Select a plate and fetch its vehicle details
    function selectPlate(plateNumber) {
        // Update selected state
        selectedPlate = plateNumber;
        updateRecentPlatesList();
        
        // Update the current plate display
        currentPlateElement.textContent = plateNumber;
        
        // Show loading state
        vehicleDetailsElement.innerHTML = `
            <div class="text-center p-5">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-3">Loading vehicle details...</p>
            </div>
        `;
        
        // Fetch vehicle details
        fetch(`/api/vehicle_details?plate=${encodeURIComponent(plateNumber)}`)
            .then(response => response.json())
            .then(data => {
                displayVehicleDetails(data);
            })
            .catch(error => {
                console.error('Error fetching vehicle details:', error);
                vehicleDetailsElement.innerHTML = `
                    <div class="alert alert-danger" role="alert">
                        Error loading vehicle details. Please try again.
                    </div>
                `;
            });
    }
    
    // Display vehicle details
    function displayVehicleDetails(data) {
        vehicleDetailsElement.innerHTML = `
            <div class="row detail-row">
                <div class="col-5 detail-label">License Plate:</div>
                <div class="col-7 detail-value">${data.license_plate}</div>
            </div>
            <div class="row detail-row">
                <div class="col-5 detail-label">Vehicle Type:</div>
                <div class="col-7 detail-value">${data.vehicle_type}</div>
            </div>
            <div class="row detail-row">
                <div class="col-5 detail-label">Manufacturer:</div>
                <div class="col-7 detail-value">${data.manufacturer}</div>
            </div>
            <div class="row detail-row">
                <div class="col-5 detail-label">Model:</div>
                <div class="col-7 detail-value">${data.model}</div>
            </div>
            <div class="row detail-row">
                <div class="col-5 detail-label">Year:</div>
                <div class="col-7 detail-value">${data.year}</div>
            </div>
            <div class="row detail-row">
                <div class="col-5 detail-label">Color:</div>
                <div class="col-7 detail-value">${data.color}</div>
            </div>
            
            <h5 class="mt-4 mb-3">Registration Details</h5>
            <div class="row detail-row">
                <div class="col-5 detail-label">Registration Date:</div>
                <div class="col-7 detail-value">${data.registration.date}</div>
            </div>
            <div class="row detail-row">
                <div class="col-5 detail-label">Expiry Date:</div>
                <div class="col-7 detail-value">${data.registration.expiry}</div>
            </div>
            <div class="row detail-row">
                <div class="col-5 detail-label">Status:</div>
                <div class="col-7 detail-value">
                    <span class="badge ${data.registration.status === 'Active' ? 'bg-success' : 'bg-danger'}">
                        ${data.registration.status}
                    </span>
                </div>
            </div>
            
            <h5 class="mt-4 mb-3">Insurance Details</h5>
            <div class="row detail-row">
                <div class="col-5 detail-label">Provider:</div>
                <div class="col-7 detail-value">${data.insurance.provider}</div>
            </div>
            <div class="row detail-row">
                <div class="col-5 detail-label">Policy Number:</div>
                <div class="col-7 detail-value">${data.insurance.policy_number}</div>
            </div>
            <div class="row detail-row">
                <div class="col-5 detail-label">Expiry Date:</div>
                <div class="col-7 detail-value">${data.insurance.expiry}</div>
            </div>
            
            <h5 class="mt-4 mb-3">Pollution Certificate</h5>
            <div class="row detail-row">
                <div class="col-5 detail-label">Certificate Number:</div>
                <div class="col-7 detail-value">${data.pollution.certificate_number}</div>
            </div>
            <div class="row detail-row">
                <div class="col-5 detail-label">Expiry Date:</div>
                <div class="col-7 detail-value">${data.pollution.expiry}</div>
            </div>
        `;
    }
    
    // Update entry/exit log
    function updateEntryExitLog() {
        // Process the detected plates to update the entry/exit log
        detectedPlates.forEach(plate => {
            const plateNumber = plate.plate_number;
            const now = new Date();
            const timeString = now.toLocaleTimeString();
            
            // If this plate is new (entry) or was seen more than 2 minutes ago (re-entry)
            if (!seenPlates.has(plateNumber) || 
                (now - seenPlates.get(plateNumber).time) > 120000) {
                
                // Add entry to log
                const logEntry = document.createElement('tr');
                logEntry.innerHTML = `
                    <td>${plateNumber}</td>
                    <td>${timeString}</td>
                    <td class="status-entry">Entry</td>
                `;
                
                // Add to beginning of log
                if (entryExitLogElement.firstChild) {
                    entryExitLogElement.insertBefore(logEntry, entryExitLogElement.firstChild);
                } else {
                    entryExitLogElement.appendChild(logEntry);
                }
                
                // Update seen plates
                seenPlates.set(plateNumber, {
                    time: now,
                    status: 'entry'
                });
            } else {
                // Update last seen time
                seenPlates.get(plateNumber).time = now;
            }
        });
        
        // Check for exits (plates that were seen but are no longer in detected plates)
        seenPlates.forEach((value, plateNumber) => {
            const now = new Date();
            
            // If plate was seen in the last 2 minutes but is not in current detections
            // and hasn't been marked as exited yet
            if (value.status === 'entry' && 
                (now - value.time) < 120000 && 
                !detectedPlates.some(p => p.plate_number === plateNumber)) {
                
                // Add exit to log
                const timeString = now.toLocaleTimeString();
                const logEntry = document.createElement('tr');
                logEntry.innerHTML = `
                    <td>${plateNumber}</td>
                    <td>${timeString}</td>
                    <td class="status-exit">Exit</td>
                `;
                
                // Add to beginning of log
                if (entryExitLogElement.firstChild) {
                    entryExitLogElement.insertBefore(logEntry, entryExitLogElement.firstChild);
                } else {
                    entryExitLogElement.appendChild(logEntry);
                }
                
                // Update status to exited
                value.status = 'exit';
                value.time = now;
            }
            
            // Clean up old records (older than 5 minutes)
            if ((now - value.time) > 300000) {
                seenPlates.delete(plateNumber);
            }
        });
        
        // Limit log size
        while (entryExitLogElement.children.length > 50) {
            entryExitLogElement.removeChild(entryExitLogElement.lastChild);
        }
    }
    
    // Start fetching detected plates
    fetchDetectedPlates();
    setInterval(fetchDetectedPlates, 2000);
}); 