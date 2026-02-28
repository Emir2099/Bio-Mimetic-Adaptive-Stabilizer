import requests
import time
import math

# --- CONFIGURATION ---
# ENSURE THIS MATCHES YOUR PHONE
PHONE_URL = "http://192.168.0.107:8080" 

# We ask for BOTH naming conventions at once to be safe
DATA_URL = f"{PHONE_URL}/get?gyrX&gyrY&gyrZ&GyroX&GyroY&GyroZ"

print(f"Connecting to {PHONE_URL}...")
print("Make sure the PLAY button is pressed on your phone!")

while True:
    try:
        # Increased timeout to 3.0 seconds to fix "Connection Timeout"
        response = requests.get(DATA_URL, timeout=3.0)
        data = response.json()
        
        # Check if the app is paused
        if "status" in data and data["status"]["measuring"] is False:
            print("Status: PAUSED (Press Play on Phone)")
            time.sleep(1)
            continue
            
        buffer = data["buffer"]
        
        # --- ROBUST DATA EXTRACTION ---
        # We check WHICH name the phone is actually sending
        gx, gy, gz = 0, 0, 0
        
        if "gyrX" in buffer:
            # Case A: Phone uses lowercase 'gyrX'
            # We also check if the buffer has data inside it
            if buffer["gyrX"]["buffer"]:
                gx = buffer["gyrX"]["buffer"][0]
                gy = buffer["gyrY"]["buffer"][0]
                gz = buffer["gyrZ"]["buffer"][0]
            else:
                # Buffer exists but is empty (packet loss)
                continue 
                
        elif "GyroX" in buffer:
            # Case B: Phone uses Capital 'GyroX'
            if buffer["GyroX"]["buffer"]:
                gx = buffer["GyroX"]["buffer"][0]
                gy = buffer["GyroY"]["buffer"][0]
                gz = buffer["GyroZ"]["buffer"][0]
            else:
                continue
                
        else:
            print("Warning: connected, but no Gyro data found. Checking...")
            time.sleep(1)
            continue

        # --- THE BIO-MIMETIC ALGORITHM ---
        
        # 1. Calculate 'Energy'
        energy = math.sqrt(gx**2 + gy**2 + gz**2)
        
        # 2. Apply Adaptive Logic
        if energy < 0.2:
            mode = "STABILIZING (Active)"
            # In a real visualizer, we would set stiffness = 0.95 here
        else:
            mode = "ACTION (Pass-through)"
            # In a real visualizer, we would set stiffness = 0.10 here
            
        print(f"Gyro: {gx:.2f} | Energy: {energy:.2f} | System: {mode}")

    except requests.exceptions.Timeout:
        print("Network Lag... (Retrying)")
    except Exception as e:
        print(f"Retrying... ({e})")
        
    time.sleep(0.1)