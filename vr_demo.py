import pygame
import requests
import math
import sys

# --- CONFIGURATION ---
PHONE_URL = "http://192.168.0.107:8080" 
DATA_URL = f"{PHONE_URL}/get?gyrX&gyrY&gyrZ&GyroX&GyroY&GyroZ"

# --- PYGAME SETUP ---
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Bio-Mimetic Stabilizer Demo")
font = pygame.font.Font(None, 36)

# --- VARIABLES ---
raw_val = 0.0        # Where the raw sensor is
stabilized_val = 0.0 # Where our "Virtual Hand" is

# Your Novelty: The Variable Stiffness
current_stiffness = 0.1 

def get_sensor_data():
    """Fetches the latest Gyro X speed from the phone."""
    try:
        response = requests.get(DATA_URL, timeout=0.1) # Fast timeout for graphics
        data = response.json()
        buffer = data["buffer"]
        
        # Robust name checking (gyrX vs GyroX)
        if "gyrX" in buffer and buffer["gyrX"]["buffer"]:
            return buffer["gyrX"]["buffer"][0] # Return X-axis rotation speed
        elif "GyroX" in buffer and buffer["GyroX"]["buffer"]:
            return buffer["GyroX"]["buffer"][0]
    except:
        pass
    return 0.0

def lerp(start, end, alpha):
    """Linear Interpolation (The Smoothing Function)"""
    return start + (end - start) * alpha

# --- MAIN LOOP ---
clock = pygame.time.Clock()
running = True

print("Starting Simulation... Shake your phone!")

while running:
    # 1. Event Handling (Close Window)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 2. GET INPUT (The 'Nerves')
    sensor_input = get_sensor_data()
    
    # Scale input for visibility (Gyro is usually small, like 0.01)
    # We multiply by 50 so it moves the bar visibly on screen
    target_position = sensor_input * 50 

    # 3. CALCULATE ENERGY (The 'Brain')
    # We use absolute value because speed is energy regardless of direction
    energy = abs(sensor_input)

    # 4. THE NOVEL ALGORITHM (Adaptive Stiffness)
    # If Energy is LOW (< 0.15), we assume tremor/noise -> High Stiffness (Low Alpha)
    # If Energy is HIGH (> 0.15), we assume Action -> Low Stiffness (High Alpha)
    
    if energy < 0.15:
        # MUSCLE TENSION MODE (Stabilizing)
        # We only move 2% towards the target per frame. Very smooth.
        target_stiffness = 0.02 
        mode_text = "MODE: STABILIZING (Jitter Filter)"
        mode_color = (0, 255, 0) # Green Text
    else:
        # ACTION MODE (Relaxed)
        # We move 50% towards the target. Snappy and fast.
        target_stiffness = 0.50
        mode_text = "MODE: ACTION (Low Latency)"
        mode_color = (255, 50, 50) # Red Text

    # Smooth the stiffness transition itself so it doesn't "pop"
    current_stiffness = lerp(current_stiffness, target_stiffness, 0.1)

    # 5. UPDATE POSITIONS
    # Red Bar jumps instantly to target (Raw)
    raw_val = target_position 
    
    # Green Bar follows smoothly based on our smart stiffness
    stabilized_val = lerp(stabilized_val, target_position, current_stiffness)

    # --- DRAWING ---
    screen.fill((30, 30, 30)) # Dark Grey Background

    # Center of screen
    center_y = HEIGHT // 2

    # Draw Center Line
    pygame.draw.line(screen, (100, 100, 100), (0, center_y), (WIDTH, center_y), 2)

    # Draw RED Bar (Raw Input) - Left Side
    pygame.draw.rect(screen, (255, 50, 50), (200, center_y, 100, raw_val))
    
    # Draw GREEN Bar (Bio-Mimetic) - Right Side
    pygame.draw.rect(screen, (50, 255, 50), (500, center_y, 100, stabilized_val))

    # Labels
    text_raw = font.render("Raw Sensor", True, (255, 100, 100))
    text_stab = font.render("Bio-Mimetic", True, (100, 255, 100))
    text_mode = font.render(mode_text, True, mode_color)

    screen.blit(text_raw, (180, center_y + 150))
    screen.blit(text_stab, (480, center_y + 150))
    screen.blit(text_mode, (WIDTH//2 - 150, 50))

    pygame.display.flip()
    clock.tick(60) # Run at 60 FPS

pygame.quit()
sys.exit()