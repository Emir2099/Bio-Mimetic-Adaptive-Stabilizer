import pygame
import requests
import math
import sys
from collections import deque

# --- CONFIGURATION (UPDATE THIS IP!) ---
PHONE_URL = "http://192.168.0.107:8080"
DATA_URL = f"{PHONE_URL}/get?gyrX&gyrY&gyrZ&GyroX&GyroY&GyroZ"

# --- ACADEMIC PARAMETERS ---
STANDARD_ALPHA = 0.05
STABLE_ALPHA = 0.02
ACTION_ALPHA = 0.60
ENERGY_THRESHOLD = 0.15

# --- PYGAME SETUP ---
pygame.init()
WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("S-Edge: Bio-Mimetic Stabilization Middleware")
font_title = pygame.font.Font(None, 40)
font_label = pygame.font.Font(None, 28)
font_metrics = pygame.font.Font(None, 24)

# --- DATA STRUCTURES ---
HISTORY_LEN = 200
raw_history = deque([0]*HISTORY_LEN, maxlen=HISTORY_LEN)
std_history = deque([0]*HISTORY_LEN, maxlen=HISTORY_LEN)
bio_history = deque([0]*HISTORY_LEN, maxlen=HISTORY_LEN)

class StabilizerSystem:
    def __init__(self):
        self.raw_val = 0.0
        self.std_val = 0.0
        self.bio_val = 0.0
        self.current_stiffness = 0.1
        self.mode = "IDLE"

    def get_sensor_data(self):
        try:
            response = requests.get(DATA_URL, timeout=0.05)
            data = response.json()
            buffer = data["buffer"]
            val = 0.0
            if "gyrX" in buffer and buffer["gyrX"]["buffer"]:
                val = buffer["gyrX"]["buffer"][0]
            elif "GyroX" in buffer and buffer["GyroX"]["buffer"]:
                val = buffer["GyroX"]["buffer"][0]
            return val
        except:
            return 0.0

    def lerp(self, start, end, alpha):
        return start + (end - start) * alpha

    def update(self):
        sensor_input = self.get_sensor_data()
        
        # SENSITIVITY MULTIPLIER
        target = sensor_input * 1000 
        self.raw_val = target

        # Standard Filter
        self.std_val = self.lerp(self.std_val, target, STANDARD_ALPHA)

        # Bio-Mimetic Filter
        energy = abs(sensor_input)
        if energy < ENERGY_THRESHOLD:
            target_alpha = STABLE_ALPHA
            self.mode = "STABILIZING"
        else:
            target_alpha = ACTION_ALPHA
            self.mode = "ACTION"

        self.current_stiffness = self.lerp(self.current_stiffness, target_alpha, 0.1)
        self.bio_val = self.lerp(self.bio_val, target, self.current_stiffness)

        raw_history.append(self.raw_val)
        std_history.append(self.std_val)
        bio_history.append(self.bio_val)

# --- VISUALIZATION HELPERS ---
def draw_bar(x, y, value, color, label, sublabel):
    # Bar background
    pygame.draw.rect(screen, (50, 50, 50), (x, y - 150, 60, 300))
    
    # --- CLAMP BAR HEIGHT ---
    # Prevents bars from growing infinitely and covering text
    # We cap the visual height at 145 pixels
    visual_height = max(min(value, 145), -145)
    
    pygame.draw.rect(screen, color, (x, y, 60, visual_height))
    
    # Labels
    text = font_label.render(label, True, color)
    screen.blit(text, (x - 10, y + 160))
    sub = font_metrics.render(sublabel, True, (200, 200, 200))
    screen.blit(sub, (x - 10, y + 185))

def draw_graph(rect):
    # Background
    pygame.draw.rect(screen, (20, 20, 20), rect)
    pygame.draw.rect(screen, (100, 100, 100), rect, 2)
    
    # Center Line
    mid_y = rect.y + rect.height // 2
    pygame.draw.line(screen, (50, 50, 50), (rect.x, mid_y), (rect.x + rect.width, mid_y))

    # --- AUTO-SCALING LOGIC ---
    # 1. Find the largest absolute value currently in history
    max_val = 0
    if len(raw_history) > 0:
        max_raw = max(abs(x) for x in raw_history)
        max_bio = max(abs(x) for x in bio_history)
        max_val = max(max_raw, max_bio)
    
    # 2. Set a minimum floor (e.g., 500)
    # If the phone is still, values might be 0.1. If we scale that to the full screen,
    # it looks like huge noise. 500 keeps the line flat when values are low.
    max_val = max(max_val, 500) 

    # 3. Calculate Scale Factor
    # We want max_val to fit into half the height (rect.height / 2)
    # We multiply by 0.9 to leave a 10% padding so it doesn't touch the border
    scale_factor = (rect.height / 2 * 0.9) / max_val

    # Plot lines
    for i in range(1, len(raw_history)):
        x1 = rect.x + (i-1) * (rect.width / HISTORY_LEN)
        x2 = rect.x + i * (rect.width / HISTORY_LEN)
        
        # Apply scale_factor to Y values
        y1_raw = mid_y + (raw_history[i-1] * scale_factor)
        y2_raw = mid_y + (raw_history[i] * scale_factor)
        
        y1_bio = mid_y + (bio_history[i-1] * scale_factor)
        y2_bio = mid_y + (bio_history[i] * scale_factor)

        # Raw (Red, Thin)
        pygame.draw.line(screen, (150, 50, 50), (x1, y1_raw), (x2, y2_raw), 1)
        
        # Bio-Mimetic (Green, Thick)
        pygame.draw.line(screen, (50, 255, 50), (x1, y1_bio), (x2, y2_bio), 3)

    # Show current Scale on screen
    scale_text = font_metrics.render(f"Scale: +/- {int(max_val)}", True, (100, 100, 100))
    screen.blit(scale_text, (rect.x + 5, rect.y + 5))

# --- MAIN LOOP ---
system = StabilizerSystem()
clock = pygame.time.Clock()

print("System Initialized. Press Ctrl+C to stop.")

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    system.update()

    # --- DRAWING ---
    screen.fill((30, 30, 35))

    # 1. Title Section
    title = font_title.render("Comparison: Standard Filter vs. Bio-Mimetic Control", True, (255, 255, 255))
    screen.blit(title, (50, 30))
    
    subtitle = font_metrics.render(f"System Mode: {system.mode} | Stiffness (Alpha): {system.current_stiffness:.3f}", True, (50, 255, 50))
    screen.blit(subtitle, (50, 70))

    # 2. The Bars
    center_y = 300
    draw_bar(200, center_y, system.raw_val, (255, 80, 80), "RAW INPUT", "No Filter")
    draw_bar(450, center_y, system.std_val, (80, 80, 255), "STANDARD", f"Fixed (a={STANDARD_ALPHA})")
    draw_bar(700, center_y, system.bio_val, (80, 255, 80), "BIO-MIMETIC", "Adaptive")

    # 3. The Graph
    graph_rect = pygame.Rect(50, 500, WIDTH - 100, 150)
    draw_graph(graph_rect)
    
    # Legend for Graph
    # screen.blit(font_metrics.render("Graph History: Red=Raw Noise, Green=Your Algorithm", True, (200, 200, 200)), (50, 480))

    pygame.display.flip()
    clock.tick(60)