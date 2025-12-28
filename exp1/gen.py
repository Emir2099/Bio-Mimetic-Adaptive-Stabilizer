import pygame
import requests
import math
import sys
import csv
import time
from collections import deque

# --- CONFIGURATION (UPDATE THIS IP!) ---
PHONE_URL = "http://192.168.0.107:8080"  # <--- CHECK THIS IP
DATA_URL = f"{PHONE_URL}/get?gyrX&gyrY&gyrZ&GyroX&GyroY&GyroZ"

# --- ACADEMIC PARAMETERS ---
STANDARD_ALPHA = 0.05
STABLE_ALPHA = 0.02
ACTION_ALPHA = 0.60
ENERGY_THRESHOLD = 0.15

# --- 1 EURO FILTER CONFIG (The Competitor) ---
ONE_EURO_MIN_CUTOFF = 1.0
ONE_EURO_BETA = 0.007
ONE_EURO_DCUTOFF = 1.0

# --- PYGAME SETUP ---
pygame.init()
WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("B-VIC Data Logger for IEEE Access")
font_title = pygame.font.Font(None, 40)
font_label = pygame.font.Font(None, 28)
font_metrics = pygame.font.Font(None, 24)
font_alert = pygame.font.Font(None, 60)

# --- DATA STRUCTURES ---
HISTORY_LEN = 200
raw_history = deque([0]*HISTORY_LEN, maxlen=HISTORY_LEN)
bio_history = deque([0]*HISTORY_LEN, maxlen=HISTORY_LEN)
euro_history = deque([0]*HISTORY_LEN, maxlen=HISTORY_LEN)

# --- 1 EURO FILTER IMPLEMENTATION ---
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = 0.0
        self.dx_prev = 0.0
        self.t_prev = time.time()

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def filter(self, x, t):
        t_e = t - self.t_prev
        
        # Avoid division by zero
        if t_e <= 0.0: return self.x_prev
        
        # Calculate jitter (dx)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(self.smoothing_factor(t_e, self.d_cutoff), dx, self.dx_prev)
        
        # Calculate Cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        # Filter signal
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

class StabilizerSystem:
    def __init__(self):
        self.raw_val = 0.0
        self.std_val = 0.0
        self.bio_val = 0.0
        self.euro_val = 0.0
        
        self.current_stiffness = 0.1
        self.mode = "IDLE"
        
        # Initialize Competitor
        self.one_euro = OneEuroFilter(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_DCUTOFF)

        # Recording State
        self.is_recording = False
        self.csv_file = None
        self.csv_writer = None
        self.start_time = 0

    def start_recording(self):
        self.is_recording = True
        self.start_time = time.time()
        self.csv_file = open('experiment_data.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        # WRITE HEADER
        self.csv_writer.writerow(["Timestamp_s", "Raw_Input", "Standard_EMA", "OneEuro", "BVIC_Output", "Stiffness_Alpha", "Mode_State"])
        print("--- RECORDING STARTED ---")

    def stop_recording(self):
        self.is_recording = False
        if self.csv_file:
            self.csv_file.close()
            print(f"--- RECORDING SAVED: experiment_data.csv ---")

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
        current_time = time.time()
        sensor_input = self.get_sensor_data()
        
        # SENSITIVITY MULTIPLIER (Scale up for visualization)
        target = sensor_input * 1000 
        self.raw_val = target

        # 1. Standard Filter (EMA)
        self.std_val = self.lerp(self.std_val, target, STANDARD_ALPHA)

        # 2. One Euro Filter (Competitor)
        self.euro_val = self.one_euro.filter(target, current_time)

        # 3. Bio-Mimetic Filter (Yours)
        energy = abs(sensor_input)
        state_code = 0 # 0=Stable, 1=Action
        
        if energy < ENERGY_THRESHOLD:
            target_alpha = STABLE_ALPHA
            self.mode = "STABILIZING"
            state_code = 0
        else:
            target_alpha = ACTION_ALPHA
            self.mode = "ACTION"
            state_code = 1

        self.current_stiffness = self.lerp(self.current_stiffness, target_alpha, 0.1)
        self.bio_val = self.lerp(self.bio_val, target, self.current_stiffness)

        # Update History for Graph
        raw_history.append(self.raw_val)
        bio_history.append(self.bio_val)
        euro_history.append(self.euro_val)

        # SAVE TO CSV
        if self.is_recording:
            elapsed = current_time - self.start_time
            self.csv_writer.writerow([
                f"{elapsed:.4f}", 
                f"{self.raw_val:.4f}", 
                f"{self.std_val:.4f}", 
                f"{self.euro_val:.4f}",
                f"{self.bio_val:.4f}", 
                f"{self.current_stiffness:.4f}",
                state_code
            ])

# --- VISUALIZATION HELPERS ---
def draw_bar(x, y, value, color, label, sublabel):
    pygame.draw.rect(screen, (50, 50, 50), (x, y - 150, 60, 300))
    visual_height = max(min(value, 145), -145)
    pygame.draw.rect(screen, color, (x, y, 60, visual_height))
    text = font_label.render(label, True, color)
    screen.blit(text, (x - 10, y + 160))
    sub = font_metrics.render(sublabel, True, (200, 200, 200))
    screen.blit(sub, (x - 10, y + 185))

def draw_graph(rect):
    pygame.draw.rect(screen, (20, 20, 20), rect)
    pygame.draw.rect(screen, (100, 100, 100), rect, 2)
    mid_y = rect.y + rect.height // 2
    pygame.draw.line(screen, (50, 50, 50), (rect.x, mid_y), (rect.x + rect.width, mid_y))

    max_val = 0
    if len(raw_history) > 0:
        max_val = max(max(abs(x) for x in raw_history), 500)

    scale_factor = (rect.height / 2 * 0.9) / max_val

    for i in range(1, len(raw_history)):
        x1 = rect.x + (i-1) * (rect.width / HISTORY_LEN)
        x2 = rect.x + i * (rect.width / HISTORY_LEN)
        
        y1_raw = mid_y + (raw_history[i-1] * scale_factor)
        y2_raw = mid_y + (raw_history[i] * scale_factor)
        
        y1_bio = mid_y + (bio_history[i-1] * scale_factor)
        y2_bio = mid_y + (bio_history[i] * scale_factor)

        # Raw (Red, Thin)
        pygame.draw.line(screen, (150, 50, 50), (x1, y1_raw), (x2, y2_raw), 1)
        # Bio (Green, Thick)
        pygame.draw.line(screen, (50, 255, 50), (x1, y1_bio), (x2, y2_bio), 3)

# --- MAIN LOOP ---
system = StabilizerSystem()
clock = pygame.time.Clock()

print("System Initialized. Press 'R' to Record.")

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            if system.is_recording: system.stop_recording()
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                if system.is_recording:
                    system.stop_recording()
                else:
                    system.start_recording()

    system.update()

    # --- DRAWING ---
    screen.fill((30, 30, 35))

    title = font_title.render("IEEE Access Data Logger: Press 'R' to Record", True, (255, 255, 255))
    screen.blit(title, (50, 30))
    
    subtitle = font_metrics.render(f"Mode: {system.mode} | Alpha: {system.current_stiffness:.3f}", True, (50, 255, 50))
    screen.blit(subtitle, (50, 70))

    if system.is_recording:
        rec_text = font_alert.render("RECORDING...", True, (255, 0, 0))
        screen.blit(rec_text, (WIDTH - 300, 30))

    center_y = 300
    draw_bar(200, center_y, system.raw_val, (255, 80, 80), "RAW", "Input")
    draw_bar(400, center_y, system.std_val, (80, 80, 255), "STD", "EMA")
    draw_bar(600, center_y, system.euro_val, (255, 255, 80), "1 EURO", "Competitor")
    draw_bar(800, center_y, system.bio_val, (80, 255, 80), "B-VIC", "Ours")

    graph_rect = pygame.Rect(50, 500, WIDTH - 100, 150)
    draw_graph(graph_rect)
    
    pygame.display.flip()
    clock.tick(60)