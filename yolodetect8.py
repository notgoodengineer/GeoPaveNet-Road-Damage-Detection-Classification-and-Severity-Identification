import argparse
import datetime
import sys
import os
import sqlite3
import threading
import queue
import serial
import pynmea2
import time
import uuid
import collections
import math
import requests
import io
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
import pytz

# Configuration
DETECTIONS_DIR = "/home/geopavenet/detections"
DATABASE_PATH = os.path.join(DETECTIONS_DIR, "detections.db")
CONFIDENCE_THRESHOLD = 0.6  # Only process detections above this confidence
GPS_SERIAL_PORT = '/dev/ttyACM0'
GPS_BAUDRATE = 9600
TILE_CACHE_SIZE = 100  # Number of tiles to cache in memory

# Ensure the detections folder exists
os.makedirs(DETECTIONS_DIR, exist_ok=True)


class MapTileManager:
    def __init__(self):
        self.tile_cache = collections.OrderedDict()
        self.tile_size = 256
        self.tile_servers = [
            "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
            "https://b.tile.openstreetmap.org/{z}/{x}/{y}.png",
            "https://c.tile.openstreetmap.org/{z}/{x}/{y}.png"
        ]
        self.current_server = 0
        self.loading_tiles = set()
        self.lock = threading.Lock()

    def download_tile(self, zoom, xtile, ytile):
        url = self.tile_servers[self.current_server].format(
            z=zoom, x=xtile, y=ytile)
        try:
            headers = {'User-Agent': 'RoadDamageDetector/2.0'}
            response = requests.get(url, headers=headers, timeout=3)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
            return np.array(img)
        except Exception as e:
            print(f"Tile download failed: {e}")
            self.current_server = (
                self.current_server + 1) % len(self.tile_servers)
            return None

    def get_tile(self, zoom, xtile, ytile):
        key = (zoom, xtile, ytile)
        with self.lock:
            if key in self.tile_cache:
                self.tile_cache.move_to_end(key)
                return self.tile_cache[key]

            if key not in self.loading_tiles:
                self.loading_tiles.add(key)
                threading.Thread(
                    target=self._async_load_tile, args=(
                        zoom, xtile, ytile), daemon=True).start()
        return None

    def _async_load_tile(self, zoom, xtile, ytile):
        tile = self.download_tile(zoom, xtile, ytile)
        with self.lock:
            if tile is not None:
                self.tile_cache[(zoom, xtile, ytile)] = tile
                if len(self.tile_cache) > TILE_CACHE_SIZE:
                    self.tile_cache.popitem(last=False)
            self.loading_tiles.discard((zoom, xtile, ytile))


class DetectionTracker:
    def __init__(self, model_names):
        self.class_counts = {name: 0 for name in model_names.values()}
        self.lock = threading.Lock()

    def update_counts(self, class_name, count=1):
        with self.lock:
            self.class_counts[class_name] += count

    def get_counts(self):
        with self.lock:
            return self.class_counts.copy()


def deg2num(lat_deg, lon_deg, zoom):
    """Convert coordinates to tile numbers"""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def num2deg(xtile, ytile, zoom):
    """Convert tile numbers to coordinates"""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def split_classification_and_severity(class_name):
    """Splits the class name into classification and severity."""
    if '-' in class_name:
        classification, severity = class_name.rsplit('-', 1)
        return classification.strip(), severity.strip().lower()
    return class_name.strip(), 'unknown'


def gps_worker(gps_queue, serial_port=GPS_SERIAL_PORT, baudrate=GPS_BAUDRATE):
    ser = None
    while True:
        try:
            # Initialize or reinitialize serial connection
            if ser is None or not ser.is_open:
                if ser is not None:
                    ser.close()
                ser = serial.Serial(serial_port, baudrate, timeout=1)
                time.sleep(2)  # Allow time for connection to establish

            line = ser.readline().decode('utf-8').strip()
            if line.startswith('$GPGGA'):
                try:
                    msg = pynmea2.parse(line)
                    if msg.latitude and msg.longitude:
                        gps_queue.put((msg.latitude, msg.longitude))
                        print(f"GPS Data: Latitude = {msg.latitude}, Longitude = {msg.longitude}")
                except Exception as e:
                    print(f"Error parsing GPS data: {e}")

        except serial.SerialException as e:
            print(f"Serial port error: {e}")
            if ser is not None:
                ser.close()
                ser = None
            gps_queue.put((None, None))  # Signal GPS disconnection
            time.sleep(5)  # Wait before reconnection attempt
        except Exception as e:
            print(f"Unexpected GPS error: {e}")
            if ser is not None:
                ser.close()
                ser = None
            time.sleep(1)


def init_database():
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            latitude REAL,
            longitude REAL,
            classification TEXT,
            severity TEXT,
            confidence REAL,
            image_path TEXT UNIQUE
        )
    ''')
    conn.commit()
    return conn


def cleanup_database(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT id, image_path FROM detections")
    for row in cursor.fetchall():
        if not os.path.exists(row[1]):
            print(
                f"Deleting database entry {row[0]} because image {row[1]} does not exist.")
            cursor.execute("DELETE FROM detections WHERE id = ?", (row[0],))
            conn.commit()


def db_cleanup_worker(interval=60):
    conn = init_database()
    while True:
        cleanup_database(conn)
        time.sleep(interval)


def db_worker(db_queue):
    conn = init_database()
    cursor = conn.cursor()
    while True:
        data = db_queue.get()
        if data is None:
            break
        cursor.execute('''
            INSERT INTO detections (timestamp, latitude, longitude, classification, severity, confidence, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', data)
        conn.commit()
        print(f"Inserted into database: {data}")
        db_queue.task_done()
    conn.close()


def image_saving_worker(image_queue):
    while True:
        filename, image = image_queue.get()
        if filename is None:
            break
        try:
            cv2.imwrite(filename, image)
            print(f"Saved image: {filename}")
        except Exception as e:
            print(f"Error saving image {filename}: {e}")
        finally:
            image_queue.task_done()


def get_yolov8_color(class_id):
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 128), (128, 128, 0), (0, 128, 128),
        (128, 0, 0)
    ]
    return colors[class_id % len(colors)]


class FPSCounter:
    def __init__(self, window_size=30):
        self.frame_times = collections.deque(maxlen=window_size)
        self.prev_time = time.perf_counter()

    def update(self):
        curr_time = time.perf_counter()
        frame_time = curr_time - self.prev_time
        self.frame_times.append(frame_time)
        self.prev_time = curr_time
        return len(self.frame_times) / \
            sum(self.frame_times) if self.frame_times else 0


def draw_text_with_background(
        image,
        text,
        position,
        font_scale,
        color,
        thickness,
        bg_color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), _ = cv2.getTextSize(
        text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(image, (x, y - text_height - 5),
                  (x + text_width, y + 5), bg_color, -1)
    cv2.putText(image, text, (x, y), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


def create_detection_table(class_counts, width, height):
    table_img = np.zeros((height, width, 3), dtype=np.uint8)
    table_img.fill(240)

    # Header
    cv2.rectangle(table_img, (0, 0), (width, 50), (50, 50, 50), -1)
    cv2.putText(table_img, "Detection Counts", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Content
    cell_height = max(30, (height - 50) // max(1, len(class_counts)))
    for i, (class_name, count) in enumerate(class_counts.items()):
        y_pos = 50 + i * cell_height
        if y_pos + cell_height > height:
            break
        if i % 2 == 0:
            cv2.rectangle(table_img, (0, y_pos),
                          (width, y_pos + cell_height), (220, 220, 220), -1)
        cv2.putText(
            table_img,
            f"{class_name}:",
            (10,
             y_pos + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,
             0,
             0),
            1,
            cv2.LINE_AA)
        cv2.putText(table_img, str(count), (width - 50, y_pos + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return table_img


def create_buttons(width, is_detecting):
    button_img = np.zeros((60, width, 3), dtype=np.uint8)
    button_img.fill(240)

    btn_width = width // 2 - 10
    start_color = (0, 200, 0) if not is_detecting else (100, 100, 100)
    stop_color = (0, 0, 200) if is_detecting else (100, 100, 100)

    cv2.rectangle(button_img, (5, 5), (btn_width, 55), start_color, -1)
    cv2.putText(button_img, "START", (btn_width // 4, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.rectangle(button_img, (btn_width + 15, 5),
                  (width - 5, 55), stop_color, -1)
    cv2.putText(button_img, "STOP", (btn_width + 15 + btn_width // 3, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return button_img


def create_enhanced_gps_display(
        lat,
        lon,
        detections,
        width=300,
        height=490,
        zoom=18):
    """Improved map display with proper tile alignment and enhanced zoom"""
    if lat is None or lon is None:
        gps_img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(gps_img, "Waiting for GPS...", (10, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return gps_img

    gps_img = np.zeros((height, width, 3), dtype=np.uint8)
    gps_img.fill(200)

    # Calculate the exact pixel coordinates of our position
    xtile, ytile = deg2num(lat, lon, zoom)
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x_pixel = int((lon + 180.0) / 360.0 * n * 256) % 256
    y_pixel = int((1.0 - math.asinh(math.tan(lat_rad)) /
                  math.pi) / 2.0 * n * 256) % 256

    # Calculate how many tiles we need to cover the display area
    tiles_x = math.ceil(width / 256) + 1
    tiles_y = math.ceil(height / 256) + 1

    # Calculate the offset to center our position
    offset_x = width // 2 - x_pixel
    offset_y = height // 2 - y_pixel

    # Load all needed tiles
    for dx in range(-tiles_x // 2, tiles_x // 2 + 1):
        for dy in range(-tiles_y // 2, tiles_y // 2 + 1):
            tile_x = xtile + dx
            tile_y = ytile + dy
            tile = tile_manager.get_tile(zoom, tile_x, tile_y)

            if tile is not None:
                if len(tile.shape) == 2:
                    tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)

                # Calculate where to place this tile
                tile_left = offset_x + dx * 256
                tile_top = offset_y + dy * 256

                # Calculate visible portion
                img_left = max(0, tile_left)
                img_top = max(0, tile_top)
                img_right = min(width, tile_left + 256)
                img_bottom = min(height, tile_top + 256)

                if img_right > img_left and img_bottom > img_top:
                    # Calculate tile portion to copy
                    tile_left_clip = max(0, -tile_left)
                    tile_top_clip = max(0, -tile_top)
                    tile_right_clip = 256 - max(0, (tile_left + 256) - width)
                    tile_bottom_clip = 256 - max(0, (tile_top + 256) - height)

                    tile_portion = tile[tile_top_clip:tile_bottom_clip,
                                        tile_left_clip:tile_right_clip]
                    gps_img[img_top:img_bottom,
                            img_left:img_right] = tile_portion

    # Draw current position (larger and more visible for navigation)
    cv2.circle(gps_img, (width // 2, height // 2),
               12, (0, 0, 255), -1)  # Red center
    cv2.circle(gps_img, (width // 2, height // 2),
               15, (255, 255, 255), 3)  # White border

    # Draw direction indicator (assuming forward motion)
    cv2.arrowedLine(gps_img, (width // 2, height // 2), (width //
                    2, height // 2 - 30), (0, 255, 0), 3, tipLength=0.5)

    # Plot detections with distance indicators
    if lat is not None and lon is not None and detections:
        for detection in detections[-10:]:  # Show only recent detections
            if detection['lat'] and detection['lon']:
                dx = int((detection['lon'] - lon) *
                         (2**zoom * 256 / 360)) + width // 2
                dy = int((lat - detection['lat']) *
                         (2**zoom * 256 / 360)) + height // 2

                if 0 <= dx < width and 0 <= dy < height:
                    # Calculate distance in meters
                    distance = haversine(
                        lon, lat, detection['lon'], detection['lat'])
                    color = (0, 165, 255)  # Orange

                    cv2.circle(gps_img, (dx, dy), 8, color, -1)
                    cv2.circle(gps_img, (dx, dy), 10, (255, 255, 255), 1)

                    # Show distance label
                    label = f"{distance:.0f}m"
                    (w, h), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(gps_img, (dx - w // 2 - 2, dy - h - 15),
                                  (dx + w // 2 + 2, dy - 10), (0, 0, 0), -1)
                    cv2.putText(gps_img, label, (dx - w // 2, dy - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Add navigation overlay
    overlay = np.zeros((60, width, 3), dtype=np.uint8)
    overlay.fill(0)  # Black overlay

    # Adjust these values to change opacity (0.0 to 1.0)
    overlay_opacity = 0.4  # Higher value = more opaque
    image_opacity = 1 - overlay_opacity  # Complementary value

    cv2.addWeighted(gps_img[:60, :], image_opacity, 
                   overlay, overlay_opacity, 
                   0, gps_img[:60, :])

    # Add coordinates and zoom
    cv2.putText(gps_img, f"Zoom: {zoom}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(gps_img, f"Lat: {lat:.6f}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(gps_img, f"Lon: {lon:.6f}", (width - 150, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return gps_img


def create_tab_buttons(width, active_tab):
    """Create buttons to switch between views"""
    tab_img = np.zeros((50, width, 3), dtype=np.uint8)
    tab_img.fill(240)

    # Detection Counts Tab
    counts_color = (0, 150, 0) if active_tab == "counts" else (100, 100, 100)
    cv2.rectangle(tab_img, (5, 5), (width // 2 - 5, 45), counts_color, -1)
    cv2.putText(tab_img, "Counts", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # GPS Map Tab
    map_color = (0, 150, 0) if active_tab == "map" else (100, 100, 100)
    cv2.rectangle(tab_img, (width // 2 + 5, 5), (width - 5, 45), map_color, -1)
    cv2.putText(tab_img, "Map", (width // 2 + 30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return tab_img


def haversine(lon1, lat1, lon2, lat2):
    """Calculate distance between two points in meters"""
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2)**2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def create_zoom_buttons(width, zoom):
    """Create zoom control buttons with consistent dimensions"""
    zoom_img = np.zeros(
        (40, width, 3), dtype=np.uint8)  # Fixed height to 40 pixels
    zoom_img.fill(240)

    # Zoom in button (disabled at max zoom)
    zoom_in_color = (200, 200, 200) if zoom < 20 else (150, 150, 150)
    cv2.rectangle(zoom_img, (5, 5), (width // 2 - 5, 35), zoom_in_color, -1)
    cv2.putText(
        zoom_img,
        "+",
        (width // 4 - 5,
         28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,
         0,
         0) if zoom < 20 else (
            100,
            100,
            100),
        2,
        cv2.LINE_AA)

    # Zoom out button (disabled at min zoom)
    zoom_out_color = (200, 200, 200) if zoom > 12 else (150, 150, 150)
    cv2.rectangle(zoom_img, (width // 2 + 5, 5),
                  (width - 5, 35), zoom_out_color, -1)
    cv2.putText(
        zoom_img,
        "-",
        (3 * width // 4 - 5,
         28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,
         0,
         0) if zoom > 12 else (
            100,
            100,
            100),
        2,
        cv2.LINE_AA)

    # Current zoom level
    cv2.putText(zoom_img, f"Zoom: {zoom}", (width // 2 - 30, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return zoom_img


def get_correct_timestamps():
    """Generate properly formatted timestamps with timezone"""
    tz = pytz.timezone('Asia/Manila')  # Change to your timezone
    now = datetime.datetime.now(tz)
    return (
        now.strftime("%Y-%m-%d %H:%M:%S"),  # Database format
        now.strftime("%Y%m%d_%H%M%S")       # Filename format
    )


def main():
    global tile_manager, is_detecting, mouse_down, active_tab, zoom, popup_image, popup_position, popup_active
    global current_popup_group, current_popup_index, last_click_time
    global slide_start_x, slide_start_y, is_sliding, slide_threshold

   # Load model
    model_path = "/home/geopavenet/yolo/train2/weights/best.engine"
    model = YOLO(model_path, task='detect', verbose=False)
    model.overrides['imgsz'] = 416
    model.overrides['device'] = '0'
    
    # Get class names from model
    if hasattr(model, 'names'):
        class_names = model.names
    else:
        class_names = {0: 'D00', 1: 'D10', 2: 'D20', 3: 'D40'}
    
    detection_tracker = DetectionTracker(class_names)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Initialize tile manager
    tile_manager = MapTileManager()
    
    # Initialize queues and threads
    gps_queue = queue.Queue()
    db_queue = queue.Queue()
    image_queue = queue.Queue(maxsize=10)
    
    gps_thread = threading.Thread(target=gps_worker, args=(gps_queue,))
    gps_thread.daemon = True
    gps_thread.start()
    
    db_thread = threading.Thread(target=db_worker, args=(db_queue,))
    db_thread.daemon = True
    db_thread.start()
    
    image_thread = threading.Thread(target=image_saving_worker, args=(image_queue,))
    image_thread.daemon = True
    image_thread.start()
    
    db_cleanup_thread = threading.Thread(target=db_cleanup_worker, args=(60,))
    db_cleanup_thread.daemon = True
    db_cleanup_thread.start()

    # Window setup
    cv2.namedWindow("Road Damage Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Road Damage Detection", 1024, 600)
    try:
        cv2.setWindowProperty("Road Damage Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except:
        cv2.setWindowProperty("Road Damage Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Road Damage Detection", 1024, 600)

    # Initialize state variables
    is_detecting = True
    mouse_down = False
    active_tab = "counts"
    zoom = 19
    popup_image = None
    popup_position = None
    popup_active = False
    current_popup_group = None
    current_popup_index = 0
    last_click_time = 0
    slide_start_x = 0
    slide_start_y = 0
    is_sliding = False
    slide_threshold = 50  # Minimum slide distance to trigger navigation

    # Initialize other variables
    lat, lon = None, None
    fps_counter = FPSCounter()
    sidebar_width = 300
    last_table_update = 0
    table_update_interval = 0.5
    detection_history = []
    MAX_HISTORY = 20
    table = None

    def mouse_callback(event, x, y, flags, param):
        global is_detecting, mouse_down, active_tab, zoom, popup_active, last_click_time
        global current_popup_group, current_popup_index, popup_image, popup_position
        global slide_start_x, slide_start_y, is_sliding
        
        current_time = time.time()
        
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_down = True
            slide_start_x = x
            slide_start_y = y
            is_sliding = False
            
            if popup_active and popup_position is not None and popup_image is not None:
                px, py = popup_position
                ph, pw = popup_image.shape[:2]
                
                # Close button area (top-right of popup frame)
                close_x1 = px + pw - 30
                close_x2 = px + pw - 10
                close_y1 = py + 10
                close_y2 = py + 30
                
                if ((close_x1 - 10 <= x <= close_x2 + 10) and 
                    (close_y1 - 10 <= y <= close_y2 + 10)):
                    popup_active = False
                    last_click_time = current_time
                    return
                
                if px <= x <= px + pw and py <= y <= py + ph:
                    is_sliding = True
                    return
            
            if x > 724:
                if y < 50:
                    if x < 724 + sidebar_width//2:
                        active_tab = "counts"
                    else:
                        active_tab = "map"
                    last_click_time = current_time
                    return
                
                if active_tab == "map" and 500 <= y <= 540:
                    if x < 724 + sidebar_width//2:
                        zoom = min(20, zoom + 1)
                    else:
                        zoom = max(12, zoom - 1)
                    last_click_time = current_time
                    return
                
                if y > 540:
                    btn_width = sidebar_width // 2 - 10
                    if x < 724 + btn_width + 5:
                        is_detecting = True
                    else:
                        is_detecting = False
                    last_click_time = current_time
                    return
                
                if (active_tab == "map" and 
                    current_time - last_click_time > 0.3 and
                    lat is not None and lon is not None):
                    
                    map_x = x - 724
                    map_y = y - 50
                    
                    for detection in reversed(detection_history):
                        if 'lat' not in detection or 'lon' not in detection or 'images' not in detection:
                            continue
                            
                        dx = int((detection['lon'] - lon) * (2**zoom * 256 / 360)) + sidebar_width//2
                        dy = int((lat - detection['lat']) * (2**zoom * 256 / 360)) + 225
                        
                        if abs(map_x - dx) < 15 and abs(map_y - dy) < 15:
                            current_popup_group = detection
                            current_popup_index = 0
                            show_popup_image(detection['images'][0]['path'])
                            popup_active = True
                            last_click_time = current_time
                            break
        
        elif event == cv2.EVENT_MOUSEMOVE and mouse_down and is_sliding:
            if abs(x - slide_start_x) > slide_threshold:
                if current_popup_group and len(current_popup_group['images']) > 1:
                    if x < slide_start_x:  # Slide left - next image
                        if current_popup_index < len(current_popup_group['images']) - 1:
                            current_popup_index += 1
                            show_popup_image(current_popup_group['images'][current_popup_index]['path'], "left")
                            slide_start_x = x
                        else:  # Wrap around to first image
                            current_popup_index = 0
                            show_popup_image(current_popup_group['images'][0]['path'], "left")
                            slide_start_x = x
                    elif x > slide_start_x:  # Slide right - previous image
                        if current_popup_index > 0:
                            current_popup_index -= 1
                            show_popup_image(current_popup_group['images'][current_popup_index]['path'], "right")
                            slide_start_x = x
                        else:  # Wrap around to last image
                            current_popup_index = len(current_popup_group['images']) - 1
                            show_popup_image(current_popup_group['images'][-1]['path'], "right")
                            slide_start_x = x
        
        elif event == cv2.EVENT_LBUTTONUP:
            mouse_down = False
            is_sliding = False

    def show_popup_image(img_path, slide_direction=None):
        global popup_image, popup_position
        
        try:
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                aspect = w / h
                popup_w = 350
                popup_h = int(popup_w / aspect)
                if popup_h > 450:
                    popup_h = 450
                    popup_w = int(popup_h * aspect)
                
                border_size = 10
                shadow_size = 5
                total_w = popup_w + 2*border_size + shadow_size
                total_h = popup_h + 2*border_size + shadow_size
                
                popup_frame = np.zeros((total_h, total_w, 3), dtype=np.uint8)
                popup_frame.fill(220)
                
                cv2.rectangle(popup_frame, 
                             (shadow_size, shadow_size),
                             (total_w-1, total_h-1),
                             (100, 100, 100), -1)
                
                cv2.rectangle(popup_frame, 
                             (0, 0),
                             (total_w-shadow_size-1, total_h-shadow_size-1),
                             (255, 255, 255), -1)
                
                img_resized = cv2.resize(img, (popup_w, popup_h))
                
                if slide_direction == "left":
                    offset = min(50, popup_w)
                    popup_frame[border_size:border_size+popup_h, 
                               border_size:border_size+popup_w-offset] = img_resized[:, offset:]
                    popup_frame[border_size:border_size+popup_h, 
                               border_size+popup_w-offset:border_size+popup_w] = 0
                elif slide_direction == "right":
                    offset = min(50, popup_w)
                    popup_frame[border_size:border_size+popup_h, 
                               border_size+offset:border_size+popup_w] = img_resized[:, :-offset]
                    popup_frame[border_size:border_size+popup_h, 
                               border_size:border_size+offset] = 0
                else:
                    popup_frame[border_size:border_size+popup_h, 
                               border_size:border_size+popup_w] = img_resized
                
                popup_image = popup_frame
                popup_position = (512 - total_w//2, 300 - total_h//2)
        except Exception as e:
            print(f"Error loading image: {e}")

    cv2.setMouseCallback("Road Damage Detection", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        fps = fps_counter.update()

        if is_detecting:
            results = model(frame)
            current_detections = {}
            
            for box in results[0].boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                class_name = model.names[class_id]
                
                if confidence > CONFIDENCE_THRESHOLD:
                    if class_name not in current_detections:
                        current_detections[class_name] = 0
                    current_detections[class_name] += 1

                    frame_copy = frame.copy()
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = get_yolov8_color(class_id)
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name} {confidence:.2f}"
                    draw_text_with_background(frame_copy, label, (x1, y1 - 5), 0.7, (255, 255, 255), 2, color)
                    
                    db_timestamp, filename_timestamp = get_correct_timestamps()
                    unique_id = str(uuid.uuid4())[:4]
                    filename = os.path.join(DETECTIONS_DIR, f"{filename_timestamp}_{class_name}_{unique_id}.jpg")
                    image_queue.put((filename, frame_copy))
                    classification, severity = split_classification_and_severity(class_name)
                    db_queue.put((db_timestamp, lat, lon, classification, severity, confidence, filename))
                    
                    if lat is not None and lon is not None:
                        grouped = False
                        for existing in detection_history:
                            if (haversine(lon, lat, existing['lon'], existing['lat']) < 5 and 
                                time.time() - existing['time'] < 60):
                                existing['images'].append({
                                    'path': filename,
                                    'class': class_name,
                                    'time': time.time()
                                })
                                grouped = True
                                break
                        
                        if not grouped:
                            detection_history.append({
                                'lat': lat,
                                'lon': lon,
                                'time': time.time(),
                                'images': [{
                                    'path': filename,
                                    'class': class_name,
                                    'time': time.time()
                                }]
                            })
                        
                        if len(detection_history) > MAX_HISTORY:
                            detection_history.pop(0)

            for class_name, count in current_detections.items():
                detection_tracker.update_counts(class_name, count)
        else:
            results = None

        try:
            lat, lon = gps_queue.get_nowait()
        except queue.Empty:
            pass

        # Prepare display frames
        if results and is_detecting:
            annotated_frame = results[0].plot()
        else:
            annotated_frame = frame.copy()

        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2, cv2.LINE_AA)

        h, w = annotated_frame.shape[:2]
        aspect = w / h
        new_w = min(724, int(600 * aspect))
        new_h = min(600, int(724 / aspect))
        resized = cv2.resize(annotated_frame, (new_w, new_h))
        
        camera_display = np.zeros((600, 724, 3), dtype=np.uint8)
        x_offset = (724 - new_w) // 2
        y_offset = (600 - new_h) // 2
        camera_display[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        tab_buttons = create_tab_buttons(sidebar_width, active_tab)
        
        if active_tab == "counts":
            current_time = time.time()
            if current_time - last_table_update > table_update_interval or table is None:
                table = create_detection_table(detection_tracker.get_counts(), sidebar_width, 450)
                last_table_update = current_time
            sidebar_content = table
        else:
            sidebar_content = create_enhanced_gps_display(lat, lon, detection_history, 
                                                         width=sidebar_width, height=450, zoom=zoom)
        
        if active_tab == "map":
            zoom_controls = create_zoom_buttons(sidebar_width, zoom)
        else:
            zoom_controls = np.zeros((40, sidebar_width, 3), dtype=np.uint8)
            zoom_controls.fill(240)
        
        buttons = create_buttons(sidebar_width, is_detecting)
        
        sidebar = np.zeros((600, sidebar_width, 3), dtype=np.uint8)
        sidebar.fill(240)
        sidebar[:50, :] = tab_buttons
        sidebar[50:500, :] = sidebar_content
        sidebar[500:540, :] = zoom_controls
        sidebar[540:, :] = buttons

        combined = np.hstack((camera_display, sidebar))
        
        if popup_active and popup_image is not None and popup_position is not None:
            px, py = popup_position
            ph, pw = popup_image.shape[:2]
            
            combined[py:py+ph, px:px+pw] = popup_image
            
            # Close button
            close_button_size = 25
            cv2.circle(combined, 
                      (px + pw - close_button_size, py + close_button_size),
                      close_button_size-5, (0, 0, 255), -1)
            cv2.line(combined, 
                    (px + pw - close_button_size - 10, py + close_button_size - 10),
                    (px + pw - close_button_size + 10, py + close_button_size + 10),
                    (255, 255, 255), 2)
            cv2.line(combined,
                    (px + pw - close_button_size - 10, py + close_button_size + 10),
                    (px + pw - close_button_size + 10, py + close_button_size - 10),
                    (255, 255, 255), 2)
            
            # Navigation dots (max 12 per group)
            if current_popup_group and len(current_popup_group['images']) > 1:
                max_dots = 12
                dot_radius = 6
                dot_spacing = 15
                
                # Calculate current group and position within group
                group_number = current_popup_index // max_dots
                pos_in_group = current_popup_index % max_dots
                start_idx = group_number * max_dots
                end_idx = min(start_idx + max_dots, len(current_popup_group['images']))
                
                total_width = max_dots * dot_spacing
                start_x = px + pw//2 - total_width//2
                
                # Draw dots for current group
                for i in range(max_dots):
                    if start_idx + i >= len(current_popup_group['images']):
                        break
                    
                    color = (0, 150, 255) if i == pos_in_group else (200, 200, 200)
                    cv2.circle(combined,
                              (start_x + i*dot_spacing, py + ph - 15),
                              dot_radius, color, -1)
                
            # Show popup if active
            if popup_active and popup_image is not None and popup_position is not None:
                px, py = popup_position
                ph, pw = popup_image.shape[:2]
                
                # Overlay the popup on the main display
                combined[py:py+ph, px:px+pw] = popup_image
                
                # Add close button (styled as an X shape)
                close_button_size = 25
                cv2.circle(combined, 
                          (px + pw - close_button_size, py + close_button_size),
                          close_button_size-5, (0, 0, 255), -1)
                # Draw X shape
                cv2.line(combined, 
                        (px + pw - close_button_size - 10, py + close_button_size - 10),
                        (px + pw - close_button_size + 10, py + close_button_size + 10),
                        (255, 255, 255), 2)
                cv2.line(combined,
                        (px + pw - close_button_size - 10, py + close_button_size + 10),
                        (px + pw - close_button_size + 10, py + close_button_size - 10),
                        (255, 255, 255), 2)
                
                # Add navigation indicators if multiple images (max 12 dots)
                if current_popup_group and len(current_popup_group['images']) > 1:
                    max_dots = 12  # Maximum number of dots to show
                    dot_radius = 6
                    dot_spacing = 15
                    
                    # Calculate which dots to show
                    total_images = len(current_popup_group['images'])
                    dots_to_show = min(max_dots, total_images)
                    total_width = (dots_to_show - 1) * dot_spacing
                    
                    # Calculate centered starting position
                    start_x = px + (pw // 2) - (total_width // 2)
                    dots_y = py + ph - 15  # Position 15px from bottom
                    
                    # Draw all visible dots
                    for i in range(dots_to_show):
                        color = (0, 150, 255) if i == current_popup_index % max_dots else (200, 200, 200)
                        cv2.circle(combined,
                                  (start_x + i * dot_spacing, dots_y),
                                  dot_radius, color, -1)
                    
                    # Show group indicators if we have multiple groups
                    if total_images > max_dots:
                        current_group = current_popup_index // max_dots
                        total_groups = (total_images - 1) // max_dots + 1
                        
                        # Left arrow if not in first group
                        if current_group > 0:
                            cv2.arrowedLine(combined,
                                           (start_x - 25, dots_y),
                                           (start_x - 10, dots_y),
                                           (0, 150, 255), 2, tipLength=0.3)
                        
                        # Right arrow if not in last group
                        if (current_group + 1) * max_dots < total_images:
                            cv2.arrowedLine(combined,
                                           (start_x + total_width + 10, dots_y),
                                           (start_x + total_width + 25, dots_y),
                                           (0, 150, 255), 2, tipLength=0.3)
                        
                        # Group indicator text
                        group_text = f"{current_group + 1}/{total_groups}"
                        text_size = cv2.getTextSize(group_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        cv2.putText(combined, group_text,
                                   (px + (pw // 2) - (text_size[0] // 2), dots_y - 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Road Damage Detection", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    gps_queue.put((None, None))
    db_queue.put(None)
    image_queue.put((None, None))
    gps_thread.join()
    db_thread.join()
    image_thread.join()

if __name__ == "__main__":
    main()
