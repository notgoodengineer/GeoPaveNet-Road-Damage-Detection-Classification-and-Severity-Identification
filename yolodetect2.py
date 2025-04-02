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
from ultralytics import YOLO
import cv2
import numpy as np

# Ensure the detections folder exists
os.makedirs("/home/geopavenet/detections", exist_ok=True)

# Function to split classification and severity
def split_classification_and_severity(class_name):
    """Splits the class name into classification and severity."""
    if '-' in class_name:
        classification, severity = class_name.rsplit('-', 1)
        return classification.strip(), severity.strip().lower()
    return class_name.strip(), 'unknown'

# Function to read GPS data
def gps_worker(gps_queue, serial_port='/dev/ttyACM0', baudrate=9600):
    try:
        ser = serial.Serial(serial_port, baudrate, timeout=1)
        while True:
            line = ser.readline().decode('utf-8')
            if line.startswith('$GPGGA'):
                msg = pynmea2.parse(line)
                gps_queue.put((msg.latitude, msg.longitude))
                print(f"GPS Data: Latitude = {msg.latitude}, Longitude = {msg.longitude}")
    except Exception as e:
        print(f"Error reading GPS data: {e}")
        gps_queue.put((None, None))

# Function to initialize database
def init_database():
    conn = sqlite3.connect('/home/geopavenet/detections/detections.db', check_same_thread=False)
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
            print(f"Deleting database entry {row[0]} because image {row[1]} does not exist.")
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

def calculate_fps(prev_time):
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    return fps, curr_time

def draw_text_with_background(image, text, position, font_scale, color, thickness, bg_color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(image, (x, y - text_height - 5), (x + text_width, y + 5), bg_color, -1)
    cv2.putText(image, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def create_detection_table(class_counts, width, height):
    table_img = np.zeros((height, width, 3), dtype=np.uint8)
    table_img.fill(240)
    
    # Header
    cv2.rectangle(table_img, (0, 0), (width, 50), (50, 50, 50), -1)
    cv2.putText(table_img, "Detection Counts", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Content
    cell_height = max(30, (height - 50) // max(1, len(class_counts)))
    for i, (class_name, count) in enumerate(class_counts.items()):
        y_pos = 50 + i * cell_height
        if y_pos + cell_height > height:
            break
        if i % 2 == 0:
            cv2.rectangle(table_img, (0, y_pos), (width, y_pos + cell_height), (220, 220, 220), -1)
        cv2.putText(table_img, f"{class_name}:", (10, y_pos + 25), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(table_img, str(count), (width - 50, y_pos + 25), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return table_img

def create_buttons(width, is_detecting):
    button_img = np.zeros((60, width, 3), dtype=np.uint8)
    button_img.fill(240)
    
    btn_width = width // 2 - 10
    start_color = (0, 200, 0) if not is_detecting else (100, 100, 100)
    stop_color = (0, 0, 200) if is_detecting else (100, 100, 100)
    
    cv2.rectangle(button_img, (5, 5), (btn_width, 55), start_color, -1)
    cv2.putText(button_img, "START", (btn_width//4, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.rectangle(button_img, (btn_width + 15, 5), (width - 5, 55), stop_color, -1)
    cv2.putText(button_img, "STOP", (btn_width + 15 + btn_width//3, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    return button_img

def main():
    # Initialize queues and threads
    gps_queue = queue.Queue()
    db_queue = queue.Queue()
    image_queue = queue.Queue(maxsize=15)
    
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

    # Load model
    model = YOLO("/home/geopavenet/yolo/train2/weights/best.engine")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Window setup for 1024x600 screen
    cv2.namedWindow("YOLOv8 Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv8 Detection", 1024, 600)
    try:
        cv2.setWindowProperty("YOLOv8 Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except:
        cv2.setWindowProperty("YOLOv8 Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLOv8 Detection", 1024, 600)

    # Initialize variables
    lat, lon = None, None
    prev_time = time.time()
    fps = 0
    is_detecting = True
    class_counts = {name: 0 for name in model.names.values()}
    sidebar_width = 300

    def mouse_callback(event, x, y, flags, param):
        nonlocal is_detecting
        if x > 1024 - sidebar_width and y > 600 - 60:
            btn_width = sidebar_width // 2 - 10
            if 1024 - sidebar_width + 5 < x < 1024 - sidebar_width + btn_width + 5:
                is_detecting = True
            elif 1024 - sidebar_width + btn_width + 15 < x < 1024 - 5:
                is_detecting = False

    cv2.setMouseCallback("YOLOv8 Detection", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        fps, prev_time = calculate_fps(prev_time)

        if is_detecting:
            results = model(frame)
            current_counts = {name: 0 for name in model.names.values()}
            
            for box in results[0].boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                class_name = model.names[class_id]
                current_counts[class_name] += 1

                if confidence > 0.6:
                    frame_copy = frame.copy()
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = get_yolov8_color(class_id)
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name} {confidence:.2f}"
                    draw_text_with_background(frame_copy, label, (x1, y1 - 5), 0.7, (255, 255, 255), 2, color)
                    
                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    unique_id = str(uuid.uuid4())[:4]  # Convert UUID to string first
                    filename = f"/home/geopavenet/detections/{timestamp}_{class_name}_{unique_id}.jpg"
                    image_queue.put((filename, frame_copy))
                    
                    classification, severity = split_classification_and_severity(class_name)
                    db_queue.put((timestamp, lat, lon, classification, severity, confidence, filename))

            for class_name, count in current_counts.items():
                if count > 0:
                    class_counts[class_name] += count
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

        # Resize camera feed to fit 724x600 (leaving 300px for sidebar)
        h, w = annotated_frame.shape[:2]
        aspect = w / h
        new_w = min(724, int(600 * aspect))
        new_h = min(600, int(724 / aspect))
        resized = cv2.resize(annotated_frame, (new_w, new_h))
        
        # Create camera display area
        camera_display = np.zeros((600, 724, 3), dtype=np.uint8)
        x_offset = (724 - new_w) // 2
        y_offset = (600 - new_h) // 2
        camera_display[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        # Create sidebar
        sidebar = np.zeros((600, sidebar_width, 3), dtype=np.uint8)
        sidebar.fill(240)
        
        # Add table and buttons
        table = create_detection_table(class_counts, sidebar_width, 540)
        buttons = create_buttons(sidebar_width, is_detecting)
        sidebar[:540, :] = table
        sidebar[540:, :] = buttons

        # Combine frames
        combined = np.hstack((camera_display, sidebar))

        # Display
        cv2.setWindowTitle("YOLOv8 Detection", f"YOLOv8 Detection | FPS: {fps:.0f}")
        cv2.imshow("YOLOv8 Detection", combined)

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
