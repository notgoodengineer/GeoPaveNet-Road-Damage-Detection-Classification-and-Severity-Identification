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
from ultralytics import YOLO  # Use YOLOv8
import cv2

# Ensure the detections folder exists
os.makedirs("/home/geopavenet/detections", exist_ok=True)

# Function to split classification and severity
def split_classification_and_severity(class_name):
    """
    Splits the class name into classification and severity.
    Example: 'pothole-low' -> ('pothole', 'low')
    """
    if '-' in class_name:
        classification, severity = class_name.rsplit('-', 1)  # Split on the last hyphen
        return classification.strip(), severity.strip().lower()
    return class_name.strip(), 'unknown'  # Default severity if not specified

# Function to read GPS data in a separate thread
def gps_worker(gps_queue, serial_port='/dev/ttyACM0', baudrate=9600):
    try:
        ser = serial.Serial(serial_port, baudrate, timeout=1)
        while True:
            line = ser.readline().decode('utf-8')
            if line.startswith('$GPGGA'):
                msg = pynmea2.parse(line)
                gps_queue.put((msg.latitude, msg.longitude))
                print(f"GPS Data: Latitude = {msg.latitude}, Longitude = {msg.longitude}")  # Debug print
    except Exception as e:
        print(f"Error reading GPS data: {e}")
        gps_queue.put((None, None))  # Signal GPS failure

# Function to initialize the SQLite3 database
def init_database():
    conn = sqlite3.connect('/home/geopavenet/detections/detections.db', check_same_thread=False)  # Allow connections from multiple threads
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            latitude REAL,
            longitude REAL,
            classification TEXT,  -- Classification (e.g., 'pothole')
            severity TEXT,        -- Severity (e.g., 'low', 'moderate', 'severe')
            confidence REAL,
            image_path TEXT UNIQUE  -- Ensure image paths are unique
        )
    ''')
    conn.commit()
    return conn

# Function to clean up the database by removing entries with missing images
def cleanup_database(conn):
    """
    Checks the database for entries where the image file no longer exists and deletes those entries.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT id, image_path FROM detections")
    rows = cursor.fetchall()

    for row in rows:
        id, image_path = row
        if not os.path.exists(image_path):
            print(f"Deleting database entry {id} because image {image_path} does not exist.")
            cursor.execute("DELETE FROM detections WHERE id = ?", (id,))
            conn.commit()

# Database cleanup worker thread
def db_cleanup_worker(interval=60):
    """
    Continuously checks the database for missing images and deletes invalid entries.
    """
    conn = init_database()  # Create a new database connection for this thread
    while True:
        cleanup_database(conn)
        time.sleep(interval)  # Wait for the specified interval before checking again

# Database worker thread
def db_worker(db_queue):
    conn = init_database()  # Create a new database connection for this thread
    cursor = conn.cursor()

    while True:
        data = db_queue.get()
        if data is None:  # Exit signal
            break

        # Insert data into the database
        cursor.execute('''
            INSERT INTO detections (timestamp, latitude, longitude, classification, severity, confidence, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', data)
        conn.commit()
        print(f"Inserted into database: {data}")  # Debug print
        db_queue.task_done()

    conn.close()  # Close the database connection when the thread exits

# Function to save images in a separate thread
def image_saving_worker(image_queue):
    while True:
        filename, image = image_queue.get()
        if filename is None:  # Exit signal
            break
        try:
            cv2.imwrite(filename, image)
            print(f"Saved image: {filename}")  # Debug print
        except Exception as e:
            print(f"Error saving image {filename}: {e}")
        finally:
            image_queue.task_done()

# Function to get YOLOv8 style colors
def get_yolov8_color(class_id):
    """
    Returns a consistent color for a given class ID, similar to YOLOv8's plot() method.
    """
    # YOLOv8 uses a specific color palette for classes
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 128),  # Purple
        (128, 128, 0),  # Olive
        (0, 128, 128),  # Teal
        (128, 0, 0),    # Maroon
    ]
    return colors[class_id % len(colors)]

# Function to calculate FPS
def calculate_fps(prev_time):
    """
    Calculates the FPS based on the time difference between frames.
    """
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    return fps, curr_time

# Function to draw text with background
def draw_text_with_background(image, text, position, font_scale, color, thickness, bg_color):
    """
    Draws text with a background rectangle for better visibility.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Draw background rectangle
    x, y = position
    cv2.rectangle(image, (x, y - text_height - 5), (x + text_width, y + 5), bg_color, -1)

    # Draw text
    cv2.putText(image, text, (x, y), font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

# Main function for YOLOv8 real-time detection
def main():
    # Initialize GPS queue
    gps_queue = queue.Queue()

    # Initialize database queue
    db_queue = queue.Queue()

    # Initialize image saving queue
    image_queue = queue.Queue(maxsize=15)

    # Start GPS worker thread
    gps_thread = threading.Thread(target=gps_worker, args=(gps_queue,))
    gps_thread.daemon = True
    gps_thread.start()

    # Start database worker thread
    db_thread = threading.Thread(target=db_worker, args=(db_queue,))
    db_thread.daemon = True
    db_thread.start()

    # Start image saving worker thread
    image_thread = threading.Thread(target=image_saving_worker, args=(image_queue,))
    image_thread.daemon = True
    image_thread.start()

    # Start database cleanup worker thread
    db_cleanup_thread = threading.Thread(target=db_cleanup_worker, args=(60,))  # Check every 60 seconds
    db_cleanup_thread.daemon = True
    db_cleanup_thread.start()

    # Load YOLOv8 model
    model = YOLO("/home/geopavenet/yolo/train2/weights/best.engine")  # Use your TensorRT model

    # Open the camera (or video stream)
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Create the OpenCV window
    cv2.namedWindow("YOLOv8 Detection", cv2.WINDOW_NORMAL)

    # Initialize GPS data
    lat, lon = None, None

    # Initialize FPS calculation
    prev_time = time.time()
    fps = 0

    while True:
        # Capture frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Calculate FPS
        fps, prev_time = calculate_fps(prev_time)

        # Perform object detection with YOLOv8
        results = model(frame)

        # Get the latest GPS data from the queue (non-blocking)
        try:
            lat, lon = gps_queue.get_nowait()
        except queue.Empty:
            pass  # No new GPS data available

        # Process detections
        for i, box in enumerate(results[0].boxes):
            class_id = int(box.cls)
            confidence = float(box.conf)
            class_name = model.names[class_id]

            # Save high-confidence detections
            if confidence > 0.6:
                # Create a copy of the original frame
                frame_copy = frame.copy()

                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Get YOLOv8 style color for the class
                color = get_yolov8_color(class_id)

                # Draw the bounding box (slightly thicker)
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness=2)

                # Add label with background
                label = f"{class_name} {confidence:.2f}"
                draw_text_with_background(frame_copy, label, (x1, y1 - 5), 0.7, (255, 255, 255), 2, color)

                # Generate a unique filename
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                unique_id = str(uuid.uuid4())[:4]
                filename = f"/home/geopavenet/detections/{timestamp}_{class_name}_{unique_id}.jpg"

                # Save the frame with the current detection in the image queue
                image_queue.put((filename, frame_copy))

                # Split class name into classification and severity
                classification, severity = split_classification_and_severity(class_name)

                # Push detection data to the database queue
                db_queue.put((timestamp, lat, lon, classification, severity, confidence, filename))

        # Display the frame with all detections (for real-time visualization)
        annotated_frame = results[0].plot()

        # Update the window title with FPS
        cv2.setWindowTitle("YOLOv8 Detection", f"YOLOv8 Detection | FPS: {fps:.0f}")

        # Show the frame
        cv2.imshow("YOLOv8 Detection", annotated_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

    # Signal threads to exit
    gps_queue.put((None, None))
    db_queue.put(None)
    image_queue.put((None, None))

    # Wait for threads to finish
    gps_thread.join()
    db_thread.join()
    image_thread.join()

if __name__ == "__main__":
    main()

