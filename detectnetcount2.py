#!/usr/bin/env python3
#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# This script saves the detected objects to individual images in a directory.
# The directory these are stored in can be set using the --snapshots argument:
#
#   python3 detectnet-snap.py --snapshots /path/to/snapshots <input_URI> <output_URI>
#
# The input and output streams are specified the same way as shown here:
# https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md
#

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
import uuid  # Added for unique filenames

from jetson_inference import detectNet
from jetson_utils import (videoSource, videoOutput, saveImage, Log,
                          cudaAllocMapped, cudaMemcpy, cudaDeviceSynchronize, cudaFont)

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
    except Exception as e:
        print(f"Error reading GPS data: {e}")
        gps_queue.put((None, None))  # Signal GPS failure

# Function to initialize the SQLite3 database
def init_database():
    conn = sqlite3.connect('detections.db')
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

# Database worker thread
def db_worker(db_queue):
    conn = init_database()
    cursor = conn.cursor()

    while True:
        # Get data from the queue
        data = db_queue.get()

        # If the data is None, exit the thread
        if data is None:
            break

        # Insert data into the database
        cursor.execute('''
            INSERT INTO detections (timestamp, latitude, longitude, classification, severity, confidence, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', data)
        conn.commit()

        # Mark the task as done
        db_queue.task_done()

    # Close the database connection
    conn.close()

# Function to monitor the image directory for deletions
def monitor_image_directory(image_dir, db_queue):
    conn = sqlite3.connect('detections.db')
    cursor = conn.cursor()

    # Track existing files
    existing_files = set(os.listdir(image_dir))

    while True:
        # Get the current list of files
        current_files = set(os.listdir(image_dir))

        # Find deleted files
        deleted_files = existing_files - current_files

        # Delete corresponding database entries
        for file in deleted_files:
            file_path = os.path.join(image_dir, file)
            cursor.execute('DELETE FROM detections WHERE image_path = ?', (file_path,))
            conn.commit()
            print(f"Deleted database entry for: {file_path}")

        # Update the list of existing files
        existing_files = current_files

        # Sleep for a short time to avoid high CPU usage
        time.sleep(1)

    # Close the database connection
    conn.close()

# Function to check for orphaned database entries
def check_orphaned_entries(image_dir, db_queue):
    conn = sqlite3.connect('detections.db')
    cursor = conn.cursor()

    # Get all image paths from the database
    cursor.execute('SELECT image_path FROM detections')
    db_image_paths = cursor.fetchall()

    # Check if each image path exists in the directory
    for db_image_path in db_image_paths:
        image_path = db_image_path[0]
        if not os.path.exists(image_path):
            # If the image file does not exist, delete the database entry
            cursor.execute('DELETE FROM detections WHERE image_path = ?', (image_path,))
            conn.commit()
            print(f"Deleted orphaned database entry for: {image_path}")

    # Close the database connection
    conn.close()

# Worker thread for image saving
def image_saving_worker(image_queue):
    while True:
        filename, image = image_queue.get()
        if filename is None:  # Sentinel value to exit the thread
            break
        try:
            saveImage(filename, image)
        except Exception as e:
            print(f"Error saving image {filename}: {e}")
        finally:
            del image  # Free the copied image memory
            image_queue.task_done()

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 
parser.add_argument("--snapshots", type=str, default="images/test/detections", help="output directory of detection snapshots")
parser.add_argument("--timestamp", type=str, default="%Y%m%d-%H%M%S-%f", help="timestamp format used in snapshot filenames")

try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# make sure the snapshots dir exists
os.makedirs(args.snapshots, exist_ok=True)

# Create a thread-safe queue for GPS data
gps_queue = queue.Queue()

# Create a thread-safe queue for database operations
db_queue = queue.Queue()

# Create a thread-safe queue for image saving with a maximum size
image_queue = queue.Queue(maxsize=15)

# Start the GPS worker thread
gps_thread = threading.Thread(target=gps_worker, args=(gps_queue,))
gps_thread.daemon = True  # Daemonize thread to exit when the main program exits
gps_thread.start()

# Start the database worker thread
db_thread = threading.Thread(target=db_worker, args=(db_queue,))
db_thread.daemon = True  # Daemonize thread to exit when the main program exits
db_thread.start()

# Start the image saving worker thread
image_thread = threading.Thread(target=image_saving_worker, args=(image_queue,))
image_thread.daemon = True  # Daemonize thread to exit when the main program exits
image_thread.start()

# Start the image directory monitor thread
monitor_thread = threading.Thread(target=monitor_image_directory, args=(args.snapshots, db_queue))
monitor_thread.daemon = True  # Daemonize thread to exit when the main program exits
monitor_thread.start()

# Check for orphaned database entries before starting the main loop
check_orphaned_entries(args.snapshots, db_queue)

# create video output object 
output = videoOutput(args.output, argv=sys.argv)
	
# load the object detection network
net = detectNet(args.network, sys.argv, args.threshold)

# create video sources
input = videoSource(args.input, argv=sys.argv)

# Initialize GPS data
lat, lon = None, None

# Initialize a dictionary to keep track of class counts (global tally)
global_class_counts = {}

# Initialize CUDA font for text overlay with a smaller font size
font = cudaFont(size=21)  # Adjust the size as needed (default is 32)

# process frames until EOS or the user exits
while True:
    # capture the next image
    img = input.Capture()

    if img is None: # timeout
        continue  

    # detect objects in the image (with overlay)
    detections = net.Detect(img, overlay=args.overlay)

    # print the detections
    print("detected {:d} objects in image".format(len(detections)))

    # Reset class counts for the current frame
    frame_class_counts = {}

    # Get the latest GPS data from the queue (non-blocking)
    try:
        lat, lon = gps_queue.get_nowait()
    except queue.Empty:
        pass  # No new GPS data available

    if lat is not None and lon is not None:
        print(f"GPS Coordinates: Latitude = {lat}, Longitude = {lon}")

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for idx, detection in enumerate(detections):
        if detection.Confidence > 0.4:  # Only save high-confidence detections
            unique_id = str(uuid.uuid4())[:4]
            filename = os.path.join(args.snapshots, f"{timestamp}-{idx}-{unique_id}.jpg")

            # Create a deep copy of the image
            img_copy = cudaAllocMapped(width=img.width, height=img.height, format=img.format)
            cudaMemcpy(img_copy, img)  # Copy the image data

            # Add the copied image to the queue for saving
            image_queue.put((filename, img_copy))

            # Collect detection data for database insert
            class_name = net.GetClassDesc(detection.ClassID)  # Get the class name (e.g., 'pothole-low')
            classification, severity = split_classification_and_severity(class_name)  # Split into classification and severity
            confidence = detection.Confidence  # Get the confidence score

            # Update frame class counts
            if classification in frame_class_counts:
                frame_class_counts[classification] += 1
            else:
                frame_class_counts[classification] = 1
            
            # Push detection data to the database queue
            db_queue.put((timestamp, lat, lon, classification, severity, confidence, filename))

    # Update global class counts (tally)
    for class_name, count in frame_class_counts.items():
        if class_name in global_class_counts:
            global_class_counts[class_name] += count
        else:
            global_class_counts[class_name] = count

    # Render the global class counts on the image (upper-left corner)
    y_offset = 30
    for class_name, count in global_class_counts.items():
        text = f"{class_name}: {count}"
        font.OverlayText(img, img.width, img.height, text, 10, y_offset, font.White, font.Gray40)
        y_offset += 30

    # Render the GPS coordinates on the image (upper-right corner)
    if lat is not None and lon is not None:
        gps_text = f"Lat: {lat:.6f}, Lon: {lon:.6f}"
        text_width = len(gps_text) * 10  # Approximate width of the text (adjust as needed)
        x_offset = img.width - text_width - 270  # Position text 20 pixels from the right edge
        y_offset = 30  # Position text 30 pixels from the top edge
        font.OverlayText(img, img.width, img.height, gps_text, x_offset, y_offset, font.White, font.Gray40)

    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break

# Signal the database worker thread to exit
db_queue.put(None)

# Signal the GPS worker thread to exit
gps_thread.join()

# Signal the image saving worker thread to exit
image_queue.put((None, None))
image_thread.join()

# Signal the image directory monitor thread to exit
monitor_thread.join()

# Wait for the database worker thread to finish
db_thread.join()
