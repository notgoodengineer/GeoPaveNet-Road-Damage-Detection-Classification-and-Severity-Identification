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
#   python3 detectnet-snap.py --snanpshots /path/to/snapshots <input_URI> <output_URI>
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

from jetson_inference import detectNet
from jetson_utils import (videoSource, videoOutput, saveImage, Log,
                          cudaAllocMapped, cudaCrop, cudaDeviceSynchronize)

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
            classification TEXT,
            confidence REAL,
            image_path TEXT
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
            INSERT INTO detections (timestamp, latitude, longitude, classification, confidence, image_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', data)
        conn.commit()

        # Mark the task as done
        db_queue.task_done()

    # Close the database connection
    conn.close()

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

# Start the GPS worker thread
gps_thread = threading.Thread(target=gps_worker, args=(gps_queue,))
gps_thread.daemon = True  # Daemonize thread to exit when the main program exits
gps_thread.start()

# Start the database worker thread
db_thread = threading.Thread(target=db_worker, args=(db_queue,))
db_thread.daemon = True  # Daemonize thread to exit when the main program exits
db_thread.start()

# create video output object 
output = videoOutput(args.output, argv=sys.argv)
	
# load the object detection network
net = detectNet(args.network, sys.argv, args.threshold)

# create video sources
input = videoSource(args.input, argv=sys.argv)

# Initialize GPS data
lat, lon = None, None

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

    # Get the latest GPS data from the queue (non-blocking)
    try:
        lat, lon = gps_queue.get_nowait()
    except queue.Empty:
        pass  # No new GPS data available

    if lat is not None and lon is not None:
        print(f"GPS Coordinates: Latitude = {lat}, Longitude = {lon}")

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for idx, detection in enumerate(detections):
        print(detection)
        roi = (int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom))
        snapshot = cudaAllocMapped(width=roi[2]-roi[0], height=roi[3]-roi[1], format=img.format)
        cudaCrop(img, snapshot, roi)
        cudaDeviceSynchronize()
        
        # Save image with GPS coordinates in the filename
        if lat is not None and lon is not None:
            filename = os.path.join(args.snapshots, f"{timestamp}-{idx}-lat{lat}-lon{lon}.jpg")
        else:
            filename = os.path.join(args.snapshots, f"{timestamp}-{idx}.jpg")
        
        saveImage(filename, snapshot)
        del snapshot

        # Collect detection data for database insert
        classification = net.GetClassDesc(detection.ClassID)  # Get the classification label
        confidence = detection.Confidence  # Get the confidence score
        
        # Push detection data to the database queue
        db_queue.put((timestamp, lat, lon, classification, confidence, filename))

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

# Wait for the database worker thread to finish
db_thread.join()
