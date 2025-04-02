import tkinter as tk
from tkinter import ttk, messagebox
import threading
import sqlite3
import os
from datetime import datetime
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput

# Global variables
is_running = False
detection_thread = None

# Function to start/stop the detection model
def toggle_detection():
    global is_running, detection_thread

    if is_running:
        # Stop the detection model
        is_running = False
        start_stop_button.config(text="Start Detection")
        status_label.config(text="Model Stopped", fg="red")
    else:
        # Start the detection model
        is_running = True
        start_stop_button.config(text="Stop Detection")
        status_label.config(text="Model Running", fg="green")

        # Run the detection model in a separate thread
        detection_thread = threading.Thread(target=run_detection)
        detection_thread.start()

# Function to run the detection model
def run_detection():
    global is_running

    # Initialize the detection model
    net = detectNet("ssd-mobilenet-v2", threshold=0.5)
    input = videoSource("/dev/video0")
    output = videoOutput("display://0")

    while is_running:
        img = input.Capture()
        if img is None:
            continue

        detections = net.Detect(img)
        output.Render(img)
        output.SetStatus(f"Network FPS: {net.GetNetworkFPS()}")

        # Save detections to the database
        save_detections_to_db(detections)

# Function to save detections to the database
def save_detections_to_db(detections):
    conn = sqlite3.connect('detections.db')
    cursor = conn.cursor()

    for detection in detections:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        classification = net.GetClassDesc(detection.ClassID)
        confidence = detection.Confidence
        severity = None  # Placeholder for future use

        cursor.execute('''
            INSERT INTO detections (timestamp, classification, confidence, severity)
            VALUES (?, ?, ?, ?)
        ''', (timestamp, classification, confidence, severity))

    conn.commit()
    conn.close()

# Function to view the database
def view_database():
    conn = sqlite3.connect('detections.db')
    cursor = conn.cursor()

    # Fetch all records from the database
    cursor.execute("SELECT * FROM detections")
    rows = cursor.fetchall()

    # Clear the treeview
    for row in treeview.get_children():
        treeview.delete(row)

    # Insert records into the treeview
    for row in rows:
        treeview.insert("", "end", values=row)

    conn.close()

# Initialize the GUI
root = tk.Tk()
root.title("DetectNetX GUI")
root.geometry("800x600")

# Start/Stop Button
start_stop_button = ttk.Button(root, text="Start Detection", command=toggle_detection)
start_stop_button.pack(pady=10)

# Status Label
status_label = ttk.Label(root, text="Model Stopped", foreground="red")
status_label.pack(pady=5)

# Database Viewer
treeview = ttk.Treeview(root, columns=("ID", "Timestamp", "Classification", "Confidence", "Severity"), show="headings")
treeview.heading("ID", text="ID")
treeview.heading("Timestamp", text="Timestamp")
treeview.heading("Classification", text="Classification")
treeview.heading("Confidence", text="Confidence")
treeview.heading("Severity", text="Severity")
treeview.pack(fill="both", expand=True, padx=10, pady=10)

# View Database Button
view_db_button = ttk.Button(root, text="View Database", command=view_database)
view_db_button.pack(pady=10)

# Run the GUI
root.mainloop()
