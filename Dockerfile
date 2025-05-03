# Use the base image
FROM ultralytics/ultralytics:latest-jetson-jetpack4

# Install additional libraries
RUN pip install pynmea2 pyserial numpy==1.23.5 fpdf2

# Set environment variables for X11
ENV DISPLAY=${DISPLAY}
ENV QT_X11_NO_MITSHM=1

# Set the working directory
WORKDIR /home/geopavenet

# Command to run when the container starts
CMD ["bash"]
