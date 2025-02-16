import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread, Timer
import importlib.util
import serial

# Define a class to handle the UART communication and bump height
class BumpHeightSender:
    def __init__(self, uart_port='/dev/ttyAMA0', baudrate=9600, interval=0.5):
        self.uart = serial.Serial(uart_port, baudrate=baudrate, timeout=1)
        self.latest_bump_height = 0  # Store the latest bump height
        self.interval = interval  # UART send interval in seconds
        self.timer = None

    def set_bump_height(self, height):
        """Update the latest bump height."""
        self.latest_bump_height = height

    def send_bump_height(self):
        """Send the latest bump height over UART."""
        bump_height = str(self.latest_bump_height)  # Convert to string
        self.uart.write(bump_height.encode())       # Send as ASCII character
        self.timer = Timer(self.interval, self.send_bump_height)  # Schedule next call
        self.timer.start()

    def stop(self):
        """Stop the timer if it's running."""
        if self.timer:
            self.timer.cancel()

# VideoStream class for threaded webcam access
class VideoStream:
    """Camera object that controls video streaming"""
    def __init__(self, resolution=(640, 480), framerate=30):
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in', required=True)
parser.add_argument('--graph', help='Name of the .tflite file', default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file', default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold', default=0.9)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH', default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator', action='store_true')
args = parser.parse_args()

# Parse resolution
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow Lite
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# Load model and labels
if use_TPU and GRAPH_NAME == 'detect.tflite':
    GRAPH_NAME = 'edgetpu.tflite'

CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
if labels[0] == '???':
    del(labels[0])

if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

outname = output_details[0]['name']
if 'StatefulPartitionedCall' in outname:
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize FPS calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize BumpHeightSender
bump_sender = BumpHeightSender(uart_port='/dev/ttyAMA0', baudrate=9600, interval=0.5)
bump_sender.send_bump_height()

# Start video stream
videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
time.sleep(1)

# Main detection loop
while True:
    t1 = cv2.getTickCount()
    frame1 = videostream.read()
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    for i in range(len(scores)):
        if scores[i] > min_conf_threshold and scores[i] <= 1.0:
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))
            bump_height = ymax - ymin

            # Update the bump height using the sender instance
            if 160 <= bump_height <= 180:
                 bump_sender.set_bump_height(1)
            elif 180 < bump_height <= 200:
                 bump_sender.set_bump_height(2)
            elif 200 < bump_height <= 220:
                 bump_sender.set_bump_height(3)
            elif 220 < bump_height <= 240:
                 bump_sender.set_bump_height(4)
            elif 240 < bump_height <= 260:
                 bump_sender.set_bump_height(5)
            elif 260 < bump_height <= 280:
                 bump_sender.set_bump_height(6)
            elif 280 < bump_height <= 300:
                 bump_sender.set_bump_height(7)
            elif 300 < bump_height <= 320:
                 bump_sender.set_bump_height(8)
            elif 320 < bump_height <= 340:
                 bump_sender.set_bump_height(9)     
            else:
                 bump_sender.set_bump_height(0)  # No valid bump height detected

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            object_name = labels[int(classes[i])]
            label = f'{object_name}: {int(scores[i] * 100)}%'
            cv2.putText(frame, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            bump_label = f'Bump Height: {bump_sender.latest_bump_height}'
            cv2.putText(frame, bump_label, (xmin, ymax + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1
    cv2.putText(frame, f'FPS: {frame_rate_calc:.2f}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow('Object Detector', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Stop all processes
bump_sender.stop()
cv2.destroyAllWindows()
videostream.stop()
