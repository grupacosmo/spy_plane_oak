from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import argparse
import json
from datetime import datetime

nnPathDefault = str((Path(__file__).parent / Path('/Users/konor/Documents/AI/depthai-python/examples/models/mobilenet-ssd_openvino_2021.4_6shave.blob')).resolve().absolute())
parser = argparse.ArgumentParser()
parser.add_argument('nnPath', nargs='?', help="Path to mobilenet detection network blob", default=nnPathDefault)
parser.add_argument('-s', '--sync', action="store_true", help="Sync RGB output with NN output", default=False)
args = parser.parse_args()


# encoding


if not Path(nnPathDefault).exists():
    import sys
    raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

# MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]



# Create pipeline
pipeline = dai.Pipeline()

objectTracker = pipeline.create(dai.node.ObjectTracker)

trackerOut = pipeline.create(dai.node.XLinkOut)
trackerOut.setStreamName("tracklets")

objectTracker.setDetectionLabelsToTrack([7, 15])  # track labels numbers
        # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
objectTracker.setTrackerType(dai.TrackerType.SHORT_TERM_KCF)
        # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)
nnNetworkOut = pipeline.create(dai.node.XLinkOut)

logger = pipeline.create(dai.node.SystemLogger)
logger.setRate(1)
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("sysinfo")
logger.out.link(xout.input)

xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")
nnNetworkOut.setStreamName("nnNetwork");

# Properties
camRgb.setPreviewSize(300, 300)
camRgb.setInterleaved(False)
camRgb.setFps(40)
# Define a neural network that will make predictions based on the source frames
nn.setConfidenceThreshold(0.8)
nn.setBlobPath(args.nnPath)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)

output = []
output_file = Path('output.json')

# Linking
if args.sync:
    nn.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

camRgb.preview.link(nn.input)
nn.out.link(nnOut.input)
nn.outNetwork.link(nnNetworkOut.input)

objectTracker.passthroughTrackerFrame.link(xoutRgb.input)
fullFrameTracking = False
if fullFrameTracking:
    camRgb.video.link(objectTracker.inputTrackerFrame)
else:
    nn.passthrough.link(objectTracker.inputTrackerFrame)

nn.passthrough.link(objectTracker.inputDetectionFrame)
nn.out.link(objectTracker.inputDetections)
objectTracker.out.link(trackerOut.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    qNN = device.getOutputQueue(name="nnNetwork", maxSize=4, blocking=False);

    tracklets = device.getOutputQueue("tracklets", 4, False)

    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 255, 255)

    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame):
        color = (255, 0, 0)
        for detection in detections:
            if labelMap[detection.label] == "person" or labelMap[detection.label] == "car":
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        # Show the frame
        cv2.imshow(name, frame)

    printOutputLayersOnce = True
    person = 0
    lost = 1
    id = 0
    while True:
        track = None
        inRgb = qRgb.get()
        inDet = qDet.get()
        inNN = qNN.get()
        track = tracklets.tryGet()
        if track is not None:
            for t in track.tracklets:
                id_set = 0
                roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
                x1 = int(roi.topLeft().x)
                y1 = int(roi.topLeft().y)
                x2 = int(roi.bottomRight().x)
                y2 = int(roi.bottomRight().y)
                status = int(t.status)
                color = (0, 255, 255)
                if status == 1:
                    appearance = appearance + 1
                    color = (255, 0, 0)
                    if appearance == 5:
                        id = id+1
                        if not id_set:
                            id_const = id
                            id_set = 1
                        output.append(f"Object {labelMap[t.label]} number {t.id} detected at {datetime.now()}")
                        output_file.write_text(json.dumps(output, indent=4))
                elif status == 0:
                    lost = 0
                    person = person + 1
                    appearance = 0
                elif status == 2:
                    color = (0, 0, 255)
                elif status == 3:
                    if appearance >= 5:
                        output.append(f"Object {labelMap[t.label]} number {t.id} lost at {datetime.now()}")
                        output_file.write_text(json.dumps(output, indent=4))

                    color = (255, 255, 0)
        if inRgb is not None:
            frame = inRgb.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)

        if inDet is not None:
            if inDet.detections and inDet.detections != detections and \
                    (labelMap[inDet.detections[0].label] == "person" or labelMap[inDet.detections[0].label] == "car"):
                detections = inDet.detections
            counter += 1

        if printOutputLayersOnce and inNN is not None:
            toPrint = 'Output layer names:'
            for ten in inNN.getAllLayerNames():
                toPrint = f'{toPrint} {ten},'
            print(toPrint)
            printOutputLayersOnce = False;

        if frame is not None:
            if track and track.tracklets == []:
                detections = []
            displayFrame("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break