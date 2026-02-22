#!/usr/bin/env python3

import cv2
import depthai as dai
import contextlib
import blobconverter
import argparse
import threading
from pathlib import Path

from depthai_sdk.managers import PipelineManager, NNetManager, BlobManager, PreviewManager
from depthai_sdk import FPSHandler, Previews, getDeviceInfo, downloadYTVideo

from pose import getKeypoints, getValidPairs, getPersonwiseKeypoints

import numpy as np


shaves = 6
labelMap = ["background","person"]

blob_path = blobconverter.from_zoo(name="human-pose-estimation-0001", shaves=shaves)

colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 0],
          [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 0, 255], [255, 0, 0],
          [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]
POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 17], [5, 16]]

running = True
pose = None
keypoints_list = None
detected_keypoints = None
personwiseKeypoints = None



def decode_thread(in_queue):
    global keypoints_list, detected_keypoints, personwiseKeypoints

    while running:
        try:
            raw_in = in_queue.get()
        except RuntimeError:
            return
        fps.tick('nn')
        heatmaps = np.array(raw_in.getLayerFp16('Mconv7_stage2_L2')).reshape((1, 19, 32, 57))
        pafs = np.array(raw_in.getLayerFp16('Mconv7_stage2_L1')).reshape((1, 38, 32, 57))
        heatmaps = heatmaps.astype('float32')
        pafs = pafs.astype('float32')
        outputs = np.concatenate((heatmaps, pafs), axis=1)

        new_keypoints = []
        new_keypoints_list = np.zeros((0, 3))
        keypoint_id = 0

        for row in range(18):
            probMap = outputs[0, row, :, :]
            probMap = cv2.resize(probMap, nm.inputSize)  # (456, 256)
            keypoints = getKeypoints(probMap, 0.3)
            new_keypoints_list = np.vstack([new_keypoints_list, *keypoints])
            keypoints_with_id = []

            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                keypoint_id += 1

            new_keypoints.append(keypoints_with_id)

        valid_pairs, invalid_pairs = getValidPairs(outputs, w=nm.inputSize[0], h=nm.inputSize[1], detected_keypoints=new_keypoints)
        newPersonwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, new_keypoints_list)

        detected_keypoints, keypoints_list, personwiseKeypoints = (new_keypoints, new_keypoints_list, newPersonwiseKeypoints)

def show(frame):
    global keypoints_list, detected_keypoints, personwiseKeypoints, nm

    if keypoints_list is not None and detected_keypoints is not None and personwiseKeypoints is not None:
        scale_factor = frame.shape[0] / nm.inputSize[1]
        offset_w = int(frame.shape[1] - nm.inputSize[0] * scale_factor) // 2

        def scale(point):
            return int(point[0] * scale_factor) + offset_w, int(point[1] * scale_factor)
        points = []
        for i in range(18):
            for j in range(len(detected_keypoints[i])):
                cv2.circle(frame, scale(detected_keypoints[i][j][0:2]), 5, colors[i], -1, cv2.LINE_AA)
                points.append(scale(detected_keypoints[i][j][0:2]))
        print(points)
        for i in range(17):
            for n in range(len(personwiseKeypoints)):
                index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                if -1 in index:
                    continue
                B = np.int32(keypoints_list[index.astype(int), 0])
                A = np.int32(keypoints_list[index.astype(int), 1])
                cv2.line(frame, scale((B[0], A[0])), scale((B[1], A[1])), colors[i], 3, cv2.LINE_AA)


print("Starting pipeline...")

# This can be customized to pass multiple parameters
"""def getPipeline():
    # Start defining a pipeline
    nm = NNetManager(inputSize=(456, 256))
    pm = PipelineManager()
    pm.setNnManager(nm)

    fps = FPSHandler()
    pm.createColorCam(previewSize=(456, 256), xout=True)

    nn = nm.createNN(pm.pipeline, pm.nodes, source=Previews.color.name if True else "host",
                     blobPath=Path(blob_path), fullFov=True)
    pm.addNn(nn=nn)"""


# https://docs.python.org/3/library/contextlib.html#contextlib.ExitStack
with contextlib.ExitStack() as stack:
    device_infos = dai.Device.getAllAvailableDevices()
    if len(device_infos) == 0:
        raise RuntimeError("No devices found!")
    else:
        print("Found", len(device_infos), "devices")
    devices = {}

    for device_info in device_infos:
        # Note: the pipeline isn't set here, as we don't know yet what device it is.
        # The extra arguments passed are required by the existing overload variants
        openvino_version = dai.OpenVINO.Version.VERSION_2021_4
        usb2_mode = False
        device = stack.enter_context(dai.Device(openvino_version, device_info, usb2_mode))

        # Note: currently on POE, DeviceInfo.getMxId() and Device.getMxId() are different!
        print("=== Connected to " + device_info.getMxId())
        mxid = device.getMxId()
        cameras = device.getConnectedCameras()
        usb_speed = device.getUsbSpeed()

        # Get a customized pipeline based on identified device type
        nm = NNetManager(inputSize=(456, 256))
        pm = PipelineManager()
        pm.setNnManager(nm)

        fps = FPSHandler()
        pm.createColorCam(previewSize=(456, 256), xout=True)

        nn = nm.createNN(pm.pipeline, pm.nodes, source=Previews.color.name if True else "host",
                         blobPath=Path(blob_path), fullFov=True)
        pm.addNn(nn=nn)



        pv = PreviewManager(display=[Previews.color.name], nnSource=Previews.color.name, fpsHandler=fps)
        pv.createQueues(device)
        nm.createQueues(device)
        seq_num = 1

        t = threading.Thread(target=decode_thread, args=(nm.outputQueue,))
        t.start()

        # Output queue will be used to get the rgb frames from the output defined above
        devices[mxid] = {
            'rgb': device.getOutputQueue(name="rgb"),
            'nn': device.getOutputQueue(name="nn"),
        }


    while True:
        fps.nextIter()
        for mxid, q in devices.items():
            if q['nn'].has():
                pv.prepareFrames()
                frame = pv.get(Previews.color.name)
                if debug:
                    show(frame)
                    cv2.putText(frame, f"RGB FPS: {round(fps.tickFps(Previews.color.name), 1)}", (5, 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                    cv2.putText(frame, f"NN FPS:  {round(fps.tickFps('nn'), 1)}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0))
                    pv.showFrames()

                #cv2.imshow(f"Preview - {mxid}", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break


        running = False

    t.join()
    fps.printStatus()
    if not args.camera:
        cap.release()