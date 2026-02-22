#!/usr/bin/env python3
# code taken from gen2-syncing/host-multiple-OAK-sync.py example
# and adapted for our purposes
import cv2
import math
import depthai as dai
import contextlib
import argparse
from datetime import timedelta
import time

parser = argparse.ArgumentParser(epilog='Press C to capture a set of frames.')
parser.add_argument('-f', '--fps', type=float, default=30,
                    help='Camera sensor FPS, applied to all cams')

args = parser.parse_args()
Pn = -2
Pn1 = -1



def create_pipeline():
    # Start defining a pipeline
    pipeline = dai.Pipeline()


    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    cam.setPreviewSize(1920, 1080)
    cam.preview.link(xout.input)

    cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam.setFps(args.fps)
    return pipeline


# https://docs.python.org/3/library/contextlib.html#contextlib.ExitStack
with contextlib.ExitStack() as stack:
    device_infos = dai.Device.getAllAvailableDevices()

    if len(device_infos) == 0: raise RuntimeError("No devices found!")
    else: print("Found", len(device_infos), "devices")
    queues = []

    for device_info in device_infos:
        # Note: the pipeline isn't set here, as we don't know yet what device it is.
        # The extra arguments passed are required by the existing overload variants
        openvino_version = dai.OpenVINO.Version.VERSION_2021_4
        usb2_mode = False
        device = stack.enter_context(dai.Device(openvino_version, device_info, usb2_mode))



        # Get a customized pipeline based on identified device type
        device.startPipeline(create_pipeline())

        # Output queue will be used to get the rgb frames from the output defined above
        for cam in {'rgb'}:
            queues.append({
                'queue': device.getOutputQueue(name=cam, maxSize=4, blocking=False),
                'msgs': [], # Frame msgs
                'mx': device.getMxId(),
                'cam': cam
            })


    def check_sync(queues, timestamp):
        matching_frames = []
        for q in queues:
            for i, msg in enumerate(q['msgs']):
                time_diff = abs(msg.getTimestamp() - timestamp)
                # So below 17ms @ 30 FPS => frames are in sync
                if time_diff <= timedelta(milliseconds=math.ceil(500 / args.fps)):
                    matching_frames.append(i)
                    break

        if len(matching_frames) == len(queues):
            # We have all frames synced. Remove the excess ones
            for i, q in enumerate(queues):
                q['msgs'] = q['msgs'][matching_frames[i]:]
            return True
        else:
            return False
    time_start = time.time()
    record = False
    while True:
        for q in queues:
            new_msg = q['queue'].tryGet()
            if new_msg is not None:
                q['msgs'].append(new_msg)
                if check_sync(queues, new_msg.getTimestamp()):
                    for q in queues:
                        def scale_img(img, scale):
                            return cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
                        #frame = q['msgs'].pop(0).getCvFrame()
                        #cv2.imshow(f"{q['cam']} - {q['mx']}", frame)
                        if q['mx'] == "1844301071990E0900":  # camera 1
                            frame1 = q['msgs'].pop(0).getCvFrame()
                            #cv2.imshow(f"Preview - {q['mx']}", frame1)
                        if q['mx'] == "18443010F1E6FF0800":  # camera 2
                            frame2 = q['msgs'].pop(0).getCvFrame()
                            #cv2.imshow(f"Preview - {q['mx']}", frame2)
                        if q['mx'] == "184430108124100900":  # camera 3
                            frame3 = q['msgs'].pop(0).getCvFrame()
                            #cv2.imshow(f"Preview - {q['mx']}", frame3)
                        if time_start+10 < time.time():
                            frame1s = scale_img(frame1, 1/3)
                            frame2s = scale_img(frame2, 1/3)
                            frame3s = scale_img(frame3, 1/3)
                            framec = cv2.hconcat([frame1s, frame2s, frame3s])
                            cv2.imshow("Preview", framec)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('p'):
            Pn += 2
            Pn1 += 2
            cv2.imwrite(f'f{Pn}.png', frame1)
            cv2.imwrite(f'f{Pn1}.png', frame2)
            print("saved")
        """if key == ord('r'):
            record = True
            print("recording")

        if key == ord('s') and record:
            record = False
            print("stopped recording")

        writer1 = cv2.VideoWriter('basicvideo1.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (1280, 720))
        writer2 = cv2.VideoWriter('basicvideo2.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (1280, 720))
        while record:
            writer1.write(frame1)
            writer2.write(frame2)
        writer1.release()
        writer2.release()"""




