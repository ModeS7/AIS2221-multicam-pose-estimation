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
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt

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
    time_p_start = time.time()
    record = False
    photo = False
    photos = False
    map1x = np.genfromtxt('map1x.dat', delimiter=',')
    map1y = np.genfromtxt('map1y.dat', delimiter=',')
    map2x = np.genfromtxt('map2x.dat', delimiter=',')
    map2y = np.genfromtxt('map2y.dat', delimiter=',')
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
                            cv2.imshow("Camera 1", frame1)
                            img1_rect = cv2.remap(frame1, map1x, map1y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
                            frame1s = scale_img(frame1, 0.5)
                            #cv2.imshow(f"Preview - {q['mx']}", frame1)
                        if q['mx'] == "18443010F1E6FF0800":  # camera 2
                            frame2 = q['msgs'].pop(0).getCvFrame()
                            cv2.imshow("Camera 2", frame2)
                            img2_rect = cv2.remap(frame2, map2x, map2y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
                            frame2s = scale_img(frame2, 0.5)
                            #cv2.imshow(f"Preview - {q['mx']}", frame2)
                        if q['mx'] == "184430108124100900":  # camera 3
                            frame3 = q['msgs'].pop(0).getCvFrame()
                            #cv2.imshow(f"Preview - {q['mx']}", frame3)
                        if time_start+5 < time.time():


                            framec = cv2.hconcat([frame1s, frame2s])
                            #cv2.imshow("Preview", framec)




                            #img1_rect = cv2.remap(frame1, map1x, map1y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
                            #img2_rect = cv2.remap(frame2, map2x, map2y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

                            """img1_rect = cv2.imread('rectified_1.png', 0)
                            img2_rect = cv2.imread('rectified_2.png', 0)"""



                            # ------------------------------------------------------------
                            # CALCULATE DISPARITY (DEPTH MAP)
                            # Adapted from: https://github.com/opencv/opencv/blob/master/samples/python/stereo_match.py
                            # and: https://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html

                            # StereoSGBM Parameter explanations:
                            # https://docs.opencv.org/4.5.0/d2/d85/classcv_1_1StereoSGBM.html

                            # Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
                            block_size = 7
                            min_disp = -32
                            max_disp = 32
                            # Maximum disparity minus minimum disparity. The value is always greater than zero.
                            # In the current implementation, this parameter must be divisible by 16.
                            num_disp = max_disp - min_disp
                            # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
                            # Normally, a value within the 5-15 range is good enough
                            uniquenessRatio = 9
                            # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
                            # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
                            speckleWindowSize = 0
                            # Maximum disparity variation within each connected component.
                            # If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
                            # Normally, 1 or 2 is good enough.
                            speckleRange = 2
                            disp12MaxDiff = 0

                            stereo = cv2.StereoSGBM_create(
                                minDisparity=min_disp,
                                numDisparities=num_disp,
                                blockSize=block_size,
                                uniquenessRatio=uniquenessRatio,
                                speckleWindowSize=speckleWindowSize,
                                speckleRange=speckleRange,
                                disp12MaxDiff=disp12MaxDiff,
                                P1=8 * 1 * block_size * block_size,
                                P2=32 * 1 * block_size * block_size,
                            )
                            disparity_SGBM = stereo.compute(img1_rect, img2_rect)

                            # Normalize the values to a range from 0..255 for a grayscale image
                            disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                                                          beta=0, norm_type=cv2.NORM_MINMAX)
                            disparity_SGBM = np.uint8(disparity_SGBM)
                            #plt.figure(80)
                            cv2.imshow('disparity', disparity_SGBM)
                            #plt.colorbar()
                            #plt.show()



        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        if key == ord('c'):
            time_p_start = time.time()
            photo = True
        if key == ord('p') or photos:
            Pn += 2
            Pn1 += 2
            cv2.imwrite(f'f{Pn}.png', frame1)
            cv2.imwrite(f'f{Pn1}.png', frame2)
            print("saved")

        if time.time() > time_p_start + 10 and photo:
            photos = True
            photo = False

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




