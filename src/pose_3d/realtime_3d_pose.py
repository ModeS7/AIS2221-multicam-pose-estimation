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
import numpy as np
from numpy import genfromtxt
import mediapipe as mp
from utils import DLT, write_keypoints_to_disk
import matplotlib.pyplot as plt
plt.style.use('seaborn')

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

frame_shape = [1920, 1080]

#add here if you need more keypoints
pose_keypoints = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28]


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
    cam.setPreviewSize(frame_shape[0], frame_shape[1])
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

    pose0 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose1 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # containers for detected keypoints for each camera. These are filled at each frame.
    # This will run you into memory issue if you run the program without stop
    kpts_cam0 = []
    kpts_cam1 = []
    kpts_3d = []

    torso = [[0, 1], [1, 7], [7, 6], [6, 0]]
    armr = [[1, 3], [3, 5]]
    arml = [[0, 2], [2, 4]]
    legr = [[6, 8], [8, 10]]
    legl = [[7, 9], [9, 11]]
    body = [torso, arml, armr, legr, legl]
    colors = ['red', 'blue', 'green', 'black', 'orange']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

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
                            frame0 = q['msgs'].pop(0).getCvFrame()
                            #cv2.imshow(f"Preview - {q['mx']}", frame1)
                        if q['mx'] == "18443010F1E6FF0800":  # camera 2
                            frame1 = q['msgs'].pop(0).getCvFrame()
                            #cv2.imshow(f"Preview - {q['mx']}", frame2)
                        if q['mx'] == "184430108124100900":  # camera 3
                            frame3 = q['msgs'].pop(0).getCvFrame()
                            #cv2.imshow(f"Preview - {q['mx']}", frame3)
                        if time_start+3 < time.time():
                            """frame1s = scale_img(frame1, 0.5)
                            frame2s = scale_img(frame2, 0.5)
                            framec = cv2.hconcat([frame1s, frame2s])
                            cv2.imshow("Preview", framec)"""
                            P0 = genfromtxt('1844301071990E0900P matrix.dat', delimiter=',')
                            P1 = genfromtxt('18443010F1E6FF0800P matrix.dat', delimiter=',')

                            #P0 = np.delete(P0, 3, 0)
                            #P1 = np.delete(P1, 3, 0)


                            # To improve performance, optionally mark the image as not writeable to
                            # pass by reference.
                            frame0.flags.writeable = False
                            frame1.flags.writeable = False
                            results0 = pose0.process(frame0)
                            results1 = pose1.process(frame1)

                            # reverse changes
                            frame0.flags.writeable = True
                            frame1.flags.writeable = True
                            frame0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR)
                            frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)

                            # check for keypoints detection
                            frame0_keypoints = []
                            if results0.pose_landmarks:
                                for i, landmark in enumerate(results0.pose_landmarks.landmark):
                                    if i not in pose_keypoints: continue  # only save keypoints that are indicated in pose_keypoints
                                    pxl_x = landmark.x * frame0.shape[1]
                                    pxl_y = landmark.y * frame0.shape[0]
                                    pxl_x = int(round(pxl_x))
                                    pxl_y = int(round(pxl_y))
                                    cv2.circle(frame0, (pxl_x, pxl_y), 3, (0, 0, 255),
                                               -1)  # add keypoint detection points into figure
                                    kpts = [pxl_x, pxl_y]
                                    frame0_keypoints.append(kpts)
                            else:
                                # if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
                                frame0_keypoints = [[-1, -1]] * len(pose_keypoints)

                            # this will keep keypoints of this frame in memory
                            kpts_cam0.append(frame0_keypoints)

                            frame1_keypoints = []
                            if results1.pose_landmarks:
                                for i, landmark in enumerate(results1.pose_landmarks.landmark):
                                    if i not in pose_keypoints: continue
                                    pxl_x = landmark.x * frame1.shape[1]
                                    pxl_y = landmark.y * frame1.shape[0]
                                    pxl_x = int(round(pxl_x))
                                    pxl_y = int(round(pxl_y))
                                    cv2.circle(frame1, (pxl_x, pxl_y), 3, (0, 0, 255), -1)
                                    kpts = [pxl_x, pxl_y]
                                    frame1_keypoints.append(kpts)

                            else:
                                # if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
                                frame1_keypoints = [[-1, -1]] * len(pose_keypoints)

                            # update keypoints container
                            kpts_cam1.append(frame1_keypoints)

                            # calculate 3d position
                            frame_p3ds = []
                            for uv1, uv2 in zip(frame0_keypoints, frame1_keypoints):
                                if uv1[0] == -1 or uv2[0] == -1:
                                    _p3d = [-1, -1, -1]
                                else:
                                    _p3d = DLT(P0, P1, uv1, uv2)  # calculate 3d position of keypoint
                                frame_p3ds.append(_p3d)

                            frame_p3ds = np.array(frame_p3ds).reshape((1, 12, 3))
                            #kpts_3d.append(frame_p3ds)

                            """cv2.imshow("Preview0", frame0)
                            cv2.imshow("Preview1", frame1)"""
                            frame0s = scale_img(frame0, 0.25)
                            frame1s = scale_img(frame1, 0.25)
                            framec = cv2.hconcat([frame0s, frame1s])
                            cv2.imshow("Preview", framec)



                            for kpts3d in frame_p3ds:
                                for bodypart, part_color in zip(body, colors):
                                    for _c in bodypart:
                                        print(kpts3d[_c])
                                        ax.plot(xs=[kpts3d[_c[0]][0], kpts3d[_c[1]][0]],
                                                ys=[kpts3d[_c[0]][1], kpts3d[_c[1]][1]],
                                                zs=[kpts3d[_c[0]][2], kpts3d[_c[1]][2]],
                                                linewidth=4, c=part_color)
                                    ax.set_xlim3d(-3, 3)
                                    ax.set_xlabel('x')
                                    ax.set_ylim3d(-3, 3)
                                    ax.set_ylabel('y')
                                    ax.set_zlim3d(-2, 0)
                                    ax.set_zlabel('z')
                                plt.pause(0.1)
                                ax.cla()


        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        if key == ord('c'):
            time_p_start = time.time()
            photo = True
        if key == ord('p') or photos:
            Pn += 2
            Pn1 += 2
            cv2.imwrite(f'f{Pn}.png', frame0)
            cv2.imwrite(f'f{Pn1}.png', frame1)
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
    cv2.destroyAllWindows()
    """np.array(kpts_cam0), np.array(kpts_cam1), np.array(kpts_3d)
    write_keypoints_to_disk('kpts_cam0.dat', kpts_cam0)
    write_keypoints_to_disk('kpts_cam1.dat', kpts_cam1)
    write_keypoints_to_disk('kpts_3d.dat', kpts_3d)"""
    print(":)")






