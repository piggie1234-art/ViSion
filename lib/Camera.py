from ctypes import *
import ctypes
import numpy as np
import cv2
import time
import sys,os
import io
import queue
import threading
import sys
from jetson_utils import videoSource, cudaToNumpy
from lib.Rectify_stereo import RectifyCam2

class POECamera():
    def __init__(self,config) -> None:
#         self.left_gs = (
#     "rtspsrc location=rtsp://192.168.1.10:554/user=admin&password=&channel=1&stream=0.sdp?real_stream ! "
#     "rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! "
#     "video/x-raw, format=(string)BGRx ! "
#     "videoconvert ! "
#     "video/x-raw, format=(string)BGR ! appsink"
# )
#         self.right_gs = (
#     "rtspsrc location=rtsp://192.168.1.11:554/user=admin&password=&channel=2&stream=0.sdp?real_stream ! "
#     "rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! "
#     "video/x-raw, format=(string)BGRx ! "
#     "videoconvert ! "
#     "video/x-raw, format=(string)BGR ! appsink"
    #time.sleep(0.08)
    # image = input.Capture(format='rgb8', timeout=1000)
   
	
    # if image is None:  # if a timeout occurred
    #     continue
    # array = cudaToNumpy(image)
    # img = cv2.resize(array,(640,480))
    # cv2.imshow("11",img[:,:,::-1])
    # cv2.waitKey(1)
# )
        self.left_rtsp = config.left_rtsp          
        self.right_rtsp = config.right_rtsp
        #self.left_rtsp = "rtsp://192.168.1.11:554/user=admin&password=&channel=1&stream=0.sdp?real_stream"
        #self.right_rtsp = "rtsp://192.168.1.12:554/user=admin&password=&channel=1&stream=0.sdp?real_stream"
        #sys.argv.append("--input-codec=h265")
        self.left_frame_queue = queue.Queue(maxsize=2)
        self.right_frame_queue = queue.Queue(maxsize=2)
        self.dist_left_queue = queue.Queue(maxsize=2)
        self.dist_right_queue = queue.Queue(maxsize=2)
        self.rec_cam = RectifyCam2(config.CameraRectify)
        self.org_flag = config.show_origin_image
        #self.left_cap = cv2.VideoCapture(self.left_rtsp)
        #self.right_cap = cv2.VideoCapture(self.right_rtsp)
        try:
            self.left_cap = videoSource(self.left_rtsp, argv=sys.argv)
        except:
            self.left_cap = None
            print("Failed to open left cam.")
        
        try:
            self.right_cap = videoSource(self.right_rtsp , argv=sys.argv)
        except:
            self.right_cap = None 
            print("Failed to open right cam.")
    
    def start_capture_frame(self):                                         #左右线程进行捕获图像帧，无法显示图像，则开启另一个线程进行畸变矫正
        left_capture_thread = threading.Thread(target=self.capture_frames, args=(self.left_rtsp, self.left_frame_queue,self.left_cap))
        left_capture_thread.start()
        right_capture_thread = threading.Thread(target=self.capture_frames, args=(self.right_rtsp, self.right_frame_queue, self.right_cap))
        right_capture_thread.start()
        if not self.org_flag:
            rectify_t = threading.Thread(target=self.rectify_frames)
            rectify_t.start()


    def rectify_frames(self):
        while True:
            time.sleep(0.05)
            if self.left_frame_queue.empty(): #检查队列中是否有图片，没有进行下一次循环，有的话取出原来的图片，把新的图片放入队列进行矫正,检查队列的容量是否小于maxsize，小于把图片放入队列，不小于取出队列中原先的图片再放入新的图片
                continue
            else:
                img = self.left_frame_queue.get()
                l_img = self.rec_cam.remap(img)
                
            if  self.dist_left_queue.qsize() < 2:  # Check the size of the queue before putting a new frame
                self.dist_left_queue.put(l_img)
            else:
                self.dist_left_queue.get()  # Remove the oldest frame from the queue
                self.dist_left_queue.put(l_img)
            
            if self.right_frame_queue.empty():
                continue
            else:
                img = self.right_frame_queue.get()
                start = time.time()
                r_img = self.rec_cam.remap(img,1)
                #print(f'remap time:{time.time()- start}')
            if  self.dist_right_queue.qsize() < 2:  # Check the size of the queue before putting a new frame
                self.dist_right_queue.put(r_img)
            else:
                self.dist_right_queue.get()  # Remove the oldest frame from the queue
                self.dist_right_queue.put(r_img)

    def capture_frames(self,rtsp_url, frame_queue,cap):
        #cap = cv2.VideoCapture(rtsp_url)
        while True:
            time.sleep(0.08)
            if cap is None:
                return
            try:
                image = cap.Capture(format='rgb8', timeout=1000)
            except:
                print("image capture failed.") 
   
	
            if image is None:  # if a timeout occurred
                continue
            frame = cudaToNumpy(image)
            # ret, frame = cap.read()
            # if not ret:
            #     print('can not cap image.')
            #     cap.release()
            #     cap = cv2.VideoCapture(rtsp_url)
            #     break
            if frame_queue.qsize() < 2:  # Check the size of the queue before putting a new frame
                frame_queue.put(frame)
            else:
                frame_queue.get()  # Remove the oldest frame from the queue
                frame_queue.put(frame)
        #cap.release()
    def rtsp_stream(self):
        img = None
        # 检查视频捕捉对象是否已经打开
        if not self.cap.isOpened():
            print("无法打开视频捕捉对象")
            return -1,img

        # 从视频捕捉对象中读取一帧
        ret, img = self.cap.read()

        # 检查是否成功读取到视频帧
        if not ret:
            print("无法读取视频帧")
            return -1,img
        
        return 1,img


if  __name__ == '__main__':


    camera = POECamera()
  
    while True:
        ret,img = camera.rtsp_stream()
        if not ret:
            break

        # Process the frame (e.g., display or save)
        cv2.imshow('Video Stream', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #cap.release()
    cv2.destroyAllWindows()

    #img = camera.ImageSnap()


    ## get one char from terminal using python
    s = sys.stdin.read(1)

    if s == 'f':
        camera.ChangeFocus(True)
    elif s == 'n':
        camera.ChangeFocus(False)



