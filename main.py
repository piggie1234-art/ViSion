import sys
import cv2
import numpy as np
import threading
from easydict import EasyDict as edict
import time
from UDPtransmit.UDPTransmitters import LocationTransmitUDP
sys.path.append("../yolov7")
sys.path.append("../Fast-ACVNet")
from lib.detection_model import DetectionModel
from lib.depth_model import DepthEstimateModelTRT
from lib.depth_model import DepthEstimateModelTRT,StereoBM
from lib.tracking_model import TrackingModelTRT,TrackingModelTRTv2
from lib.LocationModel import StereoLocationModel
from lib.Camera import POECamera
from collections import deque
import os
from lib.utils import plot_one_box
import yaml
import glob


class TargetLocationModule():
    def __init__(self,config):

        self.rtsp_camera = POECamera(config.StereoCam)
        self.use_img = False
        if config.from_img:
            self.use_img = True
            self.img_path= glob.glob("runs/depth_imgs/*.jpg")
        self.locate_udp = LocationTransmitUDP(config.LocationUDP.ip,target_port=config.LocationUDP.port)
        self.L_img = []
        self.R_img = []
        self.sonar_image = []
        self.L_org_img = None
        self.R_org_img = None
        self.saved_img_id = 0
        self.detection_flag = False

        self.disp_im = None
        self.disp_np = None


        self.cls_types = config.DetectionModel.class_types
        self.depth_model = DepthEstimateModel()
        self.depth_BM = StereoBM()
        #self.depth_model = DepthEstimateModelTRT(config.DepthEstimateModel)
        self.tracking_model = TrackingModelTRTv2(config.TrackingModel)
        self.det_model = DetectionModel(config.DetectionModel.model_path)
        self.detections = None
        self.locate_uv = deque(maxlen=2)
        self.xyz = deque(maxlen=2)
        self.detect_cls = 0
        self.locate_model = StereoLocationModel(self.rtsp_camera.rec_cam.new_k1,config.LocationModel)
        self.is_tracking = False
        self.retrack = False
        self.track_mask = None
        self.track_box = None
        self.det_box = None
    
        self.left_img_idx = 0
        self.right_img_idx = 0
        self.depth_idx = 0
        self.proc_img_idx =0
        self.disp_im_bm = None

    def start(self):
        self.start_grab_image()
        self.start_tracking()
        self.start_detection()
        self.start_estimate_dpth()
        self.start_udp()
        #self.start_sonar()
    
        #self.left_Vcapture()
    def start_sonar(self):
        threading.Thread(target=self.sonar_t).start()
    

    def sonar_dectection(self):
        while True:
            time.sleep(0.08)
            if len(self.sonar_image) > 0:
                pass
    
    def sonar_t(self):
        os.system('./sonarmutil/build/sonar')          
        time.sleep(10)
        while True:
            time.sleep(0.08)
            img = cv2.imread('sonarmutil/img1.jpg')
            self.sonar_image = img.copy()


    def start_udp(self):
        threading.Thread(target=self.send_udp,args=()).start()
    
    def send_udp(self):
        while True:
            time.sleep(0.1)
            if len(self.xyz) > 0:
                xyz = np.mean(self.xyz,axis=0)
                self.locate_udp.update_send_dict(self.detect_cls,xyz)
                print(xyz)
       
    def left_Vcapture(self):

        time.sleep(10)
        cap_left = threading.Thread(target=self.left_VCap_t)
        cap_left.start()

    def left_VCap_t(self):
        while True:
            time.sleep(1)
            if len(self.L_img )>0 and len(self.R_img)>0:
                self.left_img_idx += 1
                img = np.concatenate([self.L_img,self.R_img],axis=1)   #左右拼接L R图片
                for i in range(10):                                    #画10条线段，起始点，终点，颜色，线宽
                    cv2.line(img,(0,108*i),(1920*2,108*i),(0,0,255),2)
                cv2.imwrite(f"runs/img_{self.left_img_idx}.jpg",img)   #把img图像保存到runs/img_*

    def right_VCap_t(self):                                            #保存右图
        while True:
            time.sleep(1)                               
            if len(self.R_img)>0:
                self.right_img_idx += 1
                cv2.imwrite(f"runs/r_img_{self.right_img_idx}.jpg",self.R_img)
    
    def save_img(self):
        self.save_img_mut.acquire()            #获取互斥锁
        if not os.path.exists("saved_img/L"):  #检查是否存在路径，没有则创建
            os.makedirs("saved_img/L")
            os.makedirs("saved_img/R")
        
        cv2.imwrite(f"saved_img/L/{self.saved_img_id}.jpg", self.L_org_img)  #保存左图原始图像
        cv2.imwrite(f"saved_img/R/{self.saved_img_id}.jpg", self.R_org_img)  #保存右图原始图像
        if self.disp_im is not None:
            cv2.imwrite(f"saved_img/{self.saved_img_id}_D.jpg", self.disp_im[:,:,::-1]) #保存为RGB格式

        self.save_img_mut.release()            #释放互斥锁
        self.saved_img_id += 1

    def start_tracking(self):
        if self.is_tracking:         #如果is_tracking是False，则retrack设置为True
            self.retrack = True
        else:                        #如果is_tracking是True,则创建一个线程并开启
            track_t = threading.Thread(target=self.track)
            track_t.start()
    
    def track(self):
        self.is_tracking = True
        while True:
            time.sleep(0.1)
            mask = None
            if self.det_box is not None:                #如果监测框存在
                #box = self.track_box 
                img = self.L_img                        #备份左图到img
                self.track_box = self.det_box.copy()    #复制监测框给track_box
                mask = self.tracking_model.track(img,self.track_box) #获取图像掩码

            elif self.track_box is not None and self.L_img is not None:  #如果监测框存在，且左图存在，把图片，框，掩码备份给box，img，mask
                box = self.track_box                                     
                img = self.L_img
                mask = self.tracking_model.track(img,box)

            if mask is not None:                        #如果能获取mask
                mask =  mask[0,0]                       #
                y,x = np.where(mask)                    #获取掩码非零像素的位置
                xmin,xmax = np.min(x),np.max(x)         #获取x,y的最大最小位置，也就是图像的区域
                ymin,ymax = np.min(y),np.max(y)
             
                self.track_box = [[xmin,ymin],[xmax,ymax]]  #更新监测框范围
                self.mask_mut.acquire()                     #获取掩码的互斥锁
                self.track_mask = mask                      #更新掩码
                self.mask_mut.release()                     #释放互斥锁

    def start_grab_image(self):
        
        self.rtsp_camera.start_capture_frame()
        threading.Thread(target=self.get_image).start()
        show_left = threading.Thread(target=self.show_left_rtsp_img)
        show_left.start()
        # show_right = threading.Thread(target=self.show_right_rtsp_img)
        # show_right.start()
        
    def get_image(self):                                                    #从视频流中取出图片备份          
        while True:
            time.sleep(0.05)
            if not self.rtsp_camera.org_flag:
                if self.rtsp_camera.dist_right_queue.empty():               #不能显示原始图像，dist_right_queue且为空，进入下一次循环
                    continue
                r_img = self.rtsp_camera.dist_right_queue.get()[:,:,::-1]   #从相机流队列中取出右相机图片，并反转通道
                self.R_img = r_img.copy()                                   #备份右图像
                    #self.L_org_img = l_img.copy()
                    #l_img = self.stcam_rec.remap(l_img)
            else:
                if self.rtsp_camera.right_frame_queue.empty():              #能显示原始图像，且right_frame_queue为空，跳过当前循环
                    continue
                r_img = self.rtsp_camera.right_frame_queue.get()[:,:,::-1]  #从相机流队列中取出右相机图片，并反转通道
                self.R_org_img = r_img.copy()                               #拷贝原始右图像
                self.R_img = self.stcam_rec.remap(r_img)                    #矫正右图像
            
            if not self.rtsp_camera.org_flag:
                if self.rtsp_camera.dist_left_queue.empty():
                    continue
                l_img = self.rtsp_camera.dist_left_queue.get()[:,:,::-1]
                self.L_img = l_img.copy()
            else:
                if self.rtsp_camera.left_frame_queue.empty():
                    continue
                l_img = self.rtsp_camera.left_frame_queue.get()[:,:,::-1]
                self.L_org_img = l_img.copy()
                self.L_img = self.stcam_rec.remap(l_img)

    def start_estimate_dpth(self):                  #显示深度图的异常处理模块

        try:
            self.show_depth_thread_handle = threading.Thread(target=self.show_depth,args=())
            self.show_depth_thread_handle.start()
       
        except:
            print('Cannot open show depth thread.')
    
    def start_detection(self):                     #检测的异常处理模块
       
        self.detection_flag = True
      
        try:
            self.run_detection_thread_handle = threading.Thread(target=self.run_detection,args=())
            self.run_detection_thread_handle.start()
        except:
            print('Cannot open run detection thread.')

    def run_detection(self):                       #0.08秒detect一次，检查左右图片是否存在，获取互斥锁备份图片，
        #h,w,_ = self.L_img.shape
        while True:
            time.sleep(0.08)
            if len(self.L_img)>0 and len(self.R_img)>0:   #检查左右图像是否存在

                self.display_mut.acquire()                #获取展示图像的互斥锁

                l_img = self.L_img.copy()                 #备份图片
            
                self.display_mut.release()                #释放互斥锁
                
                # l_img = cv2.resize(l_img,(640,480))
                #start = time.time()
                out = self.det_model.detect(l_img)       #
                if out is not None:
                    # print(out)
                    if len(out) > 0:
                        self.detections = out
        
    def show_depth(self):

        while True:
            time.sleep(0.05)
            if len(self.L_img)>0 and len(self.R_img)>0:

                self.img_depth_mut.acquire()
                l_img = self.L_img.copy()
                r_img = self.R_img.copy()
            
                self.img_depth_mut.release()

                l_img = cv2.resize(l_img,(640,480))
                r_img = cv2.resize(r_img,(640,480))
                # l_img = self.wb.balanceWhite(l_img)
                # r_img = self.wb.balanceWhite(r_img)
                start = time.time()
                disp_im_bm = self.depth_BM.estimate_depth(l_img,r_img)
                disp_im, disp_np = self.depth_model.estimate_depth(l_img,r_img)
                #print(f"depth_est_time:{time.time()-start}")
                self.disp_np = disp_np.copy()
                self.disp_im_bm = disp_im_bm.copy()
                self.depth_idx += 1
                output_img = np.concatenate([l_img,disp_im,r_img],axis=1)
                if not os.path.exists("runs/depth_imgs"):
                    os.makedirs("runs/depth_imgs")
                if self.depth_idx % 2 == 0:
                    cv2.imwrite(f"runs/depth_imgs/disp_im_{self.depth_idx}.jpg",output_img)


    def show_left_rtsp_img(self):

        while True:
            time.sleep(0.03)
            try:
            
                start = time.time()
                if len(self.L_img) > 0:
                    l_img = self.L_img.copy()
                else:
                    continue

                if self.track_box is not None: #and self.disp_np is not None:
                    
                    #if self.disp_np is not None:
                    try:
                        
                        if self.disp_np is not None and self.track_mask is not None:
                            #print("try to plot")
                            box = [self.track_box[0][0],self.track_box[0][1],self.track_box[1][0],self.track_box[1][1]]
                            plot_one_box(l_img, box, label='1',score='0.95' ,color=(255,0,0), line_thickness=5)
                            xyz,uv = self.locate_model.locate_float_link(self.disp_im_bm,track_mask=self.track_mask)
                            #xyz,uv = self.locate_model.locate_mask(self.disp_np,self.track_mask)
                            #print(xyz)
                            if xyz is not None:
                                self.xyz.append(xyz)
                            if uv is not None:
                                self.locate_uv.append(uv) 
                        
                    except Exception as e:
                        print(f"Error occurred while lolcate the object: {e}")

                if self.detections is not None:
                    for det in self.detections:
                        det = det.cpu().detach().numpy()
                        if len(det) == 0:
                            self.det_box = None
                            #print("no detections.")
                            break
                        max_conf_idx = np.argmax(det[:,4])
                      
                        x1,y1,x2,y2,conf,cls = det[max_conf_idx]##.cpu().detach().numpy()
                        self.detect_cls = int(cls) + 1
                        xyxy = [x1,y1,x2,y2]
                        self.det_box = [[x1,y1],[x2,y2]]
                        
                        
                if len(self.locate_uv)!=0:
                    uv = np.mean(self.locate_uv,axis=0)
                    l_img = cv2.circle(l_img,(int(uv[0]),int(uv[1])),20,(0,0,255),-1)
                
                if len(self.xyz)!=0:
                    #xyz = np.mean(self.xyz,axis=0)
                    xyz = self.xyz[-1]

                if self.track_mask is not None:
                    red_mask = np.zeros_like(l_img)
                    red_mask[:,:] = (0,0,255)
                    self.mask_mut.acquire()
                    red_mask[~self.track_mask] = l_img[~self.track_mask]
                    l_img = cv2.addWeighted(l_img,0.7,red_mask,0.3,0)
                    self.mask_mut.release()


                l_img = cv2.resize(l_img,(640,480))
                self.proc_img_idx += 1
                # if self.proc_img_idx % 20 == 0:
                #     cv2.imwrite(f"runs/proc_l_img_{self.proc_img_idx}.jpg",l_img)
                #start = time.time()
                #print(f"detection_time:{time.time()-start}")  
            except Exception as e:
                print(f"{e} occurred in function show_left_rtsp_img.")
if __name__ == "__main__":

    with open('configs/config.yaml') as f:
        config = yaml.safe_load(f)
    config = edict(config)
    obj_locate = TargetLocationModule(config)
    obj_locate.start()

