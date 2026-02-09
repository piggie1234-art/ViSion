import sys
import cv2
import numpy as np
import threading
from easydict import EasyDict as edict
import time
from UDPtransmit.UDPTransmitters import LocationTransmitUDP
sys.path.append("../yolov7")
sys.path.append("../Fast-ACVNet")
from detect_model import DetectionModel
from depth_model import DepthEstimateModel
from lib.depth_model import DepthEstimateModelTRT
from lib.tracking_model import TrackingModelTRT,TrackingModel,TrackingModelTRTv2
from lib.LocationModel import StereoLocationModel
from lib.Camera import POECamera
from collections import deque
import os
from lib.utils import plot_one_box
import yaml
from glob import glob
from lib.depth_model import StereoBM
from lib.utils import plot_one_box
with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)
config = edict(config)
det_model = DetectionModel("trt_models/float_link1021.pt")
#track_model1 = TrackingModel()
track_model = TrackingModelTRTv2(config.TrackingModel)
locate_model = StereoLocationModel(np.array([[958.66,0,1034],[0,958.66,564.73],[0,0,1]]),config.LocationModel)
st_depth = StereoBM()
img_pths = glob("runs/depth_imgs/*.jpg")
for i,pth in enumerate(img_pths):
    img = cv2.imread(pth)
    l_img_org = img[:,:640,:]
    r_img_org = img[:,1280:,:]
    start = time.time()
    l_img = cv2.resize(l_img_org,(1920,1080))
    r_img = cv2.resize(r_img_org,(1920,1080))
    # limg1 = l_img.copy()
    # limg2 = l_img.copy()
    out = det_model.detect(l_img)
    if len(out) > 0:
        for det in out:
            det = det.cpu().detach().numpy()
            if len(det) == 0:
                det_box = None
                #print("no detections.")
                continue
            max_conf_idx = np.argmax(det[:,4])
            
            x1,y1,x2,y2,conf,cls = det[max_conf_idx]##.cpu().detach().numpy()
           
            xyxy = [x1,y1,x2,y2]
            det_box = [[x1,y1],[x2,y2]]
    #det_box = [[100,100],[200,200]]
            

            disp_im = st_depth.estimate_depth(l_img_org,r_img_org)
            mask = track_model.track(l_img,det_box)
            #print(np.max(np.abs(input_1-input_2)))
            #print(np.max(np.abs(img_e1-img_e2)))

            
            
            mask = np.ascontiguousarray(mask)[0][0]
            #mask = cv2.resize(mask,(640,480))
            if mask is not None:
                xyz,uv = locate_model.locate_float_link(disp_im,mask)
                print("total locate time:",time.time()-start)
                red_mask = np.zeros_like(l_img)
                red_mask[:,:] = (0,0,255)
                mask = mask > 0.7
                red_mask[~mask] = l_img[~mask]
                l_img = cv2.addWeighted(l_img,0.7,red_mask,0.3,0)
                plot_one_box(l_img, xyxy, label='1',score='0.95' ,color=(255,0,0), line_thickness=5)
                if xyz is not None and uv is not None:
                    l_img = cv2.circle(l_img,(int(uv[0]),int(uv[1])),20,(0,0,255),-1)
                    cv2.imwrite(f'test_imgs/locate_{i}_{xyz[0]:.2f}_{xyz[1]:.2f}_{xyz[2]:.2f}.jpg',l_img)

# disp_im,disp_np = dp.estimate_depth(l_img,r_img)
# l_img = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)
# r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)



# # 创建stereoBM对象
# stereo = cv2.StereoBM_create(numDisparities=320, blockSize=9)

# # 计算视差图
# disparity = stereo.compute(l_img, r_img)/16.0
# print(np.max(disparity))
# output = np.concatenate([l_img, r_img, disparity], axis=1)
