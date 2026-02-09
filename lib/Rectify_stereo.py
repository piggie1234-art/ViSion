import scipy.io as sio
import cv2 
import numpy as np
import os
# 读取相机内参数
from csrc import remap_cuda
import time
import threading
class RectifyCam():

    def __init__(self,path,img_size=(1440,2560)):
        self.img_size = img_size
        camera_params1 = sio.loadmat(os.path.join(path,'cameraParams1.mat'))
        camera_params2 = sio.loadmat(os.path.join(path,'cameraParams2.mat'))
        stereo_params1 = sio.loadmat(os.path.join(path,'stereoParams1.mat'))
        self.K1 = None
        self.d1 = None
        self.map1x,self.map1y, self.map2x, self.map2y, self.new_k1,self.new_k2= \
            self.init_map(camera_params1,camera_params2,stereo_params1)

        self.remap_mut = threading.Lock()
        
    def init_map(self,cp1,cp2,st1):

        K1 = cp1['K1']
        D1 = cp1['D1']
        p1 = cp1['P1']
       
        K2 = cp2['K2']
        D2 = cp2['D2']
        p2 = cp2['P2']

        R = st1['R']
        T = st1['T']
        d1 = np.zeros((5),dtype=np.float32)
        d2 = np.zeros((5),dtype=np.float32)
        # 将相机矩阵和畸变系数转换为OpenCV的格式
        K1 = np.array(K1, dtype=np.float32)
        D1 = np.array(D1, dtype=np.float32)
        p1 = np.array(p1, dtype=np.float32)
        K2 = np.array(K2, dtype=np.float32)
        D2 = np.array(D2, dtype=np.float32)
        p2 = np.array(p2, dtype=np.float32)
        d1[0] = D1[0,0]
        d1[1] = D1[0,1]
        d2[0] = D2[0,0]
        d2[1] = D2[0,1]
        self.K1 = K1
        self.d1 = d1

        # Load the parameters obtained from calibration
        # cameraMatrix1 = np.array([[1253.0, 0, 1331.0],
        #                         [0, 1253.0, 797.0],
        #                         [0, 0, 1.0]]).astype(np.float32)

        # distCoeffs1 = np.array([0.504078586508289, 0.039, 0, 0, 0]) .astype(np.float32)

        # cameraMatrix2 = np.array([[1252.5, 0, 1339.8],
        #                         [0, 1251.7, 808.23],
        #                         [0, 0, 1.0]]).astype(np.float32)

        # distCoeffs2 = np.array([0.5458, -0.0705, 0, 0, 0]) .astype(np.float32)

        # R = np.array([[0.999653547568999, 0.0166147586849347, -0.0204140791258516],
        # [-0.0165112102667612, 0.999849999472264, 0.00523053446986001],
        # [0.0204979210712195, -0.00489166118562825, 0.999777928783489]])

        # T = np.array([-68.1169404821629, 0.291811243201842, 4.61177225699134]).T

        # # The resolution of your images
        # image_size = (2560, 1440)

        # Compute the rectification transform
        #R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(K1, d1, K2, d2, self.image_size, R.T, T, alpha=0)
        #R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, image_size, R.T, T, alpha=0)


        # # 计算矫正变换矩阵
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            K1, d1, K2, d2, (self.img_size[1], self.img_size[0]), R.T, T.T)

        # # 计算校正映射表
        map1x, map1y = cv2.initUndistortRectifyMap(
            K1, d1, R1, P1, (self.img_size[1], self.img_size[0]), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(
            K2, d2, R2, P2, (self.img_size[1], self.img_size[0]), cv2.CV_32FC1)
        # Compute the maps for remapping
        # map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, self.image_size, cv2.CV_32FC1)
        # map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, self.image_size, cv2.CV_32FC1)

        return map1x, map1y, map2x, map2y,P1[:,:3],P2[:,:3]
    
    def remap(self,img,idx=0):
        # 应用校正映射表
        if idx == 0:
            # img = cv2.undistort(img,self.K1,self.d1)
            # cv2.imwrite("dist.jpg",img)
            rectified_image = cv2.remap(img, self.map1x, self.map1y, cv2.INTER_LINEAR)
            # start = time.time()
            # self.remap_mut.acquire()
            # rectified_image = remap_cuda.remap(img, self.map1x, self.map1y)
            # self.remap_mut.release()
            # print(f"left rectify:{time.time()-start}")

            
        else:

            rectified_image = cv2.remap(img, self.map2x, self.map2y, cv2.INTER_LINEAR)
            # start = time.time()
            # self.remap_mut.acquire()
            # rectified_image = remap_cuda.remap(img, self.map2x, self.map2y)
            # self.remap_mut.release()
            # print(f"right rectify:{time.time()-start}")
        return rectified_image
    


class RectifyCam2():

    def __init__(self,config):
        self.img_size = config.image_shape
    
        self.map1x,self.map1y, self.map2x, self.map2y, self.new_k1,self.new_k2= \
            self.init_map(config)

        self.remap_mut = threading.Lock()
        
    def init_map(self,config):
        K1 = np.array(config.K1)
        K2 = np.array(config.K2)
        d1 = np.array(config.d1)
        d2 = np.array(config.d2)
        R = np.array(config.R)
        T = np.array(config.T)

        # K2 = np.array([[957.671376277092,0,  966.014301366417],
        #                [0, 955.888356381391,580.718539085110],
        #                 [0,0,1] ])
        # d2 = np.array([-0.299752328584700, 0.170632365480533, -0.000297173244957054, -0.000153183442737861, 0])
        # K1 =  np.array([[962.988010051841,0 , 987.715149183363],
        #                [0,  961.430527933541, 542.818020265244],
        #                 [0,0,1] ])
        # d1 = np.array([-0.297032089816089, 0.128557589180085, -0.000240427539121958, -0.000231103002947378, 0])
        # R =np.array([[0.999834134980487, 0.0124004888086004, 0.0133390556310843],
        #              [ -0.0124799109821935, 0.999904790426368, 0.00588743613781763],
        #              [ -0.0132647785393467, -0.00605292984497026, 0.999893698195260]]) 
        # T = np.array([-238.647120468510, 5.00471270778178, 5.20500392846501])
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            K1, d1, K2, d2, (self.img_size[1], self.img_size[0]), R.T, T.T)

        # # 计算校正映射表
        map1x, map1y = cv2.initUndistortRectifyMap(
            K1, d1, R1, P1, (self.img_size[1], self.img_size[0]), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(
            K2, d2, R2, P2, (self.img_size[1], self.img_size[0]), cv2.CV_32FC1)
        # Compute the maps for remapping
        # map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, self.image_size, cv2.CV_32FC1)
        # map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, self.image_size, cv2.CV_32FC1)

        return map1x, map1y, map2x, map2y,P1[:,:3],P2[:,:3]
    
    def remap(self,img,idx=0):
        # 应用校正映射表
        if idx == 0:
            # img = cv2.undistort(img,self.K1,self.d1)
            # cv2.imwrite("dist.jpg",img)
            #rectified_image = cv2.remap(img, self.map1x, self.map1y, cv2.INTER_LINEAR)
            # start = time.time()
            # self.remap_mut.acquire()
            rectified_image = remap_cuda.remap(img, self.map1x, self.map1y)
            # self.remap_mut.release()
            # print(f"left rectify:{time.time()-start}")

            
        else:

            #rectified_image = cv2.remap(img, self.map2x, self.map2y, cv2.INTER_LINEAR)
            # start = time.time()
            # self.remap_mut.acquire()
            rectified_image = remap_cuda.remap(img, self.map2x, self.map2y)
            # self.remap_mut.release()
            # print(f"right rectify:{time.time()-start}")
        return rectified_image
