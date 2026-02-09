# 利用区域的圆度来进行检测
import cv2
import numpy as np
import glob
import os


class MarkerDetector():

    def __init__(self,config=None) -> None:
        self.blob_param = cv2.SimpleBlobDetector_Params()
        self.init_blob_param(config)
        self.detector = cv2.SimpleBlobDetector_create(self.blob_param)


    def detect(self,img):
        
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bin_img = cv2.threshold(gray_img, 70, 255, cv2.THRESH_BINARY_INV)[1]
        # cv2.imshow("sss",bin_img)
        # cv2.waitKey(0)
        try:
            keypoints = self.detector.detect(bin_img)
        
        except:
            return None

        return keypoints

        
    def init_blob_param(self,config):

        if config is not None:
            self.blob_param.blobColor = config.blob_param.blobColor
            self.blob_param.minThreshold = config.blob_param.minThreshold
        else:
            self.blob_param.blobColor = 255
         
            #二值化的起始阈值
            self.blob_param.minThreshold = 0
            #二值化的终止阈值
            self.blob_param.maxThreshold = 255

            #控制blob的区域面积大小
            self.blob_param.filterByArea = True
            self.blob_param.minArea = 100
            self.blob_param.maxArea = 300000
            #blob的圆度限制，默认为不限制，通常不限制，除非找圆形特征
            self.blob_param.filterByCircularity = True
            #blob最小的圆度
            self.blob_param.minCircularity = 0.7
            #blob的凸性
            self.blob_param.filterByConvexity = True
            self.blob_param.minConvexity = 0.7

            #blob的惯性率， 圆为1， 线为0， 大多数情况介于[0 ,1]之间
            self.blob_param.filterByInertia = True
            self.blob_param.minInertiaRatio = 0.5

            #最小的斑点距离，不同的二值图像斑点小于该值时将被认为是同一个斑点
            self.blob_param.minDistBetweenBlobs = 5
            self.blob_param.minRepeatability= 2
    

if __name__ == '__main__':
    md = MarkerDetector()
    images = glob.glob(os.path.join("configs\\camera_intrinsics\\calib_img",'*.bmp'))
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #gray = cv.equalizeHist(gray)
    #img = cv2.imread("test_blob.jpg")
    
        kps = md.detect(img)
        # out_im = cv2.drawKeypoints(img,kps,np.array([]),(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # print(len(kps))
        # cv2.imshow('out_im', out_im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    