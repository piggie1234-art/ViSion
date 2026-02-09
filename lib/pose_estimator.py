import numpy as np
import cv2 as cv
import glob
from lib.marker_detector import MarkerDetector
import json
import math
import time
class PoseEstimator():
    def __init__(self,config) -> None:
        self.K = np.array(config.K)
        self.dist_coeffs = np.array(config.dist_coff)
        self.md = MarkerDetector()
        self.marker_3d_position = np.array([(0,0,0),(15,0,0),(-15,0,0),(0,15,0),(0,-15,0)]).astype(np.float64)
        self.marker_2d = []
        self.output_img = None

    def estimate_pose(self, img,show_img, zero_pts):
        success = False
        rotation_vector = 0
        translation_vector = 0
      
        #gray = cv.cvtColor(img_cp, cv.COLOR_BGR2GRAY)
        kps = self.md.detect(img)

        # out= cv.drawKeypoints(img,kps,np.array([]),(0,0,255),cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # out = cv.resize(out,(out.shape[1],out.shape[0]))
        # cv.imshow('111',out)
        # cv.waitKey(0)
        self.check_keypoints(kps)

    

        if len(self.marker_2d) == 5:
            img_pts = np.array(self.marker_2d)
            # for i,pt in enumerate(img_pts):
            #     cv.putText(img_cp,f'{i}',(int(pt[0]),int(pt[1])),cv.FONT_HERSHEY_PLAIN,10,(0,0,255),1,8)
            # cv.imshow("11",img_cp)
            # cv.waitKey(0)
            rot_vec = np.array(self.marker_2d[1]) - np.array(self.marker_2d[0])
            if show_img:
                for pt in img_pts:
                    cv.circle(img,(int(pt[0]),int(pt[1])),10,(0,0,255),-1)
                for pt in zero_pts:
                    cv.circle(img,(int(pt[0]),int(pt[1])),10,(255,0,0),-1)
                
                self.output_img = img.copy()
                # cv.imshow("111",self.output_img)
                # cv.waitKey(0)
                
        
            (success, rotation_vector, translation_vector) = \
            cv.solvePnP(self.marker_3d_position, 
            img_pts, 
            self.K, 
            self.dist_coeffs,   
            flags=cv.SOLVEPNP_ITERATIVE)

            #print(rotation_vector)
            #print(translation_vector)
            
        return success, translation_vector, rot_vec, img_pts
        
        # display_mut.acquire()
        # self.output_img = img_cp
        # display_mut.release()

    def check_keypoints(self,key_points):

        max_d = 0
        min_d = 1000
        max_idx = 0
        min_idx = 0
        y_positive = 0
        y_negative = 0
        

        if len(key_points)!=5:
            Warning(f"Number of Keypoint is {len(key_points)}, not 5.")
            return
        
        self.marker_2d = []

        for i, keypoint in enumerate(key_points):
            if keypoint.size > max_d:
                max_d = keypoint.size
                max_idx = i
            if keypoint.size < min_d:
                min_d = keypoint.size
                min_idx = i
        #sort the keypoint index according to point positions
        center = key_points[max_idx].pt
        x_positive = key_points[min_idx].pt
        self.marker_2d.append(center)
        self.marker_2d.append(x_positive)

        ###k=(y2-y1)/(x2-x1)
        k = (center[0]-x_positive[0])/(center[1]-x_positive[1]+1e-6)
        ###b=-k*x1+y1
        b = -k*x_positive[1]+ x_positive[0]

        ##d=(k*x0+b-y0)/sqrt(1+k2)
        sqr = math.sqrt(1+k*k)
        x_positive_vec = (x_positive[1]-center[1],x_positive[0]-center[0])
        for i, keypoint in enumerate(key_points):

            distance = abs(k*keypoint.pt[1]+b-keypoint.pt[0])/sqr
            if i == max_idx or i == min_idx:
                continue
            if distance < 30:
                self.marker_2d.append(keypoint.pt)
                #break
            else:
                cur_vec = (keypoint.pt[1]-center[1],keypoint.pt[0]-center[0])
                ## x1y2 - x2y1 > < 0
                if x_positive_vec[0]*cur_vec[1]-x_positive_vec[1]*cur_vec[0]>0:
                    y_positive = keypoint.pt
                else:
                    y_negative = keypoint.pt
        
        self.marker_2d.append(y_positive)
        self.marker_2d.append(y_negative)

                
        
        ###according to x-positive x-negtive axis find y-positive







            
        

        

        





        