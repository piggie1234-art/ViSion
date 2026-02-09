import numpy as np
import cv2 as cv
import glob
import os
import json
from marker_detector import MarkerDetector
class CameraCalibrator():

    def __init__(self,config=None) -> None:
        if config is not None:
            self.board_size = config.board_size
        
        else:
            self.board_size = [6,6]
        
        self.intrinsic_param = {}
        self.img_pth = "configs\\camera_intrinsics\\calib_img"
        self.mk_detect = MarkerDetector()
        
    
    def calib_camera(self):

        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.board_size[1]*self.board_size[0],3), np.float32)
        objp[:,:2] = np.mgrid[0:self.board_size[0],0:self.board_size[1]].reshape(-1,2) *7.5
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        images = glob.glob(os.path.join(self.img_pth,'*.bmp'))
        i=0
        for fname in images:
            i+=1
            img = cv.imread(fname)
            h,w,_ = img.shape
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            gray_resized = cv.resize(gray,(int(w/4),int(h/4)))

            #gray = cv.equalizeHist(gray)
            # cv.imshow("12",gray_resized)
            # cv.waitKey(0)
            
                        # Find the chess board corners
            ret, corners = cv.findCirclesGrid(gray_resized,self.board_size,flags=cv.CALIB_CB_SYMMETRIC_GRID|cv.CALIB_CB_CLUSTERING)
            corners = corners*4
            # cv.drawChessboardCorners(img,self.board_size,corners,ret)
            # cv.imwrite(f'{i}.jpg',img)
            #ret, corners = cv.findChessboardCorners(gray, self.board_size, None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                #corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners.copy())
        
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        self.intrinsic_param["K"] = mtx.tolist()
        self.intrinsic_param["dist_coff"] = dist.tolist()
        print("done.")
    
    def save_intrinsic(self):
        file_name = "configs/camera_intrinsics/camera.json"
        with open(file_name, 'w') as s:
            json.dump(self.intrinsic_param,s)
    
    def calibed_img(self,img):
        if "K" not in self.intrinsic_param.keys():
            Warning("Please calibrate the camera first.")
            return img
        mtx = np.array(self.intrinsic_param["K"])
        dist = np.array(self.intrinsic_param["dist_coff"])
        h,  w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        return dst



        



        
if __name__ == '__main__':
    calib = CameraCalibrator()
    calib.calib_camera()
    calib.save_intrinsic()

    images = glob.glob(os.path.join("configs\\camera_intrinsics\\calib_img",'*.bmp'))
    # for fname in images:
    #     img = cv.imread(fname)
    #     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #     out = calib.calibed_img(gray)
    #     cv.imshow("out",out)
    #     cv.waitKey(0)
        #gray = cv.equalizeHist(gray)












#         # Draw and display the corners
#         cv.drawChessboardCorners(img, (7,6), corners2, ret)
#         cv.imshow('img', img)
#         cv.waitKey(500)
# cv.destroyAllWindows()