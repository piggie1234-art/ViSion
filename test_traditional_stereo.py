import cv2
import numpy as np
import glob
import sys
sys.path.append("../Fast-ACVNet")

from depth_model import DepthEstimateModel
img_paths = glob.glob("runs/depth_imgs/*.jpg")
# 加载左图和右图
dp = DepthEstimateModel()
for i,img_pth in enumerate(img_paths):
    img = cv2.imread(img_pth)
    l_img = img[:,:640,:]
    r_img = img[:,1280:,:]
    # resz_limg = cv2.resize(l_img,(320,240))
    # resz_rimg = cv2.resize(r_img,(320,240))
    disp_im,disp_np = dp.estimate_depth(l_img,r_img)
    l_img = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)
    r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)
    


    # 创建stereoBM对象
    stereo = cv2.StereoBM_create(numDisparities=320, blockSize=9)

    # 计算视差图
    disparity = stereo.compute(l_img, r_img)/16.0
    print(np.max(disparity))
    output = np.concatenate([l_img, r_img, disparity], axis=1)
    cv2.imwrite(f'ACV_disp_{i}.jpg', disp_im)
    cv2.imwrite(f'stereoBM_{i}.jpg', output)

    # # 显示视差图
    # cv2.imshow('disparity', disparity)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()