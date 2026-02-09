import numpy as np
import cv2
import json
import os
import glob
import scipy.io as sio
from lib.depth_estimate_utils import *

class StereoLocationModel():
    def __init__(self,new_K,config) -> None:
        
        self.img = None
        self.left_intrinsic = None
        self.right_intrinsic = None
        self.extrinsic = None
        self.resize_rate = [1920.0 / 640.0,1080.0 / 480.0]
        self.K = new_K
        self.image_shape =config.image_shape
        # camera_params1 = sio.loadmat(os.path.join(stereo_config_path,'cameraParams1.mat'))
        # stereo_params1 = sio.loadmat(os.path.join(stereo_config_path,'stereoParams1.mat'))
        self.base_line = config.base_line
        self.link_diam = 24.7 ##mm  
        self.K_inv = np.linalg.inv(self.K)  
        self.link_len = 208.0
        self.cable_diam = 80
        self.channel_diam = 350
    
    def init_param(self,cp1,st1):

        K1 = cp1['K1']
        d1 = cp1['D1']
        p1 = cp1['P1']

        R = st1['R']
        T = st1['T']
        # d1 = np.zeros((1,5),dtype=np.float64)
        # d2 = np.zeros((1,5),dtype=np.float64)
        # 将相机矩阵和畸变系数转换为OpenCV的格式
        K1 = np.array(K1, dtype=np.float64)
        D1 = np.array(d1, dtype=np.float64)
        p1 = np.array(p1, dtype=np.float64)
        t = np.array(T, dtype=np.float64)
        dist = np.linalg.norm(t)

        return  K1, dist

    def load_stereo_param(self, param_path):
        left_intrinsic_path = os.path.join(param_path, 'left_intrinsic.json')
        right_intrinsic_path = os.path.join(param_path, 'right_intrinsic.json')
        extrinsic_path = os.path.join(param_path, 'extrinsic.json')

        with open(left_intrinsic_path) as f:
            left_intrinsic = json.load(f)
        with open(right_intrinsic_path) as f:
            right_intrinsic = json.load(f)
        with open(extrinsic_path) as f:
            extrinsic = json.load(f)

    def convert_disp2depth(self, disp):
        depth = self.left_intrinsic['fx'] * self.extrinsic['baseline'] / disp
        
        return depth
    
    def calculate_xyz(point,z):
        pass


    def locate_cable(self,mask):
        h,w = mask.shape
        y,x = np.where(mask)
        xmax = np.max(x)
        xmin = np.min(x)
        
        ymax = np.max(y)
        ymin = np.min(y)

        bin_img = (mask.astype(np.uint8) *255)
        resize_bin_img = cv2.resize(bin_img,(int(w/4),int(h/4)))
        contours, _ = cv2.findContours(resize_bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  #检测缩小后的二值图像中的轮廓。
        # 过滤掉那些接近图像边缘的轮廓
        border_margin = 3  # 距离边缘的最小距离
        max_contour = 0
        max_con_idx = 0
        for i,cnt in enumerate(contours):
            length = cnt.shape[0]
            if length > max_contour:
                max_contour = length
                max_con_idx = i

        # 假设只有一个轮廓，获取它的所有点
        contour_points = contours[max_con_idx]
        #start = time.time()
        midpoints = get_mid_points(contour_points)
        center_x = int(len(midpoints) / 2)
        norms = calc_norm(np.array(midpoints))
        intersections = find_intersections(midpoints[center_x - 100 : center_x + 100:10 ],norms[center_x - 100 : center_x + 100:10 ],contour_points,150)
        diam_list = []
        for i in intersections:
            if len(i) != 2:
                continue
            else:
                diam_list.append(np.linalg.norm(np.array(list(i)[0]-np.array(list(i)[1]))))
                list(i)[0] *= 4
                list(i)[1] *= 4
        diam_avg = np.mean(np.array(diam_list)) * 4

        out_u = int(np.mean(x))
        out_v = int(np.mean(y))

        
        ##diam = ((right0 - left0) + (right1 - left1) + (right2 - left2) + (right3 - left3) + (right4 - left4))/5.0

        Z = self.K[1,1] * self.cable_diam/diam_avg    
        norm_coord = self.K_inv @ np.array([out_u,out_v,1])
        xyz = Z * norm_coord


        return xyz,(int(out_u),int(out_v)),intersections
    
    def locate_link(self,mask):
        y,x = np.where(mask)
        xmax = np.max(x)
        xmin = np.min(x)
        
        ymax = np.max(y)
        ymin = np.min(y)

        link_len = ymax -ymin
        out_u = int(np.mean(x))
        out_v = int(np.mean(y))
       
        Z = self.K[0,0] * self.link_len / link_len   
        norm_coord = self.K_inv @ np.array([out_u,out_v,1])
        xyz = Z * norm_coord
        return xyz,(int(out_u),int(out_v)),None

    def locate_channel(self,mask):


        r_scale = 4
        h,w = mask.shape
        bin_img = (mask.astype(np.uint8) *255)
        resize_bin_img = cv2.resize(bin_img,(int(w/r_scale),int(h/r_scale)))
        contours, _ = cv2.findContours(resize_bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        y,x = np.where(resize_bin_img)

        centroid = np.mean(x), np.mean(y)
        cov = np.cov(x - centroid[0], y - centroid[1])
        eigvals, eigvecs = np.linalg.eig(cov)
        major_axis = eigvecs[:, np.argmax(eigvals)]
        minor_axis = eigvecs[:, np.argmin(eigvals)]
        # xmax = np.max(x)
        # xmin = np.min(x)
        
        # ymax = np.max(y)
        # ymin = np.min(y)
        # # 过滤掉那些接近图像边缘的轮廓
        # border_margin = 3  # 距离边缘的最小距离
        max_contour = 0
        max_con_idx = 0
        for i,cnt in enumerate(contours):
            length = cnt.shape[0]
            if length > max_contour:
                max_contour = length
                max_con_idx = i

        # 假设只有一个轮廓，获取它的所有点
        contour_points = contours[max_con_idx]
        #start = time.time()


        midpoints = get_mid_points(contour_points)
        center_x = int(len(midpoints) / 2)
        norms = calc_norm(np.array(midpoints))
         
        intersections = find_intersections(midpoints[center_x - 50 : center_x + 50:10 ],norms[center_x - 50 : center_x + 50:10 ],contour_points,int(800/r_scale))
        #intersections = find_intersectionsv2(midpoints[center_x - 50 : center_x + 50:10],minor_axis,contour_points,int(800/r_scale))
        
        sample_points= []
        for i in range(5):
            #cv2.circle(img, (int(centroid[0] + 20*i * major_axis[0]), int(centroid[1] + 20*i * major_axis[1])), 1, (0, 0, 255), 1)
            sample_points.append((int(centroid[0] + 20*i * minor_axis[0]), int(centroid[1] + 20*i * minor_axis[1])))
            sample_points.append((int(centroid[0] - 20*i * minor_axis[0]), int(centroid[1] - 20*i * minor_axis[1])))
        intersection_y = find_intersectionsv2(sample_points,major_axis,contour_points,int(1600/r_scale))
        cnt_x = 0
        for i in intersection_y:
            cnt_x+= ((i[0][0]+i[1][0])/2)
        avg_x = int(cnt_x/len(intersection_y))
        #intersections = find_intersections(midpoints[center_x - 100 : center_x + 100:10 ],norms[center_x - 100 : center_x + 100:10 ],contour_points,300)
        diam_list = []
        
        for i in intersections:
            if len(i) != 2:
                continue
            else:
                diam_list.append(np.linalg.norm(np.array(list(i)[0]-np.array(list(i)[1]))))
        diam_avg = np.mean(np.array(diam_list)) * r_scale
        #intersections = np.array(intersections) * r_scale
        
        out_u = avg_x * r_scale
        #out_u = int(np.mean(x)) * 4
        out_v = int(np.mean(y)) * r_scale

        Z = self.K[1,1] * self.channel_diam/diam_avg    
        norm_coord = self.K_inv @ np.array([out_u,out_v,1])
        xyz = Z * norm_coord
        x = xyz[0]    ### x/z = x_delta/175
        x_delta = x/Z * 175
        x -= x_delta
        xyz[0] = x
        return xyz,(int(out_u),int(out_v)),intersections



    def locate_mask_prior(self,mask,cls):

        if cls == '1':
            return self.locate_link(mask)
            
        elif cls == '2':
            return self.locate_cable(mask)
            
        elif cls == '3':
            return self.locate_channel(mask)
    

    def filter_outliers(self,data):
        
        # 计算原始数据的均值和标准差
        mean = np.mean(data)
        std = np.std(data)

        # 定义标准差阈值
        threshold = 1

        # 筛选未超过阈值的数据
        filtered_data = [x for x in data if (x >= mean - threshold * std) and (x <= mean + threshold * std)]

        # 计算排除异常值后的新均值
        filtered_mean = np.mean(filtered_data)

        print("原始数据均值:", mean)
        print("排除异常值后的均值:", filtered_mean)
        return filtered_data,filtered_mean
    
    def locate_float_link(self,disp,mask):
        disp = cv2.resize(disp,(1920,1080))
        #mask = cv2.resize(mask,(640,480))
        mask = mask > 0.7
        y,x = np.where(mask)
        if len(x) == 0:
            return [0,0,0],[0,0]
        y = int(np.mean(y))
        x_line = mask[y]
        x = np.where(x_line)
        x = int(np.mean(x))
        u,v = int(x),int(y)
        K_inv = np.linalg.inv(self.K)
        norm_coord = K_inv @ np.array([u,v,1])

        # u /= self.resize_rate[0]
        # v /= self.resize_rate[1]
        #  ###mean disp in the mask region
        disp_valid_mask = (disp > 0)
        union_mask = disp_valid_mask & mask
        disps = disp[union_mask]
        if len(disps) <= 0:
            return None,None
        filtered_data, disp_mean = self.filter_outliers(disps)
        disp_mean*= self.resize_rate[0]
        #print(f"disp:{d}")
        # if d > 200 or d < 10:
        #     return None, (int(out_u),int(out_v))
        Z = self.K[0,0] * self.base_line / disp_mean
        xyz = Z * norm_coord

        return xyz,(int(u),int(v))



    def locate_mask(self,disp,mask):
      
        height,width = mask.shape
        y,x = np.where(mask)
                
        x = int(np.mean(x))

        y = int(np.mean(y))
        #x = np.argmax(mask[y])
        u,v = x,y
        out_u,out_v = u,v 

        ## calculate xyz for disp img at mid point
        K_inv = np.linalg.inv(self.K)
        norm_coord = K_inv @ np.array([u,v,1])

        u /= self.resize_rate[0]
        v /= self.resize_rate[1]
       

        # u = int(u)
        # v = int(v)
        # start_u = u - 2 if u - 2 >= 0 else 0
        # end_u = u + 3 if u +3 < width else width

        
        # start_v = v - 2 if v - 2 >= 0 else 0
        # end_v = v + 3 if v +3 < height else height

        # patch = disp[start_u:end_u,start_v:end_v]
        

        d = disp[int(v),int(u)]
        d*= self.resize_rate[0]
        #print(f"disp:{d}")
        if d > 200 or d < 10:
            return None, (int(out_u),int(out_v))
        Z = self.K[0,0] * self.trans / d
        xyz = Z * norm_coord

        return xyz,(int(out_u),int(out_v))


    def locate(self, left_image, disp, bbox):
        # bbox: [xmin, ymin, xmax, ymax, class_id]
        cropped_img = left_image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((35,35), np.uint8)
        # 应用开运算
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        #thresh = cv2.bitwise_not(thresh)
        # 找出连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

        # 获取所有连通区域的面积,exclude index 0 (the background)
        areas = stats[1:, cv2.CC_STAT_AREA]  # 从索引1开始，以忽略背景区域

        # 找出面积最大的两个连通区域的索引
        if len(areas) <= 2:
            largest_component_indices = [1, 2]
        else:
            largest_component_indices = np.argpartition(-areas, 2)[:2] + 1  # +1 是因为我们忽略了背景区域

        # # 创建一个空的图像，用于存放只包含面积最大的两个连通区域的结果
        # output = np.zeros_like(thresh)

        # for i in largest_component_indices:
        #     output[labels == i] = 255

        # 假设 largest_component_indices 包含了面积最大的两个连通区域的索引
        centroid1 = centroids[largest_component_indices[0]]
        centroid2 = centroids[largest_component_indices[1]]

        # 计算两个重心的连线的中点
        midpoint = (centroid1 + centroid2) / 2 + (bbox[0],bbox[1])
        # 假设 img 是你的原图
        # 将中点的坐标转换为整数
        midpoint = tuple(map(int, midpoint))

        u,v = midpoint
        out_u,out_v = u,v 

        ## calculate xyz for disp img at mid point
        K_inv = np.linalg.inv(self.K)
        norm_coord = K_inv @ np.array([u,v,1])


        u /= self.resize_rate[0]
        v /= self.resize_rate[1]
        d = disp[int(v),int(u)]
        d*= self.resize_rate[0]
        #print(f"disp:{d}")
        if d > 400:
            return None, (int(out_u),int(out_v))
        Z = self.K[0,0] * self.trans / d
        xyz = Z * norm_coord

        return xyz,(int(out_u),int(out_v))





        # # 画出中点，我们使用红色（BGR为(0, 0, 255)）来表示，半径为5，线宽为-1表示填充圆圈
        # img = cv2.circle(cropped_img, midpoint, 5, (0, 0, 255), -1)

        # # 显示图像
        # cv2.imshow('Image with midpoint', cropped_img)
        # cv2.waitKey(0)

        # # 显示图像
        # cv2.imshow('Largest Components', output)
        # cv2.waitKey(0)

       
    

def read_yolo_format(file_path, img_width, img_height):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    bbox_and_classes = []
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())

        # Convert to absolute coordinates
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        # Convert to xyxy format
        xmin = int(np.round(x_center - width / 2))
        ymin = int(np.round(y_center - height / 2))
        xmax = int(np.round(x_center + width / 2))
        ymax = int(np.round(y_center + height / 2))
        bbox_and_classes.append([xmin, ymin, xmax, ymax, class_id]
        )
        
    return bbox_and_classes

def display_image_with_bbox(image_path, bbox_and_classes):
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]
    
    for item in bbox_and_classes:
        x_center = item['x_center'] * img_width
        y_center = item['y_center'] * img_height
        width = item['width'] * img_width
        height = item['height'] * img_height
        
        # Convert to xyxy format
        xmin = int(np.round(x_center - width / 2))
        ymin = int(np.round(y_center - height / 2))
        xmax = int(np.round(x_center + width / 2))
        ymax = int(np.round(y_center + height / 2))

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(img, str(item['class_id']), (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

if __name__ == '__main__':
    stero_model = StereoLocationModel()
    img_path = 'exp4'  # replace with your directory path
    jpg_files = glob.glob(os.path.join(img_path, '*.jpg'))
    ## sort the file names
    jpg_files.sort(key=lambda x: int(x.split('\\')[-1].split('.')[0]))
    sample_img = cv2.imread(jpg_files[0])
    img_height, img_width = sample_img.shape[:2]
    for jpg_file in jpg_files:
        txt_file = os.path.splitext(jpg_file)[0] + '.txt'
        if os.path.exists(txt_file):
            bbox_and_classes = read_yolo_format(txt_file, img_width, img_height)
            img = cv2.imread(jpg_file)

            stero_model.locate(img, bbox_and_classes[0])
            
            


    
    
#     def correct_distortion(self, u, v, z, K, D):
#     # D is the distortion coefficients
#     k1, k2, p1, p2, k3 = D

#     fx = K[0, 0]
#     fy = K[1, 1]
#     cx = K[0, 2]
#     cy = K[1, 2]

#     # Convert pixel coordinates to camera coordinates
#     x = (u - cx) / fx
#     y = (v - cy) / fy

#     r2 = x*x + y*y

#     # Apply distortion
#     x_distorted = x * (1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2) + 2*p1*x*y + p2*(r2 + 2*x*x)
#     y_distorted = y * (1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2) + p1*(r2 + 2*y*y) + 2*p2*x*y

#     # Convert back to pixel coordinates
#     u_distorted = x_distorted * fx + cx
#     v_distorted = y_distorted * fy + cy

#     return u_distorted, v_distorted, z

# # Example usage:
# K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # Replace fx, fy, cx, cy with your values
# D = np.array([k1, k2, p1, p2, k3])  # Replace k1, k2, p1, p2, k3 with your values
# u = 100  # x position of pixel
# v = 200  # y position of pixel
# depth = np.random.rand(480, 640)  # Example depth map, replace with your depth map

# u_distorted, v_distorted, z = correct_distortion(u, v, depth[v, u], K, D)
# print(f'The distorted pixel coordinates are ({u_distorted}, {v_distorted}, {z})')

