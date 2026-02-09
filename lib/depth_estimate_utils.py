import cv2
import numpy as np
import time
from bresenham import bresenham
import random
import scipy
def is_near_border(bounding_rect, img, border_margin=3):
    x, y, w, h = bounding_rect
    return (x <= border_margin and
            y <= border_margin and
            x + w >= img.shape[1] - border_margin and
            y + h >= img.shape[0] - border_margin)
    
def calc_norm(points):
    start = time.time()
    # 计算每个点的切线方向
    tangents = np.diff(points, axis=0)

    # 计算每个点的法线方向
    normals = np.zeros_like(tangents)
    normals[:, 0] = -tangents[:, 1]
    normals[:, 1] = tangents[:, 0]

    # 对法线方向进行单位化
    eps = 1e-8  # 添加一个非常小的值，以避免除以零的错误
    normals = normals / (np.sqrt(np.sum(normals**2, axis=1, keepdims=True)) + eps)
    #print('norm Time: ', time.time() - start)
    return normals

def get_mid_points_y(countour):
     # 按x值排序所有的点
    contour_points_sorted = sorted(countour, key=lambda point: point[0][1])
    midpoints = []
    contour_dict = {}
    for point in contour_points_sorted:
        x = point[0][0]
        y = point[0][1]
        if y not in contour_dict:
            contour_dict[y] = {'min': x, 'max': x, 'only': True}
        else:
            contour_dict[y]['only'] = False
            contour_dict[y]['min'] = min(contour_dict[y]['min'], x)
            contour_dict[y]['max'] = max(contour_dict[y]['max'], x)

    # 对于每一列，计算y的中点
    midpoints = []
    for y, x_dict in contour_dict.items():
        if x_dict['only']:
            continue
        if abs(x_dict['min'] - x_dict['max'])<10: 
            continue
        x_middle = int((x_dict['min'] + x_dict['max']) / 2)
        midpoints.append((x_middle, y))
    midpoints = np.array(midpoints)
    #print('midpoint Time: ', time.time() - start)
    return midpoints




def get_mid_points(countour):
    # 按x值排序所有的点
    contour_points_sorted = sorted(countour, key=lambda point: point[0][0])
    midpoints = []
    contour_dict = {}
    for point in contour_points_sorted:
        x = point[0][0]
        y = point[0][1]
        if x not in contour_dict:
            contour_dict[x] = {'min': y, 'max': y, 'only': True}
        else:
            contour_dict[x]['only'] = False
            contour_dict[x]['min'] = min(contour_dict[x]['min'], y)
            contour_dict[x]['max'] = max(contour_dict[x]['max'], y)

    # 对于每一列，计算y的中点
    midpoints = []
    for x, y_dict in contour_dict.items():
        if y_dict['only']:
            continue
        if abs(y_dict['min'] - y_dict['max'])<10: 
            continue
        y_middle = int((y_dict['min'] + y_dict['max']) / 2)
        midpoints.append((x, y_middle))
    midpoints = np.array(midpoints)
    #print('midpoint Time: ', time.time() - start)
    return midpoints

def find_intersections(mid_points,norm,contour,length=300):
    
    ## 计算每个点的切线方向
    slopes = norm[:, 1] / (norm[:, 0] + 1e-8)
    z_scores = scipy.stats.zscore(slopes)
    thresh = 1.5
    outliers = np.where(np.abs(z_scores)>thresh)
    slopes = np.delete(slopes,outliers,axis=0)
    mid_points = np.delete(mid_points,outliers,axis=0)

    # 计算线段的两个端点
    dxs = np.sqrt(length**2 / (1 + slopes**2))
    dys = slopes * dxs
    end_points1 = mid_points[:,:] + np.column_stack((dxs, dys))
    end_points2 = mid_points[:,:] - np.column_stack((dxs, dys))
    # 将端点坐标四舍五入并转换为整数
    end_points1 = np.round(end_points1).astype(int)
    end_points2 = np.round(end_points2).astype(int)
    # 使用 Bresenham's line algorithm 算法获取每条线段上的所有像素
    pixels = []
    for (x1, y1), (x2, y2) in zip(end_points1, end_points2):
        pixels.append(list(bresenham(x1, y1, x2, y2)))
    intersection_list = []
    # 计算每条线段像素与轮廓的交点
    contour_set = set()
    for point in contour:
        contour_set.add(tuple(point[0]))
    for pixel_line in pixels:

        # 转换为集合
        line_set = set(pixel_line)
        # 计算交集
        intersection = line_set & contour_set
        intersection_list.append(intersection)
    
    return intersection_list

def find_intersectionsv2(mid_points,norm,contour,length=300):
    
    ## 计算每个点的切线方向
    slope = norm[1] / (norm[0] + 1e-8)
    #z_scores = scipy.stats.zscore(slopes)
    #thresh = 1.5
    #outliers = np.where(np.abs(z_scores)>thresh)
    #slopes = np.delete(slopes,outliers,axis=0)
    #mid_points = np.delete(mid_points,outliers,axis=0)

    # 计算线段的两个端点
    dxs = np.sqrt(length**2 / (1 + slope**2))
    dys = slope * dxs
    end_p = np.column_stack((dxs, dys))
    end_points1 = np.array(mid_points) + end_p
    end_points2 = np.array(mid_points) - end_p
    # 将端点坐标四舍五入并转换为整数
    end_points1 = np.round(end_points1).astype(int)
    end_points2 = np.round(end_points2).astype(int)
    # 使用 Bresenham's line algorithm 算法获取每条线段上的所有像素
    pixels = []
    for (x1, y1), (x2, y2) in zip(end_points1, end_points2):
        pixels.append(list(bresenham(x1, y1, x2, y2)))
    intersection_list = []
    # 计算每条线段像素与轮廓的交点
    contour_set = set()
    for point in contour:
        contour_set.add(tuple(point[0]))
    for i,pixel_line in enumerate(pixels):

        # 转换为集合
        line_set = set(pixel_line)
        # 计算交集
        intersection = line_set & contour_set
        ##计算每个点与中心点组成的向量，保存方向相反的两个点
        first_p = None
        v1 = None
        v2 = None
        if len(intersection)==2:
            intersection_list.append([list(intersection)[0],list(intersection)[1]])
        elif len(intersection)>2:
            for point in intersection:
                if first_p is None:
                    first_p = point
                    v1 = np.array(first_p) - np.array(mid_points[i])
                    continue
                v2 = np.array(point) - np.array(mid_points[i])
                if np.dot(v1,v2)<0:
                    intersection_list.append([first_p,point])
                    first_p = None 
                    break         
        else:
            continue
    return intersection_list



def find_intersections_horizon(mid_points,norm,contour,length=300):
    
    ## 计算每个点的切线方向
    slopes = norm[:, 1] / (norm[:, 0] + 1e-8)
    z_scores = scipy.stats.zscore(slopes)
    thresh = 1.5
    outliers = np.where(np.abs(z_scores)>thresh)
    slopes = np.delete(slopes,outliers,axis=0)
    mid_points = np.delete(mid_points,outliers,axis=0)

    # 计算线段的两个端点
    dxs = np.sqrt(length**2 / (1 + slopes**2))
    dys = slopes * dxs
    end_points1 = mid_points[:,:] + np.column_stack((dxs, dys))
    end_points2 = mid_points[:,:] - np.column_stack((dxs, dys))
    # 将端点坐标四舍五入并转换为整数
    end_points1 = np.round(end_points1).astype(int)
    end_points2 = np.round(end_points2).astype(int)
    # 使用 Bresenham's line algorithm 算法获取每条线段上的所有像素
    pixels = []
    for (x1, y1), (x2, y2) in zip(end_points1, end_points2):
        pixels.append(list(bresenham(x1, y1, x2, y2)))
    intersection_list = []
    # 计算每条线段像素与轮廓的交点
    contour_set = set()
    for point in contour:
        contour_set.add(tuple(point[0]))
    for pixel_line in pixels:

        # 转换为集合
        line_set = set(pixel_line)
        # 计算交集
        intersection = line_set & contour_set
        intersection_list.append(intersection)
    
    return intersection_list

# image_bgr = cv2.imread('test.png')
# img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
# mid_points = []
# # 应用阈值操作进行二值化
# _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# # 将图像从BGR转换为HSV

# start = time.time()
# contours, _ = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# # 过滤掉那些接近图像边缘的轮廓
# border_margin = 3  # 距离边缘的最小距离
# contours = [cnt for cnt in contours if not is_near_border(cv2.boundingRect(cnt), image_bgr, border_margin)]
# print('edge Time: ', time.time() - start)
# # 在原图上画出轮廓，轮廓颜色为红色，厚度为2

# # 假设只有一个轮廓，获取它的所有点
# contour_points = contours[0]
# start = time.time()
# midpoints = get_mid_points(contour_points)
# norms = calc_norm(np.array(midpoints))
# intersections = find_intersections(midpoints[:51],norms[:50],contour_points)
# print('All Time: ', time.time() - start)
# for i in intersections:
#     if len(i) != 2:
#         continue
#     else:
#         # use random color
        
#         color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
#         cv2.circle(image_bgr, tuple(list(i)[0]), 5, color, 4)
#         cv2.line(image_bgr, tuple(list(i)[0]), tuple(list(i)[1]), (0, 0, 255), 1)

# # 对于每个点，画一个表示法线方向的小红线
# for i in range(len(midpoints) - 1):
#     pt1 = tuple(midpoints[i])
#     pt2 = tuple((midpoints[i] + norms[i] * 20).astype(int))  # 法线长度为20
#     cv2.line(image_bgr, pt1, pt2, (0, 0, 255), 1)

# # 显示带有轮廓的图像
# cv2.imshow('Contours', image_bgr)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
