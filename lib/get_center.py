import cv2
import numpy as np
from depth_estimate_utils import *
import matplotlib.pyplot as plt

img = cv2.imread('1.jpg')
mask = ~img[:,:,0].astype(np.bool)

h,w = mask.shape
y,x = np.where(mask)

centroid = np.mean(x), np.mean(y)
cov = np.cov(x - centroid[0], y - centroid[1])
eigvals, eigvecs = np.linalg.eig(cov)
major_axis = eigvecs[:, np.argmax(eigvals)]
minor_axis = eigvecs[:, np.argmin(eigvals)]

#cv2.line(img, (int(centroid[0]), int(centroid[1])), (int(centroid[0] + 100 * major_axis[0]), int(centroid[1] + 100 * major_axis[1])), (0, 0, 255), 2)
cv2.line(img, (int(centroid[0]), int(centroid[1])), (int(centroid[0] + 100 * minor_axis[0]), int(centroid[1] + 100 * minor_axis[1])), (0, 255, 0), 2)
sample_points= []
for i in range(10):
    cv2.circle(img, (int(centroid[0] + 20*i * major_axis[0]), int(centroid[1] + 20*i * major_axis[1])), 1, (0, 0, 255), 1)
    sample_points.append((int(centroid[0] + 20*i * major_axis[0]), int(centroid[1] + 20*i * major_axis[1])))
    sample_points.append((int(centroid[0] - 20*i * major_axis[0]), int(centroid[1] - 20*i * major_axis[1])))

bin_img = (mask.astype(np.uint8) *255)
#resize_bin_img = cv2.resize(bin_img,(int(w/4),int(h/4)))
contours, _ = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
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

intersections = find_intersectionsv2(sample_points,minor_axis,contour_points,1000)
### 计算intersections的中点
inter_mid = []
for i in intersections:
    inter_mid.append((int((i[0][0]+i[1][0])/2)),(int((i[0][1]+i[1][1])/2)))
    cv2.line(img, (int(list(i)[0][0]), int(list(i)[0][1])), (int(list(i)[1][0]), int(list(i)[1][1])), (0, 0, 255), 2)
        #diam_list.append(np.linalg.norm(np.array(list(i)[0]-np.array(list(i)[1]))))

cv2.imshow('img',img)
cv2.waitKey(0)
xmax = np.max(x)
xmin = np.min(x)

ymax = np.max(y)
ymin = np.min(y)


#start = time.time()


midpoints = get_mid_points(contour_points)

# for pt in midpoints:
#     cv2.circle(img, (pt[0], pt[1]), 1, (0, 0, 255), 1)
# cv2.imshow('img',img)
# cv2.waitKey(0)
center_x = int(len(midpoints) / 2)
norms = calc_norm(np.array(midpoints))
intersections = find_intersections(midpoints[center_x - 100 : center_x + 100:10 ],norms[center_x - 100 : center_x + 100:10 ],contour_points,300)
diam_list = []
for i in intersections:
    if len(i) != 2:
        continue
    else:
        diam_list.append(np.linalg.norm(np.array(list(i)[0]-np.array(list(i)[1]))))
diam_avg = np.mean(np.array(diam_list)) * 4

out_u = int(np.mean(x))
out_v = int(np.mean(y))

# Z = self.K[1,1] * self.channel_diam/diam_avg    
# norm_coord = self.K_inv @ np.array([out_u,out_v,1])
# xyz = Z * norm_coord
#return xyz,(int(out_u),int(out_v)),intersections,midpoints
import numpy as np
import matplotlib.pyplot as plt

# 计算长短轴的函数，与之前的代码段相同
def compute_axes(mask):
   
    return major_axis, minor_axis, centroid

mask = np.zeros((100, 100))
mask[30:70, 40:60] = 1  # 一个40x20的矩形
major_axis, minor_axis, centroid = compute_axes(mask)

fig, ax = plt.subplots()
ax.imshow(mask, cmap='gray')  # 显示mask

# 画出长短轴
scale = 50  # 根据实际情况调整
ax.plot([centroid[0] - scale * major_axis[0], centroid[0] + scale * major_axis[0]], 
        [centroid[1] - scale * major_axis[1], centroid[1] + scale * major_axis[1]], 'r-')
ax.plot([centroid[0] - scale * minor_axis[0], centroid[0] + scale * minor_axis[0]], 
        [centroid[1] - scale * minor_axis[1], centroid[1] + scale * minor_axis[1]], 'b-')

plt.show()
