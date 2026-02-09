from typing import Counter
import numpy as np
import time
import cv2.aruco as aruco
import cv2

import math
from math import *

def get_euler_angle(rotation_vector):
    # calculate rotation angles
    theta = cv2.norm(rotation_vector, cv2.NORM_L2)
    #求模，即为绕旋转轴转动的角度
    
    # transformed to quaterniond 转换为四元数
    w = math.cos(theta / 2)
    x = math.sin(theta / 2)*rotation_vector[0][0] / theta
    y = math.sin(theta / 2)*rotation_vector[0][1] / theta
    z = math.sin(theta / 2)*rotation_vector[0][2] / theta
    ysqr = y * y
    # pitch (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + ysqr)
    #print('t0:{}, t1:{}'.format(t0, t1))
    pitch = math.atan2(t0, t1)
    
    # yaw (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)

    if t2 > 1.0:
        t2 = 1.0
    if t2 < -1.0:
        t2 = -1.0
    yaw = math.asin(t2)
    
    # roll (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    roll = math.atan2(t3, t4)
    
    #print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))
    
	# # 单位转换：将弧度转换为度
    # Y = int((pitch/math.pi)*180)
    # X = int((yaw/math.pi)*180)
    # Z = int((roll/math.pi)*180)
    
    return pitch,yaw,roll

def get_rotation_vec(pitch,yaw,roll):

    R_x=np.matrix([[1,0,0],[0,cos(pitch),-sin(pitch)],[0,sin(pitch),cos(pitch)]])
    R_y=np.matrix([[cos(yaw),0,sin(yaw)],[0,1,0],[-sin(yaw),0,cos(yaw)]])
    R_z=np.matrix([[cos(roll),-sin(roll),0],[sin(roll),cos(roll),0],[0,0,1]])
    R=np.dot(np.dot(R_z,R_y),R_x)
    rotation_vector= cv2.Rodrigues(R)

    return rotation_vector[0]



def norm(list_elue):
    for i,j in enumerate(list_elue):
        if j <-170/180*math.pi:
            list_elue[i]+=2*math.pi
font = cv2.FONT_HERSHEY_SIMPLEX #font for displaying text (below) 一种字体
#dist=np.array([0.336982,-1.083812,-0.0031367,-0.0033453,0])   #手机畸变参数（k1,k2,p1,p2,k3）k是径向畸变，p是切向畸变#
#dist=np.array([2*3.45148416e-01,2*7.69573140e-01,2*1.88680071e-03,2*1.46044280e-03,2*1.59465491e+00])
dist=np.array([0.0,0.0,0.0,0.0,0.0])
#1151.64263289 0.0 626.52541657 0.0 1151.60618387 471.895463104 0.0 0.0 1.0
mtx_1280=np.array([[2*567.53720406 ,  0,        2*312.66570357],
 [  0, 2*569.36175922,2*257.1729701 ],
 [  0,0,1]])
# newmtx=np.array([[1.13685101e+03, 0.00000000e+00 ,6.15014077e+02],
#  [0.00000000e+00 ,1.14689291e+03, 4.85316133e+02],
#  [0.00000000e+00 ,0.00000000e+00, 1.00000000e+00]])
#mtx_1280=newmtx
#dist=np.array([-7.68075723e-02  ,6.77727776e-01  ,3.66332423e-03, -2.99067879e-04 ,-3.57166040e+00])


saveid=7
for i in range(0,1) : 
    issaingetest=True
    imgid=i
    path="dataset_all\\newdataset\\"+str(saveid)+"\\rgb\\"+"color"+str(imgid)+".png"
    print(path)
    frame=cv2.imread(path)
    #调整图片大小
    frame=cv2.resize(frame,None,fx=1,fy=1,interpolation=cv2.INTER_CUBIC)

    #灰度话
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #设置预定义的字典
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    #使用默认值初始化检测器参数
    parameters =  aruco.DetectorParameters_create()
    #使用aruco.detectMarkers()函数可以检测到marker，返回ID和标志板的4个角点坐标
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)
    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.08, mtx_1280, dist)#估计每个姿态，并返回rvet和tvet


    #corners:detecmarkers返回的检测到的角点列表
    # markerlength:aruco标记的实际物理尺寸，以m为单位
    # mtx是相机内参矩阵
    # dist是相机畸变矩阵
    #rvec每个元素为每个标记相对于相机的旋转向量
    #tvec每个元素为每个标记相对于相机的平移向量
    #_ 每个标记角点的对应点数组,相对于标准块的中心点
    #更新
    #r=np.zeros()
    t=[]
    r=[]
    idx=[]
    c=[]
    for i in range(rvec.shape[0]):
         if ids[i][0] >16:
             continue
         t.append(tvec[i,:,:])
         r.append(rvec[i,:,:])
         idx.append(ids[i])
         c.append(corners[i])
    t=np.array(t)
    r=np.array(r)
    idx=np.array(idx)
    c=np.array(c)
    rvec=r
    tvec=t
    ids=idx
    corners=c

            # ids=np.delete(ids,i,axis=0)
            # rvec=np.delete(rvec,i,axis=0)
            # tvec=np.delete(tvec,i,axis=0)
            # corners=np.delete(corners,i,axis=0)





    id_elue={}
    print(rvec.shape[0])
    for i in range(rvec.shape[0]):
        #print(rvec[i,:,:])
        elue=[0,0,0]
        # print(tvec[i,:,:])
        print(ids[i])
        pitch,yaw,roll=get_euler_angle(rvec[i,:,:])
        elue[0]=pitch
        elue[1]=yaw
        elue[2]=roll
        id_elue[ids[i][0]]=elue
        #print("旋转矩阵为:")
        #print(cv2.Rodrigues(rvec[i,:,:])[0]) #旋转向量转化为旋转矩阵
        aruco.drawAxis(frame, mtx_1280, dist, rvec[i, :, :], tvec[i, :, :], 0.03)

    #坐标的颜色为,X:红色，Y：绿色，Z：蓝色
    axiid=9
    #aruco.drawAxis(frame, mtx_1280, dist, rvec[axiid, :, :], tvec[axiid, :, :], 0.03)
    aruco.drawDetectedMarkers(frame, corners,ids)
    #计算中心坐标
    #求旋转中心的R
    #print("大约是",rvec.shape[0])
    axienum=rvec.shape[0]
    pitch_all=0
    yaw_all=0
    roll_all=0
    for i,j in id_elue.items():
        print(i)
        norm(j)
        pitch_all+=j[0]
        yaw_all+=j[1]
        roll_all+=j[2]
    pitch_final=pitch_all/axienum
    yaw_final=yaw_all/axienum
    roll_final=roll_all/axienum

    rotation_vector=get_rotation_vec(pitch_final,yaw_final,roll_final)
    print("选择向量",rotation_vector)
    rotation_vector=rotation_vector.T.astype(np.float64)
    #求旋转中心的T
    #建立字典
    t_dict={0:[200,-200],1:[100,-200],2:[0,-200],3:[-100,-200],4:[-200,-200],
            5:[-200,-100],6:[-200,0],7:[-200,100],8:[-200,200],9:[-100,200],
            10:[0,200],11:[100,200],12:[200,200],13:[200,100],14:[200,0],15:[200,-100]}
    relativte_c={0:8,1:9,2:10,3:11,4:12,5:13,6:14,7:15,8:0,9:1,10:2,11:3,12:4,13:5,14:6,15:7}
    #tcopy=np.zeros((15,1,3))
    for i in range(rvec.shape[0]):
        #找到相应id
        #获得此时ids相对中心的xy
        #ids[[8],[13],[2],[11]]
        #8对应
        # print([relativte_c[ids[i][0]]])
        # print(ids)
        # if  [relativte_c[ids[i][0]]]  not in ids:
        #   print("不在里面")
        #   continue  
       
        x,y=t_dict[ids[i][0]]
        cur_t=tvec[i,:,:].T
        cur_r=cv2.Rodrigues(rvec[i,:,:])[0]
        RT=np.hstack((cur_r,cur_t*1000))
        newhang=np.array([[0,0,0,1]])
        newRT=np.vstack((RT,newhang))
        valueRT12=np.array([[1 , 0 , 0 ,x],
                            [ 0 , 1 ,0 ,y],
                            [0, 0, 1,  0],
                            [0,0,  0,  1]])
        #计算中间位姿RT1
        RT1=np.matmul(newRT,valueRT12)
        midt=RT1[:3,3]/1000
        midt=midt.T
        tvec[i,:,:]=midt
       # tcopy[i,:,:]=midt
        #aruco.drawAxis(frame, mtx_1280, dist, cur_r, midt, 0.03)
    t=np.sum(tvec,axis = 0)
    t/=(axienum)
    #print(tcopy)
    if issaingetest:
        rotation_vector=rvec[1,:,:]
    aruco.drawAxis(frame, mtx_1280, dist, rotation_vector, t, 0.03)
    T=t.T
    R=cv2.Rodrigues(rotation_vector)[0]
    RT=np.hstack((R,T*1000))
    savepath="dataset_all\\newdataset\\"+str(saveid)+"\\centerposes\\"+str(imgid)+".txt"
    print(RT)
    #np.savetxt(savepath,RT)
    cv2.imwrite("frame"+str(imgid)+".png",frame)
    newframe=cv2.resize(frame,(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    
    cv2.imshow("frame"+str(imgid),newframe)
    
    if cv2.waitKey(500) & 0xFF == ord(' '):
        cv2.waitKey(0)
    else:
        cv2.waitKey(500)
        cv2.destroyAllWindows()



