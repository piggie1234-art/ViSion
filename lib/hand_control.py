
#-*- coding: utf-8 -*-
from ctypes import cdll, c_ushort, c_char, c_ubyte, c_uint
from _ctypes import Structure
from _ctypes import *
import time
import _thread
import argparse
import struct
import numpy as np
from math import sin, cos,radians
#from dof5_test2_daishu import xyz2angle
    
def check_control_state(motor_pos,cmd_pos):
    motor_pos = np.array(motor_pos)
    cmd_pos = np.array(cmd_pos)
    error = np.linalg.norm(motor_pos-cmd_pos)
    # print('curent error is: ',error)
    if error < 5:
        return True
    else:
        return False
def dh_transformation(alpha, a, theta, d):  # 齐次变换矩阵公式
    alpha = radians(alpha)
    theta = radians(theta)              # 角度转弧度
    matrix = np.mat(np.zeros((4, 4)))
    matrix[0, 0] = cos(theta)
    matrix[0, 1] = -sin(theta)
    matrix[0, 3] = a
    matrix[1, 0] = sin(theta) * cos(alpha)
    matrix[1, 1] = cos(theta) * cos(alpha)
    matrix[1, 2] = -sin(alpha)
    matrix[1, 3] = -sin(alpha) * d
    matrix[2, 0] = sin(theta) * sin(alpha)
    matrix[2, 1] = cos(theta) * sin(alpha)
    matrix[2, 2] = cos(alpha)
    matrix[2, 3] = cos(alpha) * d
    matrix[3, 3] = 1
    return matrix
def DOF5_matrix(DHparameter_matrix):  # 正运动学
    DH_mat = DHparameter_matrix
    DOF5_mat = np.identity(4)
    for i in range(0, 5, 1):
        temp_mat = dh_transformation(DH_mat[i, 0], DH_mat[i, 1], DH_mat[i, 2], DH_mat[i, 3])
        DOF5_mat = DOF5_mat * temp_mat
    return DOF5_mat

def check_control_state(motor_pos,cmd_pos):
    motor_pos = np.array(motor_pos)
    cmd_pos = np.array(cmd_pos)
    error = np.linalg.norm(motor_pos-cmd_pos)
    # print('curent error is: ',error)
    if error < 15:
        return True
    else:
        return False

#1.CAN系列接口卡信息的数据类型。
class PyVCIBOARD_INFO(Structure):
    _fields_ = [
        ("hw_Version", c_ushort),
        ("fw_Version", c_ushort),
        ("dr_Version", c_ushort),
        ("in_Version", c_ushort),
        ("irq_Num", c_ushort),
        ("can_Num", c_ubyte),
        ("str_Serial_Num", c_char*20),
        ("str_hw_Type", c_char*40),
        ("Reserved", c_ushort*4)
        ]
#2.定义CAN信息帧的数据类型。
class PyVCI_CAN_OBJ(Structure):
    _fields_ = [
        ("ID", c_uint),
        ("TimeStamp", c_uint),
        ("TimeFlag", c_ubyte),
        ("SendType", c_ubyte),
        ("RemoteFlag", c_ubyte),
        ("ExternFlag", c_ubyte),
        ("DataLen", c_ubyte),
        ("Data", c_ubyte*8),
        ("Reserved", c_ubyte*3)
        ]
class PyVCI_CAN_STATUS(Structure):
    _fields_ = [
        ("c_uint", c_ubyte),
        ("regMode", c_ubyte),
        ("regStatus", c_ubyte),
        ("regALCapture", c_ubyte),
        ("regECCapture", c_ubyte),
        ("regEWLimit", c_ubyte),
        ("regRECounter", c_ubyte),
        ("regTECounter", c_ubyte),
        ("Reserved", c_uint)
        ]
#4.定义错误信息的数据类型。
class PyVCI_ERR_INFO(Structure):
    _fields_ = [
        ("ErrCode", c_uint),
        ("Passive_ErrData", c_ubyte*3),
        ("ArLost_ErrData", c_ubyte)
        ]
#5.定义初始化CAN的数据类型
class PyVCI_INIT_CONFIG(Structure):
    _fields_ = [
        ("AccCode", c_uint),
        ("AccMask", c_uint),
        ("Reserved", c_uint),
        ("Filter", c_ubyte),
        ("Timing0", c_ubyte),
        ("Timing1", c_ubyte),
        ("Mode", c_ubyte)
        ]
class CANTransmiter():
    libUSBCAN = None
    count = 0   #数据列表中，用来存储列表序号。
    runFlag = True
    #接口卡类型定义
    VCI_USBCAN1 = 3
    VCI_USBCAN_E_U = 20
    
    def __init__(self,bitrate=50000):
        self.libUSBCAN = cdll.LoadLibrary(("lib/libusbcan.so"))
        self.canInit(bitrate)
        self.hand_info_transmit_buffer = [0,72,0,0,  0.0,0.0,0.0,0.0,0.0,0.0,  0,0,0,0]
        self.motor_zero_pos = [0x6000,0xA000,0x9000,0x5000,0x8000]
        self.grasp_motor_pos = [0xC000,0x5000]
    
    def send_angle(self,id,angle):
        ext_id = (id<<4)|0x01
        org_line = int(angle/360*65535)
        lines = org_line + self.motor_zero_pos[id-1]
        angleline = struct.pack('>H',lines)
        data = struct.unpack('BB',angleline)

        send=PyVCI_CAN_OBJ()
        send.ID=ext_id
        send.SendType=0
        send.RemoteFlag=0
        send.ExternFlag=1
        send.DataLen=2
    
        for i in range(send.DataLen):
            send.Data[i]  = data[i]
            
        #while(times>0):
        if(self.libUSBCAN.VCI_Transmit(self.VCI_USBCAN1, 0, 0, byref(send), 1) == 1):
            print('send success!')
    def gripper_cmd(self,flag):
        ext_id = (6<<4)|0x01
        #org_line = int(angle/360*65535)
        lines = self.grasp_motor_pos[flag]
        angleline = struct.pack('>H',lines)
        data = struct.unpack('BB',angleline)

        send=PyVCI_CAN_OBJ()
        send.ID=ext_id
        send.SendType=0
        send.RemoteFlag=0
        send.ExternFlag=1
        send.DataLen=2
    
        for i in range(send.DataLen):
            send.Data[i]  = data[i]
            
        #while(times>0):
        if(self.libUSBCAN.VCI_Transmit(self.VCI_USBCAN1, 0, 0, byref(send), 1) == 1):
            print('gripper cmd send success!')

    
    
    def send_func(self,ext_id,data):
         #需要发送的帧，结构体设置
        send=PyVCI_CAN_OBJ()
        send.ID=ext_id
        send.SendType=0
        send.RemoteFlag=0
        send.ExternFlag=1
        send.DataLen=6
    
        for i in range(send.DataLen):
            send.Data[i]  = data[i]
            
        #while(times>0):
        if(self.libUSBCAN.VCI_Transmit(self.VCI_USBCAN1, 0, 0, byref(send), 1) == 1):
            print('send success!')
                #print(CANMsg)
                #print("Index:%04d  CAN1 TX ID:0x%08X  DLC:0x%02X data:0x",self.count,send.ID,send.DataLen)
                # self.count+=1
                # times-=1
                # send.ID+=1
            # else:
            #     break
     
    def receive_func(self):
        time.sleep(20)
        reclen=0
        rec = (PyVCI_CAN_OBJ*300)()   #接收缓存，设为300为佳。
        #recList = rec[1000]
        i=0
        j=0
        ind=0
        while(self.runFlag):
            time.sleep(5)    #10ms读取一次
            #print("Thread run again!")
            reclen=self.libUSBCAN.VCI_Receive(self.VCI_USBCAN1,0,ind,byref(rec),100,100)
            #print("Thread VCI_Receive read %d CANmsg!"%(reclen))
            if(reclen>0):   #调用接收函数，如果有数据，进行数据处理显示。
                #printf("Read %d CANDate\n"%(reclen))

                while(j<reclen):
                    # print(f"id:{rec[j].ID }")
                    # print(f'ext_flag:{rec[j].ExternFlag}')
                    # for i in range(rec[j].DataLen):
                    #     print(f"data{i}:{rec[j].Data[i] }")
                    #print(f"data{}:{rec[j].Data[0] }")
                    
                    if rec[j].ExternFlag==1 and rec[j].ID & 0x0F:
                        motor_id = (rec[j].ID >> 4) & 0x3F
                        #print(rec[j].Data[0] )
                        #print(rec[j].Data[1])
                        recieved_lines = (rec[j].Data[0] << 8) | rec[j].Data[1]
                        if motor_id == 6:
                            angle = (recieved_lines - self.grasp_motor_pos[1]) /  65535 * 360
                            #print(f"id:{motor_id},angle:{angle}")
                
                            self.hand_info_transmit_buffer[9] = angle
                    #self.grasper_q.put(angle)
                        else:
                            lines =  recieved_lines - self.motor_zero_pos[motor_id-1]
                            angle = lines / 65535 * 360
                            #print(f"id:{motor_id},angle:{angle}")
                            self.hand_info_transmit_buffer[motor_id + 3] = angle
                    j=j+1
                j=0
        print("Receive thread exit\n") #退出接收线程

    def canInit(self,bitrate=250000):
        #print(">>this is hello !")   #指示程序已运行
        openDevice = self.libUSBCAN.VCI_OpenDevice
        openDevice.argtype = [c_uint, c_uint, c_uint]
        openDevice.restype = c_uint
        
        if(openDevice(self.VCI_USBCAN1,0,0)==1):#打开设备
            print(">>open deivce success!")#打开设备成功
        else:
            print(">>open deivce error!")
            return 
        
        Py_BOARD_INFO = PyVCIBOARD_INFO()
        readBoardInfo = self.libUSBCAN.VCI_ReadBoardInfo
        readBoardInfo.argtype = [c_uint, c_uint, POINTER(PyVCIBOARD_INFO)]
        readBoardInfo.restype = c_uint
        
        if(readBoardInfo(self.VCI_USBCAN1,0,byref(Py_BOARD_INFO))==1):   #读取设备序列号、版本等信息。
            print(">>Get VCI_ReadBoardInfo success!")
            # print(Py_BOARD_INFO.hw_Version)
            # print(Py_BOARD_INFO.fw_Version)
            # print(Py_BOARD_INFO.dr_Version)
            # print(Py_BOARD_INFO.in_Version)
        else:
            print(">>Get VCI_ReadBoardInfo error!")
            return
        
        #初始化参数，严格参数二次开发函数库说明书。
        initCAN = self.libUSBCAN.VCI_InitCAN
        initCAN.argtype = [c_uint, c_uint, c_uint,POINTER(PyVCI_INIT_CONFIG)]
        initCAN.restype = c_uint
        
        config = PyVCI_INIT_CONFIG()
        config.AccCode=0x00000000
        config.AccMask=0x00000000
        config.Filter=8             #允许所有类型的数据
        if bitrate == 100000:
            config.Timing0=0x04     
            config.Timing1=0x1C
        elif bitrate == 50000:
            config.Timing0 = 0x09
            config.Timing1 = 0x1C
        elif bitrate == 500000:
            config.Timing0=0x00
            config.Timing1=0x1C
        elif bitrate == 1000000:
            config.Timing0=0x00
            config.Timing1=0x14
        elif bitrate == 250000:
            config.Timing0=0x01
            config.Timing1=0x1C
        else:
            print(">>bitrate error!")
            return
        config.Mode=0               #正常模式

        if(initCAN(self.VCI_USBCAN1,0,0,byref(config))!=1):
            print(">>Init CAN1 error\n")
            self.libUSBCAN.VCI_CloseDevice(self.VCI_USBCAN1,0)
            return 

        else:
            print(">>VCI_InitCAN success!\n")


        if(self.libUSBCAN.VCI_StartCAN(self.VCI_USBCAN1,0,0)!=1):
            print(">>Start CAN1 error\n")
            self.libUSBCAN.VCI_CloseDevice(self.VCI_USBCAN1,0)
            return
        else:
            print(">>VCI_StartCAN success!\n")
        #创建接收线程
        try:
            _thread.start_new_thread(self.receive_func, ())
        except:
            print("Error:创建接收线程失败")

     
        #self.libUSBCAN.VCI_CloseDevice(self.VCI_USBCAN1,0)
            
    

class HandControl():
   
    def __init__(self,bitrate=50000):
        self.usb_can = self.init_CAN(bitrate)
        self.R = np.array([[0,-1,0],
        [-0.707,0,-0.707],
        [0.707,0,-0.707]]).T
        self.hand_eye_mat = np.eye(4)
        self.trans = np.array([220,-25,220])
        self.hand_eye_mat[:3,:3] = self.R
        self.hand_eye_mat[:3,3] = self.trans
        self.grasp_R = np.array([[0,0,1],[1,0,0],[0,1,0]])
        #print(self.hand_eye_mat)

    def transform_eye2hand(self,position):
        p_hand = self.hand_eye_mat @ np.array([[position[0]],[position[1]],[position[2]],[1]])

        return p_hand[:3].T
    

    def move_prepare_pos(self,position):

        pose = np.eye(4)
        
        pose[:3,:3] = self.grasp_R
        pose[:3,3] = position
        angles = self.xyz2angle(pose)
        #angles = np.rad2deg(angles) 
        #angles = hand.xyz2angle(pos)
        final_angle = self.find_right_angle(angles,pose)
        self.usb_can.position_instay_flag = False
        if final_angle is None:
            print("Invalid position: ", pose)

        else:
            time.sleep(0.5)
            angles = np.array(final_angle)[0,:]
            self.usb_can.cmd_angle = angles.tolist()
            for i, angle in enumerate(angles):
                self.usb_can.send_angle((i + 1), angle)
                time.sleep(0.1)
            self.usb_can.cmd_angle = angles.tolist()
            for i, angle in enumerate(angles):
                self.usb_can.send_angle((i + 1), angle)
                time.sleep(0.1)
            
            while True:
                self.usb_can.position_instay_flag = check_control_state(self.usb_can.cmd_angle,self.usb_can.hand_info_transmit_buffer[4:9])
                if self.usb_can.position_instay_flag:
                    print("prepare position get.")
                    
                    time.sleep(3)   ##每次达到位置后等待2秒
                    self.usb_can.position_instay_flag = False
                    break
                else:
                    time.sleep(0.1)


    def grasp(self,position):
        pose = np.eye(4)
        
        pose[:3,:3] = self.grasp_R
        pose[:3,3] = position
        angles = self.xyz2angle(pose)
        #angles = np.rad2deg(angles) 
        #angles = hand.xyz2angle(pos)
        final_angle = self.find_right_angle(angles,pose)
        self.usb_can.position_instay_flag = False
        if final_angle is None:
            print("Invalid position: ", pose)

        else:
            time.sleep(0.5)
            angles = np.array(final_angle)[0,:]
            self.usb_can.cmd_angle = angles.tolist()
            for i, angle in enumerate(angles):
                self.usb_can.send_angle((i + 1), angle)
                time.sleep(0.1)
            
            while True:
                self.usb_can.position_instay_flag = check_control_state(self.usb_can.cmd_angle,self.usb_can.hand_info_transmit_buffer[4:9])
                if self.usb_can.position_instay_flag:
                    print("position get.")
                    
                    time.sleep(3)   ##每次达到位置后等待2秒
                    self.usb_can.position_instay_flag = False
                    break
                else:
                    time.sleep(0.1)
        
        #time.sleep(8)
        self.usb_can.gripper_cmd(1)
            # while True:
            #     self.usb_can.position_instay_flag = check_control_state(self.usb_can.cmd_angle,self.usb_can.hand_info_transmit_buffer[4:9])
            #     if self.usb_can.position_instay_flag:
                    
            #         time.sleep(3)   ##每次达到位置后等待2秒
            #         self.usb_can.position_instay_flag = False
            #         break
            #     else:
            #         time.sleep(0.05)


        ##self.grasp_R = np.array([[0,0,1],[0,1,0],[1,0,0]])
    def xyz2angle(self,oT):
        r11 = oT[0, 0]
        r12 = oT[0, 1]
        r13 = oT[0, 2]
        px = oT[0, 3]
        r21 = oT[1, 0]
        r22 = oT[1, 1]
        r23 = oT[1, 2]
        py = oT[1, 3]
        r31 = oT[2, 0]
        r32 = oT[2, 1]
        r33 = oT[2, 2]
        pz = oT[2, 3]

        # 求theta1
        theta1 = np.arctan2(py, px)
        s1 = sin(theta1)
        c1 = cos(theta1)

        # 求theta5
        theta5 = np.arctan2(-s1*r11+c1*r21, -s1*r12+c1*r22)
        # theta5 = theta1

        # 求theta234
        theta234 = np.arctan2(c1*r13+s1*r23, r33)
        s234 = sin(theta234)
        c234 = cos(theta234)

        # 求theta2
        d5 = 303
        a3 = 600
        a4 = 320
        A = c1*px+s1*py-s234*d5
        B = c234*d5-pz
        k = (np.square(A)+np.square(B)+np.square(a3)-np.square(a4))/(2*a3)
        theta2_1 = np.arctan2(k, np.sqrt(np.square(A)+np.square(B)-np.square(k)))-np.arctan2(A, B)
        theta2_2 = np.arctan2(k, -np.sqrt(np.square(A)+np.square(B)-np.square(k)))-np.arctan2(A, B)
        s2_1 = sin(theta2_1)
        c2_1 = cos(theta2_1)
        s2_2 = sin(theta2_2)
        c2_2 = cos(theta2_2)

        # 求theta3
        s3_1 = (B*c2_1-A*s2_1)/a4
        s3_2 = (B*c2_2-A*s2_2)/a4
        c3_1_1 = np.sqrt(1-np.square(s3_1))
        c3_1_2 = -np.sqrt(1-np.square(s3_1))
        c3_2_1 = np.sqrt(1-np.square(s3_2))
        c3_2_2 = -np.sqrt(1-np.square(s3_2))
        theta3_1_1 = np.arctan2(s3_1, c3_1_1)
        theta3_1_2 = np.arctan2(s3_1, c3_1_2)
        theta3_2_1 = np.arctan2(s3_2, c3_2_1)
        theta3_2_2 = np.arctan2(s3_2, c3_2_2)

        # 求theta4
        theta4_1_1 = theta234-theta2_1-theta3_1_1
        theta4_1_2 = theta234-theta2_1-theta3_1_2
        theta4_2_1 = theta234-theta2_2-theta3_2_1
        theta4_2_2 = theta234-theta2_2-theta3_2_2

        theta_ikine = np.mat([[theta1, theta2_1, theta3_1_1, theta4_1_1, theta5],
                            [theta1, theta2_1, theta3_1_2, theta4_1_2, theta5],
                            [theta1, theta2_2, theta3_2_1, theta4_2_1, theta5],
                            [theta1, theta2_2, theta3_2_2, theta4_2_2, theta5], ])
        return theta_ikine
    
    def move_back(self):
        self.usb_can.gripper_cmd(0)
        time.sleep(8)
        angles = [0,90,-180,180,90]
        for i, angle in enumerate(angles):
            self.usb_can.send_angle((i + 1), angle)
            time.sleep(0.1)
        


    def find_right_angle(self,angles,P0):
        theta_ikine = np.rad2deg(angles)  # 弧度转角度
        #print("逆解为", np.round(theta_ikine, 2))  # 输出四组解

        for i in range(0, 4):  # 判断各关节角度是否在限制范围内
            if (-180 <= theta_ikine[i, 0] <= 112.5 and 0 <= theta_ikine[i, 1] <= 180 and -180 <= theta_ikine[i, 2] <= 1
                and 0 <= theta_ikine[i, 3] <= 225 and -157.5 <= theta_ikine[i, 4] <= 157.5):

            # if (-180 <= theta_ikine[i, 0] <= 112.5 and 0 <= theta_ikine[i, 1] <= 180 and 0 <= theta_ikine[i, 2] <= 90
            #         and 0 <= theta_ikine[i, 3] <= 225 and -157.5 <= theta_ikine[i, 4] <= 157.5):
                #print("角度限制范围内的解为", np.round(theta_ikine[i, :], 2))
                DH_alpha = np.array([0, -90, 0, 0, 90])
                DH_a = np.array([0, 0, 600, 320, 0])
                Initial_theta = np.array([0, 0, 0, 0, 0])  # theta角的初始值
                DH_theta = np.array([theta_ikine[i, 0], theta_ikine[i, 1],
                                    theta_ikine[i, 2], theta_ikine[i, 3],
                                    theta_ikine[i, 4]])
                DH_d = np.array([0, 0, 0, 0, 303])
                # 把以上DH参数变为一个5x4的矩阵
                DHparameter_matrix = np.mat([[DH_alpha[0], DH_a[0], Initial_theta[0] + DH_theta[0], DH_d[0]],
                                        [DH_alpha[1], DH_a[1], Initial_theta[1] + DH_theta[1], DH_d[1]],
                                        [DH_alpha[2], DH_a[2], Initial_theta[2] + DH_theta[2], DH_d[2]],
                                        [DH_alpha[3], DH_a[3], Initial_theta[3] + DH_theta[3], DH_d[3]],
                                        [DH_alpha[4], DH_a[4], Initial_theta[4] + DH_theta[4], DH_d[4]], ])
                TT = DOF5_matrix(DHparameter_matrix)
                if np.abs(TT[0,3]-P0[0,3])<1 and np.abs(TT[1,3]-P0[1,3])<1 and np.abs(TT[2,3]-P0[2,3])<1:
                    #print("解为", np.round(theta_ikine[i, :], 2))  # 输出符合条件的解
                    return np.round(theta_ikine[i, :], 2)
                continue

    def init_CAN(self,bitrate):
        return CANTransmiter(bitrate)
       



if __name__ == '__main__':
    hand = HandControl(100000)
    pos = hand.transform_eye2hand([0,0,750])
    hand.grasp(pos)
   
        
