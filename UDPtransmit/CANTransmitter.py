#-*- coding: utf-8 -*-
from ctypes import cdll, c_ushort, c_char, c_ubyte, c_uint
from _ctypes import Structure
from _ctypes import *
import time
import _thread
import argparse
import struct
import numpy as np

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
    
def check_control_state(motor_pos,cmd_pos):
    motor_pos = np.array(motor_pos)
    cmd_pos = np.array(cmd_pos)
    error = np.linalg.norm(motor_pos-cmd_pos)
    # print('curent error is: ',error)
    if error < 5:
        return True
    else:
        return False

class CANTransmitter():
    libUSBCAN = None
    count = 0   #数据列表中，用来存储列表序号。
    runFlag = True
    #接口卡类型定义
    VCI_USBCAN1 = 3
    VCI_USBCAN_E_U = 20
    
    def __init__(self,bitrate=250000):
        self.libUSBCAN = cdll.LoadLibrary('libusbcan.so')
        self.canInit(bitrate)
        self.hand_info_transmit_buffer = [0,92,0,0,  0.0,0.0,0.0,0.0,0.0,0.0,  0,0,0,0, 0,0,0,0,0]
        self.motor_zero_pos = [0x6000,0xA000,0x9000,0x5000,0x8000]
        self.grasp_motor_pos = [0xC000,0x5000]
        self.position_instay_flag = False
        self.cmd_angle = [0,0,0,0,0]
        self.hand_line_info = [0,0,0,0,0,0]
        self.hand_angle_info = [0,0,0,0,0,0]
        #self.cur_angle = [0,0,0,0,0]
    def send_angle(self,id,angle):
        self.cmd_angle[id-1] = angle
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
            pass
            #print('send success!')
    
    def grasp(self,flag):
          
        ext_id = (6<<4)|0x01
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
            pass
            #print('send success!')
    


    def cut(self,times):
        step_lines = int(0.08/ 360 * 65535)
        for i in range(times):
            
            ext_id = (3<<4)|0x01
            lines = self.hand_line_info[2] + step_lines
            # org_line = int(cur_angle/360*65535)
            # lines = org_line + self.motor_zero_pos[id-1]
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
                pass
            time.sleep(5)

    
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
            pass

     
    def receive_func(self):
        reclen=0
        rec = (PyVCI_CAN_OBJ*300)()   #接收缓存，设为300为佳。
        #recList = rec[1000]
        i=0
        j=0
        ind=0
        while(self.runFlag):
            time.sleep(0.1)    #10ms读取一次
            reclen=self.libUSBCAN.VCI_Receive(self.VCI_USBCAN1,0,ind,byref(rec),100,100)
            if(reclen>0):   #调用接收函数，如果有数据，进行数据处理显示。
                #print("Read %d CANDate\n"%(reclen))
                while(j<reclen):

                    if rec[j].ExternFlag==1 and rec[j].ID & 0x0F:
                        motor_id = (rec[j].ID >> 4) & 0x3F
                        recieved_lines = rec[j].Data[0] << 8 | rec[j].Data[0]
                        self.hand_line_info[motor_id-1] = recieved_lines
                        if motor_id == 6:
                            angle = (recieved_lines - self.grasp_motor_pos[1]) /  65535 * 360
                            self.hand_info_transmit_buffer[9] = angle

                        else:
                            lines =  recieved_lines - self.motor_zero_pos[motor_id-1]
                            angle = lines / 65535 * 360
                            self.hand_angle_info[motor_id-1] = angle
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
        elif bitrate == 500000:
            config.Timing0=0x00
            config.Timing1=0x1C
        elif bitrate == 50000:
            config.Timing0=0x09
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

if __name__ == '__main__':
    ###获取命令含参数args
    parser = argparse.ArgumentParser()
    ###添加参数
    # argparse.add_argument('--can', type=int, default=0, help='can port')
    parser.add_argument('--bitrate', type=int, default=250000, help='baud rate')
    args = parser.parse_args()

    can_tm = CANTransmitter(args.bitrate)
    # for i, angle in enumerate(angle0):
    #     can_tm.send_angle((i+1), angle0)
    #     time.sleep(5)
    for i in range(30):
        positions = np.array([[600, 0, -463],
                              [852, 252, -211],
                              [852, -252, -211],
                              [348, -252, -715],
                              [348, 252, -715]
                              ])  ######把要运动的点写在这里
        # P = np.identity(4)
        # 末端位置
        for p in positions:
            # P[0:3, 3:4] = p[np.newaxis].T
            angles = xyz2angle(p)
            print(f"position:{p},angles:{angles} ")
            can_tm.position_instay_flag = False
            # if angles is None:
            #     print("Invalid position: ", p)
            #     continue
            # else:
            #     angles = np.array(angles)[0,:]
            can_tm.cmd_angle = angles.tolist()
            for i, angle in enumerate(angles):
                can_tm.send_angle((i + 1), angle)
                time.sleep(0.03)
            while True:
                can_tm.position_instay_flag = check_control_state(can_tm.cmd_angle,can_tm.hand_info_transmit_buffer[4:9])
                if can_tm.position_instay_flag:
                    
                    time.sleep(5)   ##每次达到位置后等待2秒
                    can_tm.position_instay_flag = False
                    break
                else:
                    time.sleep(0.05)
            #time.sleep(15)  
            #time.sleep(30)  ##每次发送完所有点后等待10秒   



# if __name__ == '__main__':
#     ###获取命令含参数args
#     parser = argparse.ArgumentParser()
#     ###添加参数
#     #argparse.add_argument('--can', type=int, default=0, help='can port')
#     parser.add_argument('--bitrate', type=int, default=250000, help='baud rate')
#     args = parser.parse_args()
#
#     can_tm = CANTransmiter(args.bitrate)
#
#     while True:
#         #time.sleep(1)
#         user_input = input("Enter ID and Angle separated by a space, or 'q' to quit: ")
#
#         if user_input.lower() == 'q':
#             break
#
#         try:
#             id, angle = map(float, user_input.split())
#             can_tm.send_angle(int(id),angle)
#         except ValueError:
#             print("Invalid input. Please enter two numbers separated by a space.")
#
#
