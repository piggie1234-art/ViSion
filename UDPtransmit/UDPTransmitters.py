import socket
import struct
import threading
import queue
import time
from UDPtransmit.CANTransmitter import CANTransmitter
from UDPtransmit.logger import InfoLogger
from easydict import EasyDict as edict
from UDPtransmit.utils import *
from UDPtransmit.handposesolver import HandPoseSolver
import argparse
class TransmitUDP():
    def __init__(self,target_addr,logger_name='info',self_addr='192.168.1.55',target_port=2333,self_port=6666):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target_address = (target_addr, target_port)
        self.self_address = (self_addr,self_port)
        try:
            self.sock.bind(self.self_address)
        except Exception as e:
            print(f"can not bind self socket:{e}")
        self.sock.settimeout(0.5)
        self.logger = InfoLogger(logger_name)


####target_addr 192.168.1.33 port 1852
class LocationTransmitUDP(TransmitUDP):
    def __init__(self,target_addr,logger_name="location",self_addr='192.168.1.200',target_port=1852,self_port=23333):
        super().__init__(target_addr,logger_name,self_addr,target_port,self_port)
        self.send_dict = self.init_send_dict()
        self.start_time = time.perf_counter()
        angle = 40/180*np.pi
        ### x = [0,-1,0] y = [-sin(40),0,-cos(40)] z = [cos(40),0,-sin(40)]
        self.hand_eye_mat = np.array([[0,-np.sin(angle),np.cos(angle),560],
                                      [-1,0,0,120],
                                      [0,-np.cos(angle),-np.sin(angle),-20],
                                      [0,0,0,1]])
        self.hand_base_mat = np.array([[1,0,0,1900],
                                       [0,-1,0,0],
                                       [0,0,-1,200]])
        self.start_locate_info_transmit()

    
    def init_send_dict(self):
    
        #  0       偏移量（默认0） Int(4 个字节)
        # 1 数据长度（从序号 3 开始的所有字节数）Int(4 个字节)   16+ 12*8 = 112
        # 2 计数状态 Int(4 个字节) 
        # 3 识别状态 Int(4 个字节)0：未识别；1：识别目标悬浮；2：识别目标管路；3:识别目标沉底缆
        # 4 X 偏差_本体 Double(8 个字节) e_x_c 本体与相机
        # 5 Y 偏差_本体 Double(8 个字节) e_y_c 
        # 6 Z 偏差_本体 Double(8 个字节) e_z_c 
        # 7 p 偏差_本体 Double(8 个字节) e_pitch_c 
        # 8 q 偏差_本体 Double(8 个字节) e_roll_c 
        # 9 r 偏差_本体 Double(8 个字节) e_psi_c 
        # 10是否启用像素 Int(4 个字节) e_x_move  0：启动1：不启动
        # 11  左右 Int(4 个字节) e_y_move 0：不动1：左移2：右移
        # 12上下 Int(4 个字节) e_z_move 0：不动1：上移2：下移
        # 13 X 偏差_机械手 Double(8 个字节) e_x_manipulator_c  机械手基座与目标
        # 14 Y 偏差_机械手 Double(8 个字节) e_y_manipulator_c 
        # 15 Z 偏差_机械手 Double(8 个字节) e_z_manipulator_c 
        # 16 p 偏差_机械手 Double(8 个字节) e_p_manipulator_c 
        # 17 q 偏差_机械手 Double(8 个字节) e_q_manipulator_c 
        # 18 r 偏差_机械手 Double(8 个字节) e_r_manipulator_c
        send_dict = edict()
        send_dict.bias = 0
        send_dict.length = 112
        send_dict.time_cnt = 0
        send_dict.detect_mode = 0
        send_dict.e_x_c = 0
        send_dict.e_y_c = 0
        send_dict.e_z_c = 0
        send_dict.e_pitch_c = 0
        send_dict.e_roll_c = 0
        send_dict.e_psi_c = 0
        send_dict.e_x_move = 0
        send_dict.e_y_move = 0
        send_dict.e_z_move = 0
        send_dict.e_x_manipulator_c = 0
        send_dict.e_y_manipulator_c = 0
        send_dict.e_z_manipulator_c = 0
        send_dict.e_p_manipulator_c = 0
        send_dict.e_q_manipulator_c = 0
        send_dict.e_r_manipulator_c = 0
        return send_dict
      
    def start_locate_info_transmit(self):
        threading.Thread(target=self.send_locate_info).start()
    
    def send_locate_info(self):
        while True:
            time.sleep(1)
            counter = int(time.perf_counter() - self.start_time)
          
            
            data_trans = struct.pack('=iiiiddddddiiidddddd',
                                        self.send_dict.bias,
                                        self.send_dict.length,
                                        counter,
                                        self.send_dict.detect_mode,
                                        self.send_dict.e_x_c,
                                        self.send_dict.e_y_c,
                                        self.send_dict.e_z_c,
                                        self.send_dict.e_pitch_c,
                                        self.send_dict.e_roll_c,
                                        self.send_dict.e_psi_c,
                                        self.send_dict.e_x_move,
                                        self.send_dict.e_y_move,
                                        self.send_dict.e_z_move,
                                        self.send_dict.e_x_manipulator_c,
                                        self.send_dict.e_y_manipulator_c,
                                        self.send_dict.e_z_manipulator_c ,
                                        self.send_dict.e_p_manipulator_c ,
                                        self.send_dict.e_q_manipulator_c,
                                        self.send_dict.e_r_manipulator_c ,
                    )
            
            try:   
                sent = self.sock.sendto(data_trans, self.target_address)
                print(f"send ip:{self.target_address[0]},{self.target_address[1]}")
                print(f"send data:{self.send_dict.e_x_manipulator_c},\
                      {self.send_dict.e_y_manipulator_c},\
                      {self.send_dict.e_z_manipulator_c}")
            except Exception as e:
                print(e)
    def update_send_dict(self,obj_cls,obj_pos_incam):
        self.send_dict.detect_mode = obj_cls
        pos_inhand = np.dot(self.hand_eye_mat,np.array([obj_pos_incam[0],obj_pos_incam[1],obj_pos_incam[2],1]))
        #print(f"pose in hand:{pos_inhand}")
        # self.send_dict.e_x_manipulator_c = pos_inhand[0]
        # self.send_dict.e_y_manipulator_c = pos_inhand[1]
        # self.send_dict.e_z_manipulator_c = pos_inhand[2]
        self.send_dict.e_x_manipulator_c =1.1
        self.send_dict.e_y_manipulator_c = 2.2
        self.send_dict.e_z_manipulator_c = 3.3
        self.send_dict.e_p_manipulator_c = 4.4
        self.send_dict.e_q_manipulator_c = 5.5
        self.send_dict.e_r_manipulator_c = 6.6
        self.send_dict.e_x_c = obj_pos_incam[0]
        self.send_dict.e_y_c = obj_pos_incam[1]
        self.send_dict.e_z_c = obj_pos_incam[2]
        self.send_dict.e_pitch_c = 0
        self.send_dict.e_roll_c = 0
        self.send_dict.e_psi_c = 0
        self.send_dict.e_x_move = 1
        self.send_dict.e_y_move = 2
        self.send_dict.e_z_move = 3
    

class ImageTransmitUDP(TransmitUDP):
    def __init__(self,target_addr,logger_name="image",self_addr='192.168.1.131',target_port=44479,self_port=55579):
        super().__init__(target_addr,logger_name,self_addr,target_port,self_port)

class SonarTransmitUDP(TransmitUDP):
    def __init__(self,target_addr,logger_name="sonar",self_addr='192.168.1.131',target_port=44479,self_port=55579):
        super().__init__(target_addr,logger_name,self_addr,target_port,self_port)




class HandTransmitUDP(TransmitUDP):
    def __init__(self,target_addr,can_bitrate=50000,logger_name="hand",self_addr='192.168.1.200',target_port=55579,self_port=55509):
        super().__init__(target_addr,logger_name,self_addr,target_port,self_port)
        ##### init can   param: bitrate
        self.init_listen()
        try:
            self.can_transmiter =  CANTransmitter(can_bitrate)
        except Exception as e:
            print(e)
        ##bias,length,workmode,xyz,abc,a1,a2,a3,a4,a5,a6,grasp_cmd,cut-cmd
        self.recv_dict = self.init_recv_dict()
        self.send_dict = self.init_send_dict()
        self.start_time = time.perf_counter()
        self.do_cut = False
        # #bias,length,time_cnt,workmode,a1,a2,a3,a4,a5,a6,move_finish_flag,grasp_flag,cut_flag,error_flag
        # #self.hand_info_transmit_buffer = [0,72,0,0,  0.0,0.0,0.0,0.0,0.0,0.0,  0,0,0,0]
        self.motor_zero_pos = [0x6000,0xA000,0x9000,0x5000,0x8000]
        self.grasp_zero_pos = [0xC000,0x5000]
        self.hand_sovler = HandPoseSolver(66+50)
        self.cut_steps = 3
        self.prev_recv_msg = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        self.start_hand_info_transmit()
        time.sleep(3)
        
        
        self.arb_id = None
        self.data = None
        

    def start_hand_info_transmit(self):
        threading.Thread(target=self.send_hand_info()).start()
    def init_send_dict(self):
        send_dict = edict()
        send_dict.bias = 0
        send_dict.length = 116
        send_dict.time_cnt = 0
        send_dict.workmode = 0

        send_dict.a1 = 0
        send_dict.a2 = 0
        send_dict.a3 = 0
        send_dict.a4 = 0
        send_dict.a5 = 0
        send_dict.a6 = 0
        send_dict.x = 0
        send_dict.y = 0
        send_dict.z = 0
        send_dict.r = 0
        send_dict.p = 0
        send_dict.yaw = 0

        send_dict.move_flag = 0
        send_dict.grasp_flag = 0
        send_dict.cut_flag = 0
        send_dict.error_flag = 0
        send_dict.a1_error = 0
        send_dict.a2_error = 0
        send_dict.a3_error = 0
        send_dict.a4_error = 0
        send_dict.a5_error = 0
        send_dict.a6_error = 0
        return send_dict

    def init_recv_dict(self):
        recv_dict = edict()
        recv_dict.bias = 0
        recv_dict.length = 108
        recv_dict.time_cnt = 0
        recv_dict.workmode = 0
        recv_dict.x = 0
        recv_dict.y = 0
        recv_dict.z = 0
        recv_dict.a = 0
        recv_dict.b = 0
        recv_dict.c = 0
        recv_dict.a1 = 0
        recv_dict.a2 = 0
        recv_dict.a3 = 0
        recv_dict.a4 = 0
        recv_dict.a5 = 0
        recv_dict.a6 = 0
        recv_dict.grasp_cmd = 0
        recv_dict.cut_cmd = 0
        return recv_dict

    def get_hand_states(self):

        move_finish_flag = False
        grasp_flag = False

        error_a1 = (abs(self.recv_dict.a1 - self.send_dict.a1) < 1)
        error_a2 = (abs(self.recv_dict.a2 - self.send_dict.a2)< 1)
        error_a3 = (abs(self.recv_dict.a3 - self.send_dict.a3)< 1)
        error_a4 = (abs(self.recv_dict.a4 - self.send_dict.a4)< 1)
        error_a5 = (abs(self.recv_dict.a5 - self.send_dict.a5)< 1)
        if error_a1 & error_a2 & error_a3 & error_a4 & error_a5:
            move_finish_flag = True
        else:
            move_finish_flag = False
        
        if abs(self.can_transmiter.hand_angle_info[5] - self.grasp_zero_pos[1])/65535*360 < 1:
            grasp_flag = True
        else:
            grasp_flag = False
        
        return move_finish_flag, grasp_flag

    def send_hand_info(self):
        
        while True:
            time.sleep(1)
            counter = int(time.perf_counter() - self.start_time)
            move_finish_flag, grasp_flag = self.get_hand_states()
            self.send_dict.move_flag = int(move_finish_flag)
            self.send_dict.grasp_flag = int(grasp_flag)
            self.send_dict.a1 = self.can_transmiter.hand_angle_info[0]
            self.send_dict.a2 = self.can_transmiter.hand_angle_info[1]
            self.send_dict.a3 = self.can_transmiter.hand_angle_info[2]
            self.send_dict.a4 = self.can_transmiter.hand_angle_info[3]
            self.send_dict.a5 = self.can_transmiter.hand_angle_info[4]
            self.send_dict.a6 = self.can_transmiter.hand_angle_info[5]
            pose = self.hand_sovler.solver(self.can_transmiter.hand_angle_info)
            R = pose[:3,:3]
            position = pose[:3,3]
            rpy = self.hand_sovler.Rot2RPY(R)
            self.send_dict.x = position[0]
            self.send_dict.y = position[1]
            self.send_dict.z = position[2]
            self.send_dict.r = rpy[0]
            self.send_dict.p = rpy[1]
            self.send_dict.yaw = rpy[2]

            try:
                data_trans = struct.pack('=iiddddddddddddiiiiiiiiiii',
                            0,
                            int(self.send_dict.length),
                            self.send_dict.a1,
                            self.send_dict.a2,
                            self.send_dict.a3,
                            self.send_dict.a4,
                            self.send_dict.a5,
                            self.send_dict.a6,
                            self.send_dict.x,
                            self.send_dict.y,
                            self.send_dict.z,
                            int(self.send_dict.workmode),
                            counter,
                            int(self.send_dict.move_flag),
                            int(self.send_dict.grasp_flag),
                            int(self.send_dict.cut_flag),
                            int(self.send_dict.a1_error),
                            int(self.send_dict.a2_error),
                            int(self.send_dict.a3_error),
                            int(self.send_dict.a4_error),
                            int(self.send_dict.a5_error),
                            int(self.send_dict.a6_error)
                        )
            except Exception as e:
                print(f"hand info transmit error:{e}")

            try:
                print(f"send ip:{self.target_address[0]},{self.target_address[1]}")
                sent = self.sock.sendto(data_trans, self.target_address)
            except Exception as e:
                print(e)

    def cut_t(self):
        pass

    # def check_command(self,cmd):
    #     if cmd.grasp == 


    def init_listen(self):
        threading.Thread(target=self.listen).start()

    def do_command(self,work_flag,pos_flag):
        #print("do command")

        if int(self.recv_dict.cut_cmd) == 1 and work_flag:
            self.can_transmiter.cut(self.cut_steps)
            return
        if self.recv_dict.grasp_cmd == 1 and work_flag:
            print("do grasp")
            self.can_transmiter.grasp(1)
            return
        elif self.recv_dict.grasp_cmd == 0:
            self.can_transmiter.grasp(0)
            
        if not work_flag and pos_flag:
            if self.recv_dict.workmode == 0:
                x,y,z,a,b,c =   self.recv_dict.x,\
                                self.recv_dict.y,\
                                self.recv_dict.z,\
                                self.recv_dict.a,\
                                self.recv_dict.b,\
                                self.recv_dict.c
                mat = transform_xyzabc2mat([x,y,z,a,b,c])
                print(f'end effector:{mat}')
                angles = self.hand_sovler.reverse_solver(mat)
                if angles is None:
                    
                    self.logger.error(f"x:{x},y:{y},z:{z},a:{a},b:{b},c:{c}, no solution")
                    print('no solution')
                    self.send_dict.workmode = 2
                    return
                else:
                    angles = angles[0]
                    self.recv_dict.a1 = angles[0]
                    self.recv_dict.a2 = angles[1]
                    self.recv_dict.a3 = angles[2]
                    self.recv_dict.a4 = angles[3]
                    self.recv_dict.a5 = angles[4]
                    
                    #self.recv_dict.a6 = angles[5]
                    print(f"solved angles:{angles}")


                    for i in range(1,6):
                        self.can_transmiter.send_angle(i,self.recv_dict[f'a{i}'])

            else:
                for i in range(1,6):
                        print(f"angels:{self.recv_dict[f'a{i}']}")
                #print(f"angles:{}")
                if self.check_angle():
                    print("angle mode done")
                    for i in range(1,6):
                        self.can_transmiter.send_angle(i,self.recv_dict[f'a{i}'])
                else:
                    print("angle error")
                    self.logger.error(f"angle error- a1:{self.recv_dict.a1},a2:{self.recv_dict.a2},a3:{self.recv_dict.a3},a4:{self.recv_dict.a4},a5:{self.recv_dict.a5}")
                    self.send_dict.workmode = 3
                    return
    def check_angle(self):
        a1 = self.recv_dict.a1
        a2 = self.recv_dict.a2
        a3 = self.recv_dict.a3
        a4 = self.recv_dict.a4
        a5 = self.recv_dict.a5

        if (-45 <= a1 <= 180
            and 0 <= a2 <= 180 
            and -180 <= a3 <= 180
            and -90 <= a4 <= 225 
            and -157.5 <= a5 <= 157.5):
            return True
        else:
            return False

    def check_cmd(self,data):
        position_flag = False
        work_flag = False
        #wm,x,y,z,a,b,c,a1,a2,a3,a4,a5,a6,grasp,cut = self.prev_recv_msg
        # grasp
        if 1 == data[13]:
            if 1 == self.prev_recv_msg[13]:
                work_flag = False
                position_flag = False
            else:
                work_flag = True
                position_flag = False
            self.prev_recv_msg = data.copy()
            return work_flag, position_flag
        
        elif 1 == data[14]:
            if 1 == self.prev_recv_msg[14]:
                work_flag = False
                position_flag = False
                self.do_cut = True
            
            else:
                work_flag = True
                position_flag = False
                self.do_cut = False
            self.prev_recv_msg = data.copy()
            return work_flag, position_flag
        
        else:
            work_flag = False
            position_flag = True
            self.prev_recv_msg = data.copy()
            return work_flag, position_flag





      
    # def go_to_rest_position(self,):



    def process_data(self,data):
        #if data[0] == 0:
            ##接收了机械手的控制信号
        #state = self.check_command(data)
        try:
            zero, number, wm,a1,a2,a3,a4,a5,a6,a,b,c,x,y,z,grasp,cut = struct.unpack('=iiiddddddddddddii',data)
            work_flag, move_flag = self.check_cmd([wm,x,y,z,a,b,c,a1,a2,a3,a4,a5,a6,grasp,cut])
            print(f"move:{move_flag}.work:{work_flag}")
            self.logger.info(f"udp data recieved: workmode:{wm},x:{x},y:{y},z:{z},a:{a},b:{b},c:{c},angle1:{a1},angle2:{a2},angle3:{a3},angle4:{a4},angle5:{a5},angle6:{a6},grasp:{grasp},cut:{cut}")
        except Exception as e:
            self.logger.error(f"udp data process error:{e}")
            print("Wrong data recieved.")
        self.recv_dict.bias = zero
        self.recv_dict.length = number
        #self.recv_dict.time_cnt = 0
        self.recv_dict.workmode = wm
        self.recv_dict.x = x
        self.recv_dict.y = y
        self.recv_dict.z = z
        self.recv_dict.a = a
        self.recv_dict.b = b
        self.recv_dict.c = c
        self.recv_dict.a1 = a1
        self.recv_dict.a2 = a2
        self.recv_dict.a3 = a3
        self.recv_dict.a4 = a4
        self.recv_dict.a5 = a5
        self.recv_dict.a6 = a6
        self.recv_dict.grasp_cmd = grasp
        self.recv_dict.cut_cmd = cut
      
        print(f'recieved udp info: wordkmode:{wm},x:{x},y:{y},z:{z},a:{a},b:{b},c:{c},angle1:{a1},angle2:{a2},angle3:{a3},angle4:{a4},angle5:{a5},angle6:{a6},grasp:{grasp},cut:{cut}')
        self.do_command(work_flag,move_flag)
        #self.transmit2can(self.hand_info_recv_buffer)
   # def check_command(self,data):

    def listen(self):
        while True:
            try:
                data, address = self.sock.recvfrom(4096)
                if data:
                    if address[0]==self.target_address[0]:
                        print('transform data to hand')
                        
                        self.process_data(data)
                else:
                    print('udp data is none.')
            except Exception as e:
                pass
                #print(f"udp data recieve error:{e}")

                                        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ###添加参数
    #argparse.add_argument('--can', type=int, default=0, help='can port')
    parser.add_argument('--ip', type=str, default='192.168.1.133', help='baud rate')
    args = parser.parse_args()
    transmitor = HandTransmitUDP(args.ip)
   
    while True:
        time.sleep(1)
        a = input()
        if a == "1":
            angles = transmitor.can_transmiter.hand_angle_info
            for i in range(1,6):
                transmitor.can_transmiter.send_angle(i,angles[i])
        #transmitor.transmit2can(transmitor.hand_info_recv_buffer)
