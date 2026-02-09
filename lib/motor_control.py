
from ctypes import *
import struct
import time
from collections import deque
import threading
VCI_USBCAN2 = 4
STATUS_OK = 1
class VCI_INIT_CONFIG(Structure):  
    _fields_ = [("AccCode", c_uint),
                ("AccMask", c_uint),
                ("Reserved", c_uint),
                ("Filter", c_ubyte),
                ("Timing0", c_ubyte),
                ("Timing1", c_ubyte),
                ("Mode", c_ubyte)
                ]  
class VCI_CAN_OBJ(Structure):  
    _fields_ = [("ID", c_uint),
                ("TimeStamp", c_uint),
                ("TimeFlag", c_ubyte),
                ("SendType", c_ubyte),
                ("RemoteFlag", c_ubyte),
                ("ExternFlag", c_ubyte),
                ("DataLen", c_ubyte),
                ("Data", c_ubyte*8),
                ("Reserved", c_ubyte*3)
                ] 

class VCI_CAN_OBJ_ARRAY(Structure):
    _fields_ = [('SIZE', c_uint16), ('STRUCT_ARRAY', POINTER(VCI_CAN_OBJ))]

    def __init__(self,num_of_structs):
                                                                 #这个括号不能少
        self.STRUCT_ARRAY = cast((VCI_CAN_OBJ * num_of_structs)(),POINTER(VCI_CAN_OBJ))#结构体数组
        self.SIZE = num_of_structs#结构体长度
        self.ADDR = self.STRUCT_ARRAY[0]#结构体数组地址  byref()转c地址

class MotorParams():
    pulse_per_round = 65536
    pitch = 2
    
    def __init__(self,id,velo=100,acc=10,state=False) -> None:

        self.position = 0.0
        self.pulse = 0
        self.acceleration = acc
        self.velocity = velo
        self.id = id
        self.state = state
        self.pos_in16 = (0,0,0,0)

    
    def set_position(self,pos_in16):
        self.pos_in16 = pos_in16
        self.position = self.pulse2mm(pos_in16)
        
    @classmethod
    def pulse2mm(cls,pulses):
       
        mm = pulses/cls.pulse_per_round*cls.pitch
        return mm
    @classmethod
    def mm2pulse(cls,mm):
        circles = mm/cls.pitch
        pulses = int(circles*cls.pulse_per_round)
        return pulses

    @classmethod
    def pulse2bit(cls,pulse):

        a = (pulse >> 24) & 0xFF
        b = (pulse >> 16) & 0xFF
        c = (pulse >> 8) & 0xFF
        d = pulse & 0xFF

        result = [a, b, c, d]
                
        return result
        
        
class MotorController():
    canDLL = cdll.LoadLibrary('lib/libcontrolcan.so')
    rx_vci_can_obj = VCI_CAN_OBJ_ARRAY(2500)#结构体数组
    

    def __init__(self):

        self.init_CAN()
        self.axis2id = {'x':1,'y':2,'z':3}
        self.id2axis = {1:'x',2:'y',3:'z'}
        self.motors = {}
        self.velo_en = [False,False,False]

        self.motors['x'] = MotorParams(self.axis2id['x'])
        self.motors['y'] = MotorParams(self.axis2id['y'])
        self.motors['z'] = MotorParams(self.axis2id['z'])
        self.id_can_dict = {1:0,2:1,3:1}

        self.pos_q = {}
        self.pos_q['x'] = deque(maxlen=3)
        self.pos_q['y'] = deque(maxlen=3)
        self.pos_q['z'] = deque(maxlen=3)

        self.cur_q = {}
        self.cur_q['x'] = deque(maxlen=3)
        self.cur_q['y'] = deque(maxlen=3)
        self.cur_q['z'] = deque(maxlen=3)
        self.init_thread()
        self.led_rate = 0
    
    def init_thread(self):
        # 发送电机位置和电流获取指令的线程
        threading.Thread(target=self.position_req).start()
        # 获取数据的线程
        threading.Thread(target=self.recieve_buffer_data).start()
    
    def set_led_up(self):
        if self.led_rate < 255:
            self.led_rate += 16
        ubyte_3array = c_ubyte*3
        b = ubyte_3array(0x00, 0x00, 0x00)
        
        ubyte_array = c_ubyte*8
        ### 03 读取  07 位置信息
        a = ubyte_array(self.led_rate, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00)
    
        vci_can_obj = VCI_CAN_OBJ(0x3F1, 0, 0, 1, 0, 1,  2, a, b)#单次发送
        ret = self.canDLL.VCI_Transmit(VCI_USBCAN2, 0, 0, byref(vci_can_obj), 1)

    
    def set_led_down(self):
        if self.led_rate >0:
            self.led_rate -= 16
        ubyte_3array = c_ubyte*3
        b = ubyte_3array(0x00, 0x00, 0x00)
        
        ubyte_array = c_ubyte*8
        ### 03 读取  07 位置信息
        a = ubyte_array(self.led_rate, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00)
    
        vci_can_obj = VCI_CAN_OBJ(0x3F1, 0, 0, 1, 0, 1,  2, a, b)#单次发送
        ret = self.canDLL.VCI_Transmit(VCI_USBCAN2, 0, 0, byref(vci_can_obj), 1)
    
        


    def get_motor_position(self,axis):
        ubyte_3array = c_ubyte*3
        b = ubyte_3array(0x00, 0x00, 0x00)
        
        ubyte_array = c_ubyte*8
        ### 03 读取  07 位置信息
        a = ubyte_array(0x03, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00)
        id = self.motors[axis].id
        can_id = self.id_can_dict[id]
        vci_can_obj = VCI_CAN_OBJ(id, 0, 0, 1, 0, 0,  6, a, b)#单次发送
        ret = self.canDLL.VCI_Transmit(VCI_USBCAN2, 0, can_id, byref(vci_can_obj), 1)
    
    def get_motor_current(self,axis):
        ubyte_3array = c_ubyte*3
        b = ubyte_3array(0x00, 0x00, 0x00)
        
        ubyte_array = c_ubyte*8
        ### 03 读取  05 电流信息
        a = ubyte_array(0x03, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00)
        id = self.motors[axis].id
        can_id = self.id_can_dict[id]
        vci_can_obj = VCI_CAN_OBJ(id, 0, 0, 1, 0, 0,  6, a, b)#单次发送
        ret = self.canDLL.VCI_Transmit(VCI_USBCAN2, 0, can_id, byref(vci_can_obj), 1)

    def recieve_buffer_data(self):
        while True:
            time.sleep(0.03)
            n_msg = self.canDLL.VCI_GetReceiveNum(VCI_USBCAN2, 0, 0)
            if n_msg > 0:
                self.canDLL.VCI_Receive(VCI_USBCAN2, 0, 0, byref(self.rx_vci_can_obj.ADDR), 2500, 0)
                for i in range(n_msg):
                    id = int(self.rx_vci_can_obj.STRUCT_ARRAY[i].ID)
                    data = self.rx_vci_can_obj.STRUCT_ARRAY[i].Data
                    data = list(data)
                    self.process_data(id,data)
            
            n_msg = self.canDLL.VCI_GetReceiveNum(VCI_USBCAN2, 0, 1)
            if n_msg > 0:
                self.canDLL.VCI_Receive(VCI_USBCAN2, 0, 1, byref(self.rx_vci_can_obj.ADDR), 2500, 0)
                for i in range(n_msg):
                    id = int(self.rx_vci_can_obj.STRUCT_ARRAY[i].ID)
                    data = self.rx_vci_can_obj.STRUCT_ARRAY[i].Data
                    data = list(data)
                    self.process_data(id,data)
            

    def process_data(self,id,data):
        if data[0] == 4 and data[1] == 7:
            pulse = self.int_from_bytes(data[2:6])
            pos = MotorParams.pulse2mm(pulse)
            try:
                self.pos_q[self.id2axis[id]].append(pos)
            except Exception as e:
                print(f'can data reviceve error:{e}')
        elif data[0] == 4 and data[1] == 5:
            try:
                cur = self.int_from_bytes(data[2:6])
                self.cur_q[self.id2axis[id]].append(cur)
            except Exception as e:
                print(f'can data reviceve error:{e}')

    def check_state(self,can_id,motor_id):

        ret=0
        time_spent = time.time()
        while ret <= 0:#如果没有接收到数据，一直循环查询接收。
            n_msg = self.canDLL.VCI_GetReceiveNum(VCI_USBCAN2, 0, can_id)
            #print(n_msg)
           
            ret = self.canDLL.VCI_Receive(VCI_USBCAN2, 0, can_id, byref(self.rx_vci_can_obj.ADDR), 2500, 0)
            if time.time() - time_spent >=2:
                return False

        data = self.rx_vci_can_obj.STRUCT_ARRAY[0].Data
        id = int(self.rx_vci_can_obj.STRUCT_ARRAY[0].ID)
        data = list(data)
        if data[2]==1 and id == motor_id:
            return True
        else:
            return False
    
    #def get_init_pulse(self,can_id,motor_id):

    def get_init_pulse(self, can_id, motor_id, timeout=3):

        #start_time = time.time()
        while True:
            n_msg = self.canDLL.VCI_GetReceiveNum(VCI_USBCAN2, 0, can_id)
            print(n_msg)
            ret = self.canDLL.VCI_Receive(VCI_USBCAN2, 0, can_id, byref(self.rx_vci_can_obj.ADDR), 2500, 0)
            if ret > 0:
                data = self.rx_vci_can_obj.STRUCT_ARRAY[0].Data
                id = int(self.rx_vci_can_obj.STRUCT_ARRAY[0].ID)
                data = list(data)
                if data[0] == 4 and data[1] == 7 and id == motor_id:
                    print("ok")
                    return True, data
                else:
                    return False, data
            # if time.time() - start_time > timeout:
            #     print(f"id:{motor_id} signal not recieved.")
            #     break
        return False, []

        # ret=0
        # while ret <= 0:#如果没有接收到数据，一直循环查询接收。
        #     ret = self.canDLL.VCI_Receive(VCI_USBCAN2, 0, can_id, byref(self.rx_vci_can_obj.ADDR), 2500, 0)
        # data = self.rx_vci_can_obj.STRUCT_ARRAY[0].Data
        # id = int(self.rx_vci_can_obj.STRUCT_ARRAY[0].ID)
        # data = list(data)
        # if data[0]==4 and data[1]==7 and id == motor_id:
        #     return True, data
        # else:
        #     return False, data
    
    def int_from_bytes(self,data):

        byte_data = bytes(data)
        result = int.from_bytes(byte_data, byteorder='big', signed=True)
        return result

    def get_cur_pulse(self,axis):
        ubyte_3array = c_ubyte*3
        b = ubyte_3array(0x00, 0x00, 0x00)
        
        ubyte_array = c_ubyte*8
        a = ubyte_array(0x03, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00)
        id = self.motors[axis].id
        can_id = self.id_can_dict[id]
        vci_can_obj = VCI_CAN_OBJ(id, 0, 0, 1, 0, 0,  6, a, b)#单次发送
        ret = self.canDLL.VCI_Transmit(VCI_USBCAN2, 0, can_id, byref(vci_can_obj), 1)
        if ret != STATUS_OK:
            Warning(f"ask for init pulse failed. can_id:{can_id},motor_id:{id}\r\n")
        ret=0
        while ret <= 0:#如果没有接收到数据，一直循环查询接收。
            ret,data = self.get_init_pulse(can_id,id)
        #if ret:
        self.motors[axis].pulse = self.int_from_bytes(data[2:6])
        self.motors[axis].position = MotorParams.pulse2mm(self.motors[axis].pulse)

        return True


       
    def deactivate_motor(self,axis):
        motor_id = self.axis2id[axis]
        #电机失能
        ubyte_3array = c_ubyte*3
        ubyte_array = c_ubyte*8
        a = ubyte_array(0x01, 0x10, 0x00, 0x00, 0x00, 0x00,0x00,0x00)
        b = ubyte_3array(0x00, 0x00, 0x00)
        vci_can_obj = VCI_CAN_OBJ(motor_id, 0, 0, 1, 0, 0,  6, a, b)#单次发送
        
        ret = self.canDLL.VCI_Transmit(VCI_USBCAN2, 0, self.id_can_dict[motor_id], byref(vci_can_obj), 1)
        if ret != STATUS_OK:
            Warning(f"电机失能发送失败.can_id:{self.id_can_dict[motor_id]},motor_id:{motor_id}\r\n")
        
        self.velo_en[motor_id-1] = False
        
   
    # def get_motor_state(self):
    #     x_id = self.motors['x'].id
    #     y_id = self.motors['y'].id
    #     z_id = self.motors['z'].id
    #     ##从电机x读取当前pulse
    #     ubyte_3array = c_ubyte*3
    #     b = ubyte_3array(0x00, 0x00, 0x00)
    #     #电机使能
    #     ubyte_array = c_ubyte*6
    #     a = ubyte_array(0x03, 0x07, 0x00, 0x00, 0x00, 0x00)

    #     vci_can_obj = VCI_CAN_OBJ(x_id, 0, 0, 1, 0, 0,  6, a, b)#单次发送
        
    #     ret = self.canDLL.VCI_Transmit(VCI_USBCAN2, 0, self.id_can_dict[x_id], byref(vci_can_obj), 1)
    #     if ret != STATUS_OK:
    #         Warning(f"电机使能发送失败.can_id:{self.id_can_dict[x_id]},motor_id:{x_id}\r\n")
        

    
    def move(self,axis,distance,velocity=10):
        pulses = MotorParams.mm2pulse(distance)
        if self.get_cur_pulse(axis):
            self.motors[axis].pulse += pulses
            self.motors[axis].pos_in16 = MotorParams.pulse2bit(self.motors[axis].pulse)
            self._move_position_(self.motors[axis])
            print("move success")
        else:
            print("move failed.")
            
    def shutdown_CAN(self):
        #关闭
        self.canDLL.VCI_CloseDevice(VCI_USBCAN2, 0) 
        print("关闭设备")
    
    def move_v(self,axis,velo=10):

        ##change v to 16bit-pulse
        # 将 -10000 转换为 32 位有符号整数的二进制数据
        pulse = int(velo * 65536 / 60.0)
        data = MotorParams.pulse2bit(pulse)
        # bin_data = struct.pack("<i", pulse)

        # hex_str = bin_data.hex()
        # int_list = [int(hex_str[i:i+2], 16) for i in range(0, len(hex_str), 2)]

        #hex_list = [hex(num) for num in int_list]

        motor_id = self.axis2id[axis]
        can_id = self.id_can_dict[motor_id]
        if self.velo_en[motor_id-1]:
            return
        ubyte_3array = c_ubyte*3
        b = ubyte_3array(0x00, 0x00, 0x00)
        #电机使能
        ubyte_array = c_ubyte*8
        a = ubyte_array(0x01, 0x10, 0x00, 0x00, 0x00, 0x01,0x00,0x00)

        vci_can_obj = VCI_CAN_OBJ(motor_id, 0, 0, 1, 0, 0,  6, a, b)#单次发送
        ret = self.canDLL.VCI_Transmit(VCI_USBCAN2, 0, can_id, byref(vci_can_obj), 1)
        if ret != STATUS_OK:
            Warning(f"电机使能发送失败.can_id:{can_id},motor_id:{motor_id}\r\n")
        
        # if not self.check_state(can_id,motor_id):
        #     Warning("使能 failed")

        
        ## 设置velocity mode
        a = ubyte_array(0x01, 0x0F, 0x00, 0x00, 0x00, 0x03,0x00,0x00)
        
        vci_can_obj = VCI_CAN_OBJ(motor_id, 0, 0, 1, 0, 0,  6, a, b)#单次发送
        
        ret = self.canDLL.VCI_Transmit(VCI_USBCAN2, 0, can_id, byref(vci_can_obj), 1)
        if ret != STATUS_OK:
            Warning(f"速度参数发送失败.can_id:{can_id},motor_id:{motor_id}\r\n")
    #         lsof: WARNING: can't stat() fuse.gvfsd-fuse file system /run/user/125/gvfs
    #   Output information may be incomplete.
        # if not self.check_state(can_id,motor_id):
        #     Warning("velocity mode failed")

        
        ## 设置速度参数
        a = ubyte_array(0x01, 0x09, data[0], data[1], data[2], data[3],0x00,0x00)
        
        vci_can_obj = VCI_CAN_OBJ(motor_id, 0, 0, 1, 0, 0,  6, a, b)#单次发送
        
        ret = self.canDLL.VCI_Transmit(VCI_USBCAN2, 0, can_id, byref(vci_can_obj), 1)
        if ret != STATUS_OK:
            Warning(f"速度参数发送失败.can_id:{can_id},motor_id:{motor_id}\r\n")
        
        # if not self.check_state(can_id,motor_id):
        #     Warning("速度参数 failed")
    

    def _move_position_(self,motor):
        ubyte_3array = c_ubyte*3
        b = ubyte_3array(0x00, 0x00, 0x00)
        #电机使能
        ubyte_array = c_ubyte*8
        a = ubyte_array(0x01, 0x10, 0x00, 0x00, 0x00, 0x01,0x00,0x00)

        vci_can_obj = VCI_CAN_OBJ(motor.id, 0, 0, 1, 0, 0,  6, a, b)#单次发送
        can_id = self.id_can_dict[motor.id]
        
        ret = self.canDLL.VCI_Transmit(VCI_USBCAN2, 0, can_id, byref(vci_can_obj), 1)
        if ret != STATUS_OK:
            print(f"电机使能发送失败.can_id:{can_id},motor_id:{motor.id}\r\n")
        
        # if not self.check_state(can_id,motor.id):
        #     print("使能 failed")

        ## 设置position mode
        a = ubyte_array(0x01, 0x0F, 0x00, 0x00, 0x00, 0x01,0x00,0x00)
        
        vci_can_obj = VCI_CAN_OBJ(motor.id, 0, 0, 1, 0, 0,  6, a, b)#单次发送
        
        ret = self.canDLL.VCI_Transmit(VCI_USBCAN2, 0, self.id_can_dict[motor.id], byref(vci_can_obj), 1)
        if ret != STATUS_OK:
            print(f"速度参数发送失败.can_id:{self.id_can_dict[motor.id]},motor_id:{motor.id}\r\n")
            
        # if not self.check_state(can_id,motor.id):
        #     print("position mode failed")

        
        ## 设置速度参数
        ##change v to 16bit-pulse
        # 将 -10000 转换为 32 位有符号整数的二进制数据
        pulse = int(motor.velocity * 65536 / 60.0)
        # bin_data = struct.pack("<i", pulse)
        data = MotorParams.pulse2bit(pulse)
        # hex_str = bin_data.hex()
        # int_list = [int(hex_str[i:i+2], 16) for i in range(0, len(hex_str), 2)]

        a = ubyte_array(0x01, 0x09, data[0], data[1], data[2], data[3],0x00,0x00)
        
        vci_can_obj = VCI_CAN_OBJ(motor.id, 0, 0, 1, 0, 0,  6, a, b)#单次发送
        
        ret = self.canDLL.VCI_Transmit(VCI_USBCAN2, 0, self.id_can_dict[motor.id], byref(vci_can_obj), 1)
        if ret != STATUS_OK:
            print(f"速度参数发送失败.can_id:{self.id_can_dict[motor.id]},motor_id:{motor.id}\r\n")
        
        # if not self.check_state(can_id,motor.id):
        #     print("速度参数 failed")
        ## 设置位置参数
        #a = ubyte_array(0x01, 0x0A, 0x00,0x00,0x00,0x00,0x00,0x00)
        a = ubyte_array(0x01, 0x0A, motor.pos_in16[0], motor.pos_in16[1], motor.pos_in16[2], motor.pos_in16[3])
        
        vci_can_obj = VCI_CAN_OBJ(motor.id, 0, 0, 1, 0, 0,  6, a, b)#单次发送
        
        ret = self.canDLL.VCI_Transmit(VCI_USBCAN2, 0, self.id_can_dict[motor.id], byref(vci_can_obj), 1)
        if ret != STATUS_OK:
            print(f"position参数发送失败.can_id:{self.id_can_dict[motor.id]},motor_id:{motor.id}\r\n")
            
        # if not self.check_state(can_id,motor.id):
        #     print("relative position failed")

    def position_req(self):
        while True:
            time.sleep(0.03)
            ### 请求位置
            self.get_motor_position('x')
            self.get_motor_position('y')
            self.get_motor_position('z')

            ### 获取电流
            self.get_motor_current('x')
            self.get_motor_current('y')
            self.get_motor_current('z')



    def init_CAN(self):
        #self.canDLL.VCI_CloseDevice(VCI_USBCAN2, 0) 
        ret = self.canDLL.VCI_OpenDevice(VCI_USBCAN2, 0, 0)
        if ret == STATUS_OK:
            print('调用 VCI_OpenDevice成功\r\n')
        else:
            print('调用 VCI_OpenDevice出错\r\n')
        
        #初始0通道
        vci_initconfig = VCI_INIT_CONFIG(0x80000000, 0xFFFFFFFF, 0,
                                 0, 0x00, 0x1C, 0)#波特率500k，正常模式
        ret = self.canDLL.VCI_InitCAN(VCI_USBCAN2, 0, 0, byref(vci_initconfig))
        if ret == STATUS_OK:
             print('调用 VCI_InitCAN1成功\r\n')
        else:
            print('调用 VCI_InitCAN1出错\r\n')
        ret = self.canDLL.VCI_StartCAN(VCI_USBCAN2, 0, 0)
        if ret == STATUS_OK:
            print('调用 VCI_StartCAN1成功\r\n')
        else:
            print('调用 VCI_StartCAN1出错\r\n')
        
        #初始1通道
        vci_initconfig = VCI_INIT_CONFIG(0x80000000, 0xFFFFFFFF, 0,
                                 0, 0x00, 0x1C, 0)#波特率500k，正常模式
        ret = self.canDLL.VCI_InitCAN(VCI_USBCAN2, 0, 1, byref(vci_initconfig))
        if ret == STATUS_OK:
             print('调用 VCI_InitCAN2成功\r\n')
        else:
            print('调用 VCI_InitCAN2出错\r\n')
        ret = self.canDLL.VCI_StartCAN(VCI_USBCAN2, 0, 1)
        if ret == STATUS_OK:
            print('调用 VCI_StartCAN2成功\r\n')
        else:
            print('调用 VCI_StartCAN2出错\r\n')


if __name__ == '__main__':
    motors = MotorController()
    #can.init_CAN()
    motors.move('z',-50)

    # while True:
    #     if keyboard.is_pressed('q'):  # 检测是否按下了'q'键
    #         motors.move_v('x',10)
            
    #     else:
    #         motors.deactivate_motor('x')
        

           
