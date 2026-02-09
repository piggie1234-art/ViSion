# -- coding: utf-8 --
import sys
import sys, os
sys.path.append("MvImport")
from MvCameraControl_class import *
from ctypes import *
#from PyQt5 import QtCore, QtWidgets, QtGui
from lib.utils import *
import threading
import time
import numpy as np
import cv2
libc = cdll.LoadLibrary('libc.so.6')
#获取选取设备信息的索引，通过[]之间的字符去解析
def TxtWrapBy(start_str, end, all):
    start = all.find(start_str)
    if start >= 0:
        start += len(start_str)
        end = all.find(end, start)
        if end >= 0:
            return all[start:end].strip()

#将返回的错误码转换为十六进制显示
def ToHexStr(num):
    chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    hexStr = ""
    if num < 0:
        num = num + 2**32
    while num >= 16:
        digit = num % 16
        hexStr = chaDic.get(digit, str(digit)) + hexStr
        num //= 16
    hexStr = chaDic.get(num, str(num)) + hexStr   
    return hexStr

global deviceList
global tlayerType
deviceList = MV_CC_DEVICE_INFO_LIST()
tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
#L_Device = cast(deviceList.pDeviceInfo[int(0)], POINTER(MV_CC_DEVICE_INFO)).contents

class HKCamera():

    def __init__(self,id) -> None:
      
        self.cam = MvCamera()
        self.cam_id = id
        self.b_open_device = False
        self.b_thread_closed = True
        self.b_start_grabbing = False
        self.open_device()

        #self.device_list = []
        self.img_buffer = []
        
    #ch:枚举相机 | en:enum devices
    # def enum_devices(self):

    #     ret = MvCamera.MV_CC_EnumDevices(self.tlayer_type, self.device_list)
    #     if ret != 0:
    #         print('Enum devices fail!')
    #         #tkinter.messagebox.showerror('show error','enum devices fail! ret = '+ ToHexStr(ret))

    #     if self.device_list.nDeviceNum == 0:
    #         print('find no device!')
    #         #tkinter.messagebox.showinfo('show info','find no device!')

    #     print ("Find %d devices!" % self.device_list.nDeviceNum)
    
    
    def open_device(self):
        if False == self.b_open_device:
            global deviceList
            global tlayerType
            cam_name = cast(deviceList.pDeviceInfo[int(self.cam_id)], POINTER(MV_CC_DEVICE_INFO)).contents
            ret = self.cam.MV_CC_CreateHandle(cam_name)
            if ret != 0:
                self.cam.MV_CC_DestroyHandle()
                print('create handle fail! ret = '+ ToHexStr(ret))
            #tkinter.messagebox.showerror('show error','create handle fail! ret = '+ self.To_hex_str(ret))
                return ret
            # ch:选择设备并创建句柄 | en:Select device and create handle
            # nConnectionNum = int(self.n_connect_num)
            # stDeviceList = cast(self.device_list.pDeviceInfo[self.cam_id], POINTER(MV_CC_DEVICE_INFO)).contents
            
            # ret = self.cam.MV_CC_CreateHandle(stDeviceList)
            # if ret != 0:
            #     self.cam.MV_CC_DestroyHandle()
            #     #tkinter.messagebox.showerror('show error','create handle fail! ret = '+ self.To_hex_str(ret))
            #     return ret
            # retry_count = 0
            ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            while ret != 0:
                print('open device fail! ret = '+ ToHexStr(ret))
                # if retry_count < 50:
                #     ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
                #     retry_count += 1
                # else:
                    #print('can not open devcice.')

                
                return ret
            print ("open device successfully!")
            self.b_open_device = True
            self.b_thread_closed = False

            #ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
            if cam_name.nTLayerType == MV_GIGE_DEVICE:
                nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
                if int(nPacketSize) > 0:
                    ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize",nPacketSize)
                    if ret != 0:
                        print ("warning: set packet size fail! ret[0x%x]" % ret)
                else:
                    print ("warning: set packet size fail! ret[0x%x]" % nPacketSize)

            stBool = c_bool(False)
            ret =self.cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)
            if ret != 0:
                print ("get acquisition frame rate enable fail! ret[0x%x]" % ret)

            # ch:设置触发模式为off | en:Set trigger mode as off
            ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            if ret != 0:
                print ("set trigger mode fail! ret[0x%x]" % ret)
            return 0

    def Start_grabbing(self,mut):
        if False == self.b_start_grabbing and True == self.b_open_device:
            self.b_exit = False
            ret = self.cam.MV_CC_StartGrabbing()
            if ret != 0:
                #tkinter.messagebox.showerror('show error','start grabbing fail! ret = '+ self.To_hex_str(ret))
                print("start grabbing fail! {}".format(self.To_hex_str(ret)))
                return
            self.b_start_grabbing = True
            
            try:
                #self.n_win_gui_id = random.randint(1,10000)
                h_thread_handle = threading.Thread(target=HKCamera.Work_thread, args=(self,mut))
                h_thread_handle.start()
                print ("start grabbing successfully!")
                self.b_thread_closed = True
            except:
                #tkinter.messagebox.showerror('show error','error: unable to start thread')
                False == self.b_start_grabbing
        
        self.Set_parameter(20,10000,23)

    def Stop_grabbing(self):
        if True == self.b_start_grabbing and self.b_open_device == True:
            #退出线程
            if True == self.b_thread_closed:
                Stop_thread(self.h_thread_handle)
                self.b_thread_closed = False
            ret = self.cam.MV_CC_StopGrabbing()
            if ret != 0:
                tkinter.messagebox.showerror('show error','stop grabbing fail! ret = '+self.To_hex_str(ret))
                return
            print ("stop grabbing successfully!")
            self.b_start_grabbing = False
            self.b_exit  = True      

    def Close_device(self):
        if True == self.b_open_device:
            #退出线程
            if True == self.b_thread_closed:
                Stop_thread(self.h_thread_handle)
                self.b_thread_closed = False
            ret = self.cam.MV_CC_CloseDevice()
            if ret != 0:
                #tkinter.messagebox.showerror('show error','close deivce fail! ret = '+self.To_hex_str(ret))
                return
                
        # ch:销毁句柄 | Destroy handle
        self.cam.MV_CC_DestroyHandle()
        self.b_open_device = False
        self.b_start_grabbing = False
        self.b_exit  = True
        print ("close device successfully!")

    def Get_parameter(self):
        if True == self.b_open_device:
            stFloatParam_FrameRate =  MVCC_FLOATVALUE()
            memset(byref(stFloatParam_FrameRate), 0, sizeof(MVCC_FLOATVALUE))
            stFloatParam_exposureTime = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_exposureTime), 0, sizeof(MVCC_FLOATVALUE))
            stFloatParam_gain = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_gain), 0, sizeof(MVCC_FLOATVALUE))
            ret = self.cam.MV_CC_GetFloatValue("AcquisitionFrameRate", stFloatParam_FrameRate)
            # if ret != 0:
            #     tkinter.messagebox.showerror('show error','get acquistion frame rate fail! ret = '+self.To_hex_str(ret))
            self.frame_rate = stFloatParam_FrameRate.fCurValue
            ret = self.cam.MV_CC_GetFloatValue("ExposureTime", stFloatParam_exposureTime)
            # if ret != 0:
            #     tkinter.messagebox.showerror('show error','get exposure time fail! ret = '+self.To_hex_str(ret))
            self.exposure_time = stFloatParam_exposureTime.fCurValue
            ret = self.cam.MV_CC_GetFloatValue("Gain", stFloatParam_gain)
            # if ret != 0:
            #     tkinter.messagebox.showerror('show error','get gain fail! ret = '+self.To_hex_str(ret))
            self.gain = stFloatParam_gain.fCurValue
            #tkinter.messagebox.showinfo('show info','get parameter success!')

    def Set_parameter(self,frameRate,exposureTime,gain):
        if '' == frameRate or '' == exposureTime or '' == gain:
            #tkinter.messagebox.showinfo('show info','please type in the text box !')
            return
        if True == self.b_open_device:
            ret = self.cam.MV_CC_SetFloatValue("ExposureTime",float(exposureTime))
            # if ret != 0:
            #     tkinter.messagebox.showerror('show error','set exposure time fail! ret = '+self.To_hex_str(ret))

            ret = self.cam.MV_CC_SetFloatValue("Gain",float(gain))
            # if ret != 0:
            #     tkinter.messagebox.showerror('show error','set gain fail! ret = '+self.To_hex_str(ret))

            ret = self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate",float(frameRate))
            # if ret != 0:
            #     tkinter.messagebox.showerror('show error','set acquistion frame rate fail! ret = '+self.To_hex_str(ret))

            # tkinter.messagebox.showinfo('show info','set parameter success!')

    

    def Is_mono_data(self,enGvspPixelType):
        if PixelType_Gvsp_Mono8 == enGvspPixelType or PixelType_Gvsp_Mono10 == enGvspPixelType \
            or PixelType_Gvsp_Mono10_Packed == enGvspPixelType or PixelType_Gvsp_Mono12 == enGvspPixelType \
            or PixelType_Gvsp_Mono12_Packed == enGvspPixelType:
            return True
        else:
            return False

    def Is_color_data(self,enGvspPixelType):
        if PixelType_Gvsp_BayerGR8 == enGvspPixelType or PixelType_Gvsp_BayerRG8 == enGvspPixelType \
            or PixelType_Gvsp_BayerGB8 == enGvspPixelType or PixelType_Gvsp_BayerBG8 == enGvspPixelType \
            or PixelType_Gvsp_BayerGR10 == enGvspPixelType or PixelType_Gvsp_BayerRG10 == enGvspPixelType \
            or PixelType_Gvsp_BayerGB10 == enGvspPixelType or PixelType_Gvsp_BayerBG10 == enGvspPixelType \
            or PixelType_Gvsp_BayerGR12 == enGvspPixelType or PixelType_Gvsp_BayerRG12 == enGvspPixelType \
            or PixelType_Gvsp_BayerGB12 == enGvspPixelType or PixelType_Gvsp_BayerBG12 == enGvspPixelType \
            or PixelType_Gvsp_BayerGR10_Packed == enGvspPixelType or PixelType_Gvsp_BayerRG10_Packed == enGvspPixelType \
            or PixelType_Gvsp_BayerGB10_Packed == enGvspPixelType or PixelType_Gvsp_BayerBG10_Packed == enGvspPixelType \
            or PixelType_Gvsp_BayerGR12_Packed == enGvspPixelType or PixelType_Gvsp_BayerRG12_Packed== enGvspPixelType \
            or PixelType_Gvsp_BayerGB12_Packed == enGvspPixelType or PixelType_Gvsp_BayerBG12_Packed == enGvspPixelType \
            or PixelType_Gvsp_YUV422_Packed == enGvspPixelType or PixelType_Gvsp_YUV422_YUYV_Packed == enGvspPixelType:
            return True
        else:
            return False

    def Mono_numpy(self,data,nWidth,nHeight):
        data_ = np.frombuffer(data, count=int(nWidth * nHeight), dtype=np.uint8, offset=0)
        data_mono_arr = data_.reshape(nHeight, nWidth)
        numArray = np.zeros([nHeight, nWidth, 1],"uint8")
        numArray[:, :, 0] = data_mono_arr
        return numArray

    def Color_numpy(self,data,nWidth,nHeight):
        data_ = np.frombuffer(data, count=int(nWidth*nHeight*3), dtype=np.uint8, offset=0)
        data_r = data_[0:nWidth*nHeight*3:3]
        data_g = data_[1:nWidth*nHeight*3:3]
        data_b = data_[2:nWidth*nHeight*3:3]

        data_r_arr = data_r.reshape(nHeight, nWidth)
        data_g_arr = data_g.reshape(nHeight, nWidth)
        data_b_arr = data_b.reshape(nHeight, nWidth)
        numArray = np.zeros([nHeight, nWidth, 3],"uint8")

        numArray[:, :, 0] = data_r_arr
        numArray[:, :, 1] = data_g_arr
        numArray[:, :, 2] = data_b_arr
        return numArray

    
    def Work_thread(self,mut):
        

        # ch:获取数据包大�?| en:Get payload size
        stParam =  MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
        
        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            print ("get payload size fail! ret[0x%x]" % ret)
            sys.exit()
        nPayloadSize = stParam.nCurValue

        # #image data buffer
        # stFrameInfo = MV_FRAME_OUT_INFO_EX()
        # memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
        data_buf = (c_ubyte * nPayloadSize)()

        #转换像素结构体赋值
        stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
        memset(byref(stConvertParam), 0, sizeof(stConvertParam))

        
        stOutFrame = MV_FRAME_OUT()
        img_buff = None
        buf_cache = None
        numArray = None
       
        while True:
            
            #test_t = time.time()
            ret = self.cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
            if ret != 0:
               continue
            #end_t = time.time() - test_t
            #print(f"image buffer method cost time:{end_t}")
            self.st_frame_info = stOutFrame.stFrameInfo
            self.n_save_image_size = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3 + 2048
            # if img_buff is None:
            #     img_buff = (c_ubyte * self.n_save_image_size)()
            if None == buf_cache:
                buf_cache = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
                #cdll.msvcrt.memcpy(byref(buf_cache), stOutFrame.pBufAddr, self.st_frame_info.nFrameLen)
            memmove(byref(buf_cache), stOutFrame.pBufAddr, self.st_frame_info.nFrameLen)
            #self.Save_jpg(buf_cache)
            
            # #转换像素结构体赋值
            stConvertParam.nWidth = self.st_frame_info.nWidth
            stConvertParam.nHeight = self.st_frame_info.nHeight
            stConvertParam.pSrcData = cast(buf_cache, POINTER(c_ubyte))
            stConvertParam.nSrcDataLen = self.st_frame_info.nFrameLen
            stConvertParam.enSrcPixelType = self.st_frame_info.enPixelType 

          

            mode = None     # array转为Image图像的转换模式
            # RGB8直接显示
            if PixelType_Gvsp_RGB8_Packed == self.st_frame_info.enPixelType :
                numArray = HKCamera.Color_numpy(self,buf_cache,self.st_frame_info.nWidth,self.st_frame_info.nHeight)
                mode = "RGB"

            # Mono8直接显示
            elif PixelType_Gvsp_Mono8 == self.st_frame_info.enPixelType :
                numArray = HKCamera.Mono_numpy(self,buf_cache,self.st_frame_info.nWidth,self.st_frame_info.nHeight)
                mode = "L"

            # 如果是彩色且非RGB则转为RGB后显示
            elif self.Is_color_data(self.st_frame_info.enPixelType):
                nConvertSize = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3
                stConvertParam.enDstPixelType = PixelType_Gvsp_RGB8_Packed
                stConvertParam.pDstBuffer = (c_ubyte * nConvertSize)()
                stConvertParam.nDstBufferSize = nConvertSize
                #time_start=time.time()
                ret = self.cam.MV_CC_ConvertPixelType(stConvertParam)
                #time_end=time.time()
                #print('MV_CC_ConvertPixelType to RGB:',time_end - time_start) 
                if ret != 0:
                    print('show error:convert pixel fail!')
                    continue
                img_buff = (c_ubyte * nConvertSize)()
                memmove(byref(img_buff), stConvertParam.pDstBuffer, stConvertParam.nDstLen)
                numArray = HKCamera.Color_numpy(self,img_buff,self.st_frame_info.nWidth,self.st_frame_info.nHeight)
                mode = "RGB"
                
            # 如果是黑白且非Mono8则转为Mono8后显示
            elif self.Is_mono_data(self.st_frame_info.enPixelType):
                nConvertSize = self.st_frame_info.nWidth * self.st_frame_info.nHeight
                stConvertParam.enDstPixelType = PixelType_Gvsp_Mono8
                stConvertParam.pDstBuffer = (c_ubyte * nConvertSize)()
                stConvertParam.nDstBufferSize = nConvertSize
                time_start=time.time()
                ret = self.obj_cam.MV_CC_ConvertPixelType(stConvertParam)
                time_end=time.time()
                #print('MV_CC_ConvertPixelType to Mono8:',time_end - time_start) 
                if ret != 0:
                    print('show error:convert pixel fail!')
                    continue
                libc.memcpy(byref(img_buff), stConvertParam.pDstBuffer, nConvertSize)
                numArray = HKCamera.Mono_numpy(img_buff,self.st_frame_info.nWidth,self.st_frame_info.nHeight)
                mode = "L"
            
            # get_cam_t = time.time() - cam_t
            # print(f"get image from camera:{get_cam_t} s")
            
            mut.acquire()
            #cv2.imwrite("test.jpg",numArray)
            
            
            self.img_buffer = numArray
            mut.release()
            nRet = self.cam.MV_CC_FreeImageBuffer(stOutFrame)

    def Save_jpg(self,buf_cache):
        if(None == buf_cache):
            return
        self.buf_save_image = None
        file_path = str(self.st_frame_info.nFrameNum) + ".jpg"
        self.n_save_image_size = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3 + 2048
        if self.buf_save_image is None:
            self.buf_save_image = (c_ubyte * self.n_save_image_size)()

        stParam = MV_SAVE_IMAGE_PARAM_EX()
        stParam.enImageType = MV_Image_Jpeg;                                        # ch:需要保存的图像类型 | en:Image format to save
        stParam.enPixelType = self.st_frame_info.enPixelType                               # ch:相机对应的像素格式 | en:Camera pixel type
        stParam.nWidth      = self.st_frame_info.nWidth                                    # ch:相机对应的宽 | en:Width
        stParam.nHeight     = self.st_frame_info.nHeight                                   # ch:相机对应的高 | en:Height
        stParam.nDataLen    = self.st_frame_info.nFrameLen
        stParam.pData       = cast(buf_cache, POINTER(c_ubyte))
        stParam.pImageBuffer=  cast(byref(self.buf_save_image), POINTER(c_ubyte)) 
        stParam.nBufferSize = self.n_save_image_size                                 # ch:存储节点的大小 | en:Buffer node size
        stParam.nJpgQuality = 80;                                                    # ch:jpg编码，仅在保存Jpg图像时有效。保存BMP时SDK内忽略该参数
        return_code = self.cam.MV_CC_SaveImageEx2(stParam)            

        if return_code != 0:
            print('show error save jpg fail! ret =')
            self.b_save_jpg = False
            return
        file_open = open(file_path.encode('ascii'), 'wb+')
        img_buff = (c_ubyte * stParam.nImageLen)()
        try:
            memmove(byref(img_buff), stParam.pImageBuffer, stParam.nImageLen)
            file_open.write(img_buff)
            self.b_save_jpg = False
            print('show info save jpg success!')
        except:
            self.b_save_jpg = False
            raise Exception("get one frame failed:%s" % e.message)
        if None != img_buff:
            del img_buff
        if None != self.buf_save_image:
            del self.buf_save_image
    
        