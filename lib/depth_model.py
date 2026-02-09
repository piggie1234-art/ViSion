# from __future__ import print_function, divisionen
import sys
sys.path.append(".")
import argparse
import os
import torch.backends.cudnn as cudnn
import numpy as np
import time
import cv2
from easydict import EasyDict as edict
cudnn.benchmark = True
import json
import tensorrt as trt
from cuda import cudart
import torchvision.transforms as transforms


class StereoBM():
    def __init__(self):
        self.stereo = cv2.StereoBM_create(numDisparities=320, blockSize=9)
    
    def estimate_depth(self,l_img,r_img):
        l_img = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)
        r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)
        disparity = self.stereo.compute(l_img, r_img)/16.0
        return disparity

def get_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

class DepthEstimateModelTRT():

    def __init__(self,config) -> None:
        self.engine,self.context = self.init_engine(config.model_path)
        self.preprocess = get_transform()
    

    def init_engine(self,trt_path):
        logger = trt.Logger(trt.Logger.ERROR)                                                             # load serialized network and skip building process if .plan file existed
        with open(trt_path, "rb") as f:
            engineString = f.read()
        if engineString == None:
            print("Failed getting serialized engine!")
            return
        print("Succeeded getting serialized engine!")
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)          # create inference Engine using Runtime
        if engine == None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")

        nIO = engine.num_io_tensors                                                 # since TensorRT 8.5, the concept of Binding is replaced by I/O Tensor, all the APIs with "binding" in their name are deprecated
        lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]               # get a list of I/O tensor names of the engine, because all I/O tensor in Engine and Excution Context are indexed by name, not binding number like TensorRT 8.4 or before
        nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)  # get the count of input tensor
        #nOutput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.OUTPUT)  # get the count of output tensor

        context = engine.create_execution_context()    
        context.set_input_shape(lTensorName[0], [1,3,480,640])                            # create Excution Context from the engine (analogy to a GPU context, or a CPU process)
        context.set_input_shape(lTensorName[1], [1,3,480,640])  
        return engine, context
    
    def estimate_depth(self,l_img,r_img):
        start = time.time()
        
        l_img = self.preprocess(l_img)[None,:]
        r_img = self.preprocess(r_img)[None,:]
        
        nIO = self.engine.num_io_tensors                                                 # since TensorRT 8.5, the concept of Binding is replaced by I/O Tensor, all the APIs with "binding" in their name are deprecated
        lTensorName = [self.engine.get_tensor_name(i) for i in range(nIO)]               # get a list of I/O tensor names of the engine, because all I/O tensor in Engine and Excution Context are indexed by name, not binding number like TensorRT 8.4 or before
        nInput = [self.engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)  # get the count of input tensor

        bufferH = []                                                                # prepare the memory buffer on host and device
        bufferH.append(np.ascontiguousarray(l_img.numpy(), dtype=trt.nptype(self.engine.get_tensor_dtype(lTensorName[0]))))
        bufferH.append(np.ascontiguousarray(r_img.numpy(), dtype=trt.nptype(self.engine.get_tensor_dtype(lTensorName[1]))))
        for i in range(nInput, nIO):
            bufferH.append(np.empty(self.context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(self.engine.get_tensor_dtype(lTensorName[i]))))
        bufferD = []
        for i in range(nIO):
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

        for i in range(nInput):                                                     # copy input data from host buffer into device buffer
            cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        for i in range(nIO):
            self.context.set_tensor_address(lTensorName[i], int(bufferD[i]))             # set address of all input and output data in device buffer
        
        self.context.execute_async_v3(0)                                                 # do inference computation

        for i in range(nInput, nIO):                                                # copy output data from device buffer into host buffer
            cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        for b in bufferD:                                                           # free the GPU memory buffer after all work
            cudart.cudaFree(b)
        
        disp_ests = bufferH[2]

        disp_est_np_trt = disp_ests.transpose(1,2,0).copy()
        disp_est_uint_trt = np.round(disp_est_np_trt * 256).astype(np.uint16)

        pred_np_show = cv2.applyColorMap(cv2.convertScaleAbs(disp_est_uint_trt, alpha=0.01),cv2.COLORMAP_JET)[:,:,::-1]
        #print(f'depth estimate time:{time.time()-start}')
        return pred_np_show, disp_est_np_trt



if __name__ == '__main__':
    
    dp = DepthEstimateModelTRT()
 
    l_img = cv2.imread("im1.png")[:,:,::-1]
    r_img = cv2.imread("im0.png")[:,:,::-1]
    l_img = cv2.resize(l_img,(640,480))
    r_img = cv2.resize(r_img,(640,480))
    for  i in range(10000):
        disp_im, disp_np = dp.estimate_depth(l_img,r_img)

    cv2.imwrite("output_disp.jpg",disp_im)
