
import numpy as np
import cv2
from mobile_sam import sam_model_registry, SamPredictor
import onnxruntime
import time 
import tensorrt as trt
from cuda import cudart
import cv2

def apply_coords(coords: np.ndarray, original_size,new_size) -> np.ndarray:
    """
    Expects a numpy array of length 2 in the final dimension. Requires the
    original image size in (H, W) format.
    """
    old_h, old_w = original_size
    new_h, new_w = new_size
    coords = coords.copy()
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    return coords


class TrackingModelTRTv2():
    def __init__(self,config):
        onnx_model_path = config.decoder_path
        trtFile = config.encoder_path
       
        # onnx_model_path = 'trt_models/mobile_sam_decoder.onnx'
        # trtFile = 'trt_models/vit_t_encoder.plan'
        self.vit_t_engine,self.vit_context = self.init_engine(trtFile)
        self.ort_session = onnxruntime.InferenceSession(onnx_model_path,providers=['CUDAExecutionProvider'])
        self.img_shape = config.image_shape
        checkpoint = "trt_models/mobile_sam.pt"
        #trtFile = "trt_models/model.plan"
        #self.vit_t_engine,self.vit_context = self.init_engine(trtFile)
        sam = sam_model_registry["vit_t"](checkpoint=checkpoint)
        sam.to(device='cuda')
        self.predictor = SamPredictor(sam)

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
        context.set_input_shape(lTensorName[0], [1,3,1024,1024])                            # create Excution Context from the engine (analogy to a GPU context, or a CPU process)
        return engine, context

    def preprocess(self,img):
        scale = min(1024.0 / img.shape[0], 1024.0 / img.shape[1])
        new_size = (int(np.round(img.shape[1]*scale)), int(np.round(img.shape[0]*scale)))  # 新的图像大小
        resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)[:,:,::-1].astype(np.float32)
        pixel_mean = np.array([123.675, 116.28, 103.53])[None,None,:]
        pixel_std = np.array([58.395, 57.12, 57.375])[None,None,:]
        resized_img = ((resized_img - pixel_mean ) / pixel_std)#.astype(np.float32)
        # 创建一个1024x1024的背景
        final_img = np.zeros((1024, 1024, 3), dtype=np.float32)

        # 计算pasting的起始点，这将会把resized_img放到final_img的中间位置
        start_x = (final_img.shape[1] - resized_img.shape[1]) // 2
        start_y = (final_img.shape[0] - resized_img.shape[0]) // 2

        # 把resized_img放到final_img的指定位置
        final_img[start_y : start_y+resized_img.shape[0], start_x : start_x+resized_img.shape[1]] = resized_img
        #print(f"ours:{final_img[:,:,0]}")
        # 将final_img的维度调整为模型输入的维度
        input_img = final_img.transpose((2, 0, 1))  # 从[1024,1024,3]转化为[3,1024,1024]
        input_img = np.expand_dims(input_img, axis=0)  # 从[3,1024,1024]转化为[1,3,1024,1024]


    def track(self,img,box):
        input_img = self.predictor.set_image(img).cpu().numpy()
        
        nIO = self.vit_t_engine.num_io_tensors                                                 # since TensorRT 8.5, the concept of Binding is replaced by I/O Tensor, all the APIs with "binding" in their name are deprecated
        lTensorName = [self.vit_t_engine.get_tensor_name(i) for i in range(nIO)]               # get a list of I/O tensor names of the engine, because all I/O tensor in Engine and Excution Context are indexed by name, not binding number like TensorRT 8.4 or before
        nInput = [self.vit_t_engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)  # get the count of input tensor

        bufferH = []   
            #.cpu().numpy()                                                             # prepare the memory buffer on host and device
        bufferH.append(np.ascontiguousarray(input_img, dtype=trt.nptype(self.vit_t_engine.get_tensor_dtype(lTensorName[0]))))
        for i in range(nInput, nIO):
            bufferH.append(np.empty(self.vit_context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(self.vit_t_engine.get_tensor_dtype(lTensorName[i]))))
        bufferD = []
        for i in range(nIO):
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

        for i in range(nInput):                                                     # copy input data from host buffer into device buffer
            cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        for i in range(nIO):
            self.vit_context.set_tensor_address(lTensorName[i], int(bufferD[i]))             # set address of all input and output data in device buffer

        self.vit_context.execute_async_v3(0)                                                 # do inference computation

        for i in range(nInput, nIO):                                                # copy output data from device buffer into host buffer
            cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        # for i in range(nIO):
        #     print(lTensorName[i])
        #     print(bufferH[i])

        for b in bufferD:                                                           # free the GPU memory buffer after all work
            cudart.cudaFree(b)

        image_embedding = bufferH[1]


        onnx_coord = np.array(box)[None, :].astype(np.int64)
        onnx_label = np.array([2,3])[None, :].astype(np.float32)
        onnx_coord = apply_coords(onnx_coord, self.img_shape,[1024,1024]).astype(np.float32)

        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        ort_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(self.img_shape, dtype=np.float32)
        }
        

        masks, _, _ = self.ort_session.run(None, ort_inputs)

        return masks
    

class TrackingModelTRT():
    def __init__(self,config):
       
        onnx_model_path = config.decoder_path
        trtFile = config.encoder_path
        self.vit_t_engine,self.vit_context = self.init_engine(trtFile)
        self.ort_session = onnxruntime.InferenceSession(onnx_model_path,providers=['CUDAExecutionProvider'])
        self.img_shape = config.image_shape
        checkpoint = "trt_models/mobile_sam.pt"
        onnx_model_path = "trt_models/mobile_sam.onnx"
        model_type = "vit_t"
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device='cuda')
        self.predictor = SamPredictor(sam)

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
        context.set_input_shape(lTensorName[0], [1,3,1024,1024])                            # create Excution Context from the engine (analogy to a GPU context, or a CPU process)
        return engine, context

    def preprocess(self,img):
        scale = min(1024.0 / img.shape[0], 1024.0 / img.shape[1])
        new_size = (int(np.round(img.shape[1]*scale)), int(np.round(img.shape[0]*scale)))  # 新的图像大小
        resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)[:,:,::-1].astype(np.float32)
        pixel_mean = np.array([123.675, 116.28, 103.53])[None,None,:]
        pixel_std = np.array([58.395, 57.12, 57.375])[None,None,:]
        resized_img = ((resized_img - pixel_mean ) / pixel_std)#.astype(np.float32)
        # 创建一个1024x1024的背景
        final_img = np.zeros((1024, 1024, 3), dtype=np.float32)

        # 计算pasting的起始点，这将会把resized_img放到final_img的中间位置
        start_x = (final_img.shape[1] - resized_img.shape[1]) // 2
        start_y = (final_img.shape[0] - resized_img.shape[0]) // 2

        # 把resized_img放到final_img的指定位置
        final_img[start_y : start_y+resized_img.shape[0], start_x : start_x+resized_img.shape[1]] = resized_img
        #print(f"ours:{final_img[:,:,0]}")
        # 将final_img的维度调整为模型输入的维度
        input_img = final_img.transpose((2, 0, 1))  # 从[1024,1024,3]转化为[3,1024,1024]
        input_img = np.expand_dims(input_img, axis=0)  # 从[3,1024,1024]转化为[1,3,1024,1024]


    def track(self,img,box):
        
        start = time.time()
        input_img = self.predictor.set_image(img)
        #input_img = self.preprocess(img)
        #print(f'img preprocess time:{time.time()-start}')
        nIO = self.vit_t_engine.num_io_tensors                                                 # since TensorRT 8.5, the concept of Binding is replaced by I/O Tensor, all the APIs with "binding" in their name are deprecated
        lTensorName = [self.vit_t_engine.get_tensor_name(i) for i in range(nIO)]               # get a list of I/O tensor names of the engine, because all I/O tensor in Engine and Excution Context are indexed by name, not binding number like TensorRT 8.4 or before
        nInput = [self.vit_t_engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)  # get the count of input tensor

        bufferH = []   
            #.cpu().numpy()                                                             # prepare the memory buffer on host and device
        bufferH.append(np.ascontiguousarray(input_img.cpu().numpy(), dtype=trt.nptype(self.vit_t_engine.get_tensor_dtype(lTensorName[0]))))
        for i in range(nInput, nIO):
            bufferH.append(np.empty(self.vit_context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(self.vit_t_engine.get_tensor_dtype(lTensorName[i]))))
        bufferD = []
        for i in range(nIO):
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

        for i in range(nInput):                                                     # copy input data from host buffer into device buffer
            cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        for i in range(nIO):
            self.vit_context.set_tensor_address(lTensorName[i], int(bufferD[i]))             # set address of all input and output data in device buffer

        self.vit_context.execute_async_v3(0)                                                 # do inference computation

        for i in range(nInput, nIO):                                                # copy output data from device buffer into host buffer
            cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        # for i in range(nIO):
        #     print(lTensorName[i])
        #     print(bufferH[i])

        for b in bufferD:                                                           # free the GPU memory buffer after all work
            cudart.cudaFree(b)

        image_embedding = bufferH[1]


        # onnx_coord = np.array(box)[None, :].astype(np.int64)
        # onnx_label = np.array([2,3])[None, :].astype(np.float32)
        # onnx_coord = apply_coords(onnx_coord, self.img_shape,[1024,1024]).astype(np.float32)

        # onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        # onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        # ort_inputs = {
        #     "image_embeddings": image_embedding,
        #     "point_coords": onnx_coord,
        #     "point_labels": onnx_label,
        #     "mask_input": onnx_mask_input,
        #     "has_mask_input": onnx_has_mask_input,
        #     "orig_im_size": np.array(self.img_shape, dtype=np.float32)
        # }
        

        # masks, _, _ = self.ort_session.run(None, ort_inputs)
        # masks = masks > 0.7
        # #mask = masks[0][0].astype(np.uint8) * 255
        # #cv2.imwrite("mask.jpg",mask)
        masks = None
        return masks,image_embedding,np.ascontiguousarray(input_img.cpu().numpy())
    

class TrackingModel():
    def __init__(self):
        checkpoint = "trt_models/mobile_sam.pt"
        onnx_model_path= "trt_models/mobile_sam.onnx"
        model_type = "vit_t"
        trtFile = "trt_models/model.plan"
        #self.vit_t_engine,self.vit_context = self.init_engine(trtFile)
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device='cuda')
        self.predictor = SamPredictor(sam)
        self.ort_session = onnxruntime.InferenceSession(onnx_model_path,providers=['CUDAExecutionProvider'])
        self.img_shape = [1080,1920]
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
        context.set_input_shape(lTensorName[0], [1,3,1024,1024])                            # create Excution Context from the engine (analogy to a GPU context, or a CPU process)
        return engine, context

    def preprocess(self,img):
        scale = min(1024.0 / img.shape[0], 1024.0 / img.shape[1])
        new_size = (int(np.round(img.shape[1]*scale)), int(np.round(img.shape[0]*scale)))  # 新的图像大小
        resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)[:,:,::-1].astype(np.float32)
        pixel_mean = np.array([123.675, 116.28, 103.53])[None,None,:]
        pixel_std = np.array([58.395, 57.12, 57.375])[None,None,:]
        resized_img = ((resized_img - pixel_mean ) / pixel_std)#.astype(np.float32)
        # 创建一个1024x1024的背景
        final_img = np.zeros((1024, 1024, 3), dtype=np.float32)

        # 计算pasting的起始点，这将会把resized_img放到final_img的中间位置
        start_x = (final_img.shape[1] - resized_img.shape[1]) // 2
        start_y = (final_img.shape[0] - resized_img.shape[0]) // 2

        # 把resized_img放到final_img的指定位置
        final_img[start_y : start_y+resized_img.shape[0], start_x : start_x+resized_img.shape[1]] = resized_img
        #print(f"ours:{final_img[:,:,0]}")
        # 将final_img的维度调整为模型输入的维度
        input_img = final_img.transpose((2, 0, 1))  # 从[1024,1024,3]转化为[3,1024,1024]
        input_img = np.expand_dims(input_img, axis=0)  # 从[3,1024,1024]转化为[1,3,1024,1024]


    def track(self,img,box):
        numpy_array = self.predictor.set_image(img)
        # input_img = self.preprocess(img)

        # nIO = self.vit_t_engine.num_io_tensors                                                 # since TensorRT 8.5, the concept of Binding is replaced by I/O Tensor, all the APIs with "binding" in their name are deprecated
        # lTensorName = [self.vit_t_engine.get_tensor_name(i) for i in range(nIO)]               # get a list of I/O tensor names of the engine, because all I/O tensor in Engine and Excution Context are indexed by name, not binding number like TensorRT 8.4 or before
        # nInput = [self.vit_t_engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)  # get the count of input tensor

        # bufferH = []                                                                # prepare the memory buffer on host and device
        # bufferH.append(np.ascontiguousarray(numpy_array))
        # for i in range(nInput, nIO):
        #     bufferH.append(np.empty(self.vit_context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(self.vit_t_engine.get_tensor_dtype(lTensorName[i]))))
        # bufferD = []
        # for i in range(nIO):
        #     bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

        # for i in range(nInput):                                                     # copy input data from host buffer into device buffer
        #     cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        # for i in range(nIO):
        #     self.vit_context.set_tensor_address(lTensorName[i], int(bufferD[i]))             # set address of all input and output data in device buffer

        # self.vit_context.execute_async_v3(0)                                                 # do inference computation

        # for i in range(nInput, nIO):                                                # copy output data from device buffer into host buffer
        #     cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        # # for i in range(nIO):
        # #     print(lTensorName[i])
        # #     print(bufferH[i])

        # for b in bufferD:                                                           # free the GPU memory buffer after all work
        #     cudart.cudaFree(b)

        # start = time.time()
        
        # image_embedding = bufferH[1]
        image_embedding = self.predictor.get_image_embedding().cpu().numpy()
        onnx_coord = np.array(box)[None, :].astype(np.int64)
        onnx_label = np.array([2,3])[None, :].astype(np.float32)
        onnx_coord = self.predictor.transform.apply_coords(onnx_coord, self.img_shape).astype(np.float32)

        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        # ort_inputs = {
        #     "image_embeddings": image_embedding,
        #     "point_coords": onnx_coord,
        #     "point_labels": onnx_label,
        #     "mask_input": onnx_mask_input,
        #     "has_mask_input": onnx_has_mask_input,
        #     "orig_im_size": np.array(self.img_shape, dtype=np.float32)
        # }

        # masks, _, _ = self.ort_session.run(None, ort_inputs)
        # #masks = masks > 0.5
        # #mask = masks[0][0].astype(np.uint8) * 255
        # #cv2.imwrite("mask.jpg",mask)
        masks = None
        return masks,image_embedding,np.ascontiguousarray(numpy_array.cpu().numpy())

if __name__ == '__main__':
        

    # 读取图像
    img = cv2.imread('l_rect.jpg')

    mobile_sam = TrackingModel()
    mobile_sam.track(img,[[100,100],[200,200]])

