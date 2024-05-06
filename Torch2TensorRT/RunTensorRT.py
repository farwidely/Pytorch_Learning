

import time
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import numpy as np
import os
import tensorrt as trt

TRT_LOGGER = trt.Logger()
model_path = 'MyModel.onnx'
engine_file_path = "MyModel.trt"


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_tensor_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_tensor_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime, \
        runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    # 开始推理
    image = cv2.imread(f'xxx.jpg', cv2.IMREAD_COLOR)
    # image = cv2.resize(image, (224, 224))
    # print(image.shape)
    image = image[np.newaxis, :, :, :].astype(np.float32)
    inputs[0].host = image
    trt_outputs = do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    print(trt_outputs)
