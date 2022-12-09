
import numpy as np
import cv2
import os
import tensorrt as trt
# from cuda import cudart  # cuda-python在jetson上不好装
import pycuda.driver as cuda  #https://documen.tician.de/pycuda/driver.html
# 此导入将导致pycuda自动管理CUDA上下文的创建和清理。
import pycuda.autoinit

INPUT_W = 640
INPUT_H = 640

# logger = trt.Logger(trt.Logger.ERROR)
# trt.init_libnvinfer_plugins(logger, '')


def preprocess(cv_image):
    """
    description: Read an image from image path, convert it to RGB,
                 resize and pad it to target size, normalize to [0,1],
                 transform to NCHW format.
    param:
        input_image_path: str, image path
    return:
        image:  the processed image
        image_raw: the original image
        h: original height
        w: original width
    """
    image_raw = cv_image
    h, w, c = image_raw.shape
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    # Calculate widht and height and paddings
    r_w = INPUT_W / w
    r_h = INPUT_H / h
    if r_h > r_w:
        tw = INPUT_W
        th = int(r_w * h)
        tx1 = tx2 = 0
        ty1 = int((INPUT_H - th) / 2)
        ty2 = INPUT_H - th - ty1
    else:
        tw = int(r_h * w)
        th = INPUT_H
        tx1 = int((INPUT_W - tw) / 2)
        tx2 = INPUT_W - tw - tx1
        ty1 = ty2 = 0
    # Resize the image with long side while maintaining ratio
    image = cv2.resize(image, (tw, th))
    # Pad the short side with (128,128,128)
    image = cv2.copyMakeBorder(image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128))
    image = image.astype(np.float32)
    # Normalize to [0,1]
    image /= 255.0
    # HWC to CHW format:
    image = np.transpose(image, [2, 0, 1])
    # CHW to NCHW format
    image = np.expand_dims(image, axis=0)
    # Convert the image to row-major order, also known as "C order":
    image = np.ascontiguousarray(image)
    return image, image_raw, h, w


def rescale(ori_shape, boxes, target_shape):
    '''Rescale the output to the original image shape'''
    ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
    padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

    boxes[:, [0, 2]] -= padding[0]
    boxes[:, [1, 3]] -= padding[1]
    boxes[:, :4] /= ratio

    # boxes[:, 0].clamp(0, target_shape[1])  # x1
    # boxes[:, 1].clamp(0, target_shape[0])  # y1
    # boxes[:, 2].clamp(0, target_shape[1])  # x2
    # boxes[:, 3].clamp(0, target_shape[0])  # y2

    return boxes



def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    # Add one xyxy box to image with label
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color,
                    thickness=tf, lineType=cv2.LINE_AA)


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TrtYOLOv6(object):
  
    def __init__(self, model, input_shape=(640, 640)):
        self.cfx = cuda.Device(0).make_context()
        """Initialize TensorRT plugins, engine and conetxt."""
        self.model = model
        
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.trt_logger , '')
        self.engine = self._load_engine()
        self.context = self._create_context()
        self.inputs, self.outputs, self.bindings, self.stream = \
            self.allocate_buffers(self.engine)
        self.inference_fn = self.do_inference_v2




    def allocate_buffers(self, engine):
        
        """Allocates all host/device in/out buffers required for an engine."""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * \
                engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
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


    def do_inference(self, context, bindings, inputs, outputs, stream, batch_size=1):
        """do_inference (for TensorRT 6.x or lower)
        
        This function is generalized for multiple inputs/outputs.
        Inputs and outputs are expected to be lists of HostDeviceMem objects.
        """
        self.cfx.push()
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async(batch_size=batch_size,
                            bindings=bindings,
                            stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]


    def do_inference_v2(self, context, bindings, inputs, outputs, stream):
        """do_inference_v2 (for TensorRT 7.0+)

        This function is generalized for multiple inputs/outputs for full
        dimension networks.
        Inputs and outputs are expected to be lists of HostDeviceMem objects.
        """
        self.cfx.push()
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        self.cfx.pop()
        return [out.host for out in outputs]

    def _load_engine(self):
        TRTbin = '%s' % self.model
        with open(TRTbin, 'rb') as f:
            return trt.Runtime(self.trt_logger).deserialize_cuda_engine(f.read())

    def _create_context(self):
        return self.engine.create_execution_context()



    def __del__(self):
        """Free CUDA memories."""
        del self.stream
        del self.outputs
        del self.inputs

    def detect(self, cv_image):
        image, image_raw, h, w = preprocess(cv_image)

        self.inputs[0].host = np.ascontiguousarray(image)
        trt_outputs = self.inference_fn(
            context=self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream)

        # Before doing post-processing, we need to reshape the outputs
        # as do_inference() will give us flat arrays.

        outputH0 = trt_outputs[0].reshape(self.context.get_binding_shape(1))
        outputH1 = trt_outputs[1].reshape(self.context.get_binding_shape(2))
        outputH2 = trt_outputs[2].reshape(self.context.get_binding_shape(3))
        outputH3 = trt_outputs[3].reshape(self.context.get_binding_shape(4))


        pred_box = rescale((INPUT_H,INPUT_W), outputH1[:,:outputH0[0],:][0], image_raw.shape).round()
        pred_score = outputH2[:,:outputH0[0]][0]
        pred_label = outputH3[:,:outputH0[0]][0]

        

        return pred_box, pred_score, pred_label


