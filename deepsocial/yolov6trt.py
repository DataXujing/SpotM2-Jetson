
import numpy as np
import cv2
import os
import tensorrt as trt
from cuda import cudart

INPUT_W = 640
INPUT_H = 640

logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, '')


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



def get_detect(context,engine,cv_image):
    # image = cv2.imread(os.path.join("./test_img",file))
    image, image_raw, h, w = preprocess(cv_image)

    # 分配内存
    inputH0 = np.ascontiguousarray(image.astype(np.float32).reshape(-1))
    outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
    outputH1 = np.empty(context.get_binding_shape(2), dtype=trt.nptype(engine.get_binding_dtype(2)))
    outputH2 = np.empty(context.get_binding_shape(3), dtype=trt.nptype(engine.get_binding_dtype(3)))
    outputH3 = np.empty(context.get_binding_shape(4), dtype=trt.nptype(engine.get_binding_dtype(4)))

    # 分配显存
    _, inputD0 = cudart.cudaMalloc(inputH0.nbytes)
    _, outputD0 = cudart.cudaMalloc(outputH0.nbytes)
    _, outputD1 = cudart.cudaMalloc(outputH1.nbytes)
    _, outputD2 = cudart.cudaMalloc(outputH2.nbytes)
    _, outputD3 = cudart.cudaMalloc(outputH3.nbytes)

    # input 内存copy到显存
    cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    # 执行推断
    context.execute_v2([int(inputD0), int(outputD0), int(outputD1), int(outputD2), int(outputD3)])

    # output 显存copy到内存
    cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    cudart.cudaMemcpy(outputH1.ctypes.data, outputD1, outputH1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    cudart.cudaMemcpy(outputH2.ctypes.data, outputD2, outputH2.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    cudart.cudaMemcpy(outputH3.ctypes.data, outputD3, outputH3.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    pred_box = rescale((INPUT_H,INPUT_W), outputH1[:,:outputH0[0],:][0], image_raw.shape).round()
    pred_score = outputH2[:,:outputH0[0]][0]
    pred_label = outputH3[:,:outputH0[0]][0]

    # 释放显存
    cudart.cudaFree(inputD0)
    cudart.cudaFree(outputD0)
    cudart.cudaFree(outputD1)
    cudart.cudaFree(outputD2)
    cudart.cudaFree(outputD3)

    return pred_box, pred_score, pred_label




if __name__ == "__main__":
    # 反序列化模型
    engine_file = "./model/yolov6n.plan"
    with open(engine_file, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    print(engine.get_binding_index("image_arrays"))  #0
    print(context.get_binding_shape(engine.get_binding_index("num_detections")))
    print(engine.get_binding_index("num_detections"))  #1
    print(engine.get_binding_index("nmsed_boxes"))  #2
    print(engine.get_binding_index("nmsed_scores"))  #3
    print(engine.get_binding_index("nmsed_classes"))  #4



    files = os.listdir("./test_img")
    class_names = ["face","mask","person"]



    # for file in files:
    #     image = cv2.imread(os.path.join("./test_img",file))
    cap = cv2.VideoCapture("test.mp4")

    # 获取fps 每秒多少张图片
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # 获取视频图像宽、高
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # # 可以指定图像的宽高，但要与写入时一致
    # size = (224, 224)

    # 生成处理后的视频文件
    # 根据原视频，生成新视频文件，
    videoWriter = cv2.VideoWriter('detect.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)


    color=((240, 32, 160),(127,255,0),(255,0,0))


    while True:
        ret, image = cap.read()
        image, image_raw, h, w = preprocess(image)

        # 分配内存
        inputH0 = np.ascontiguousarray(image.astype(np.float32).reshape(-1))
        outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
        outputH1 = np.empty(context.get_binding_shape(2), dtype=trt.nptype(engine.get_binding_dtype(2)))
        outputH2 = np.empty(context.get_binding_shape(3), dtype=trt.nptype(engine.get_binding_dtype(3)))
        outputH3 = np.empty(context.get_binding_shape(4), dtype=trt.nptype(engine.get_binding_dtype(4)))

        # 分配显存
        _, inputD0 = cudart.cudaMalloc(inputH0.nbytes)
        _, outputD0 = cudart.cudaMalloc(outputH0.nbytes)
        _, outputD1 = cudart.cudaMalloc(outputH1.nbytes)
        _, outputD2 = cudart.cudaMalloc(outputH2.nbytes)
        _, outputD3 = cudart.cudaMalloc(outputH3.nbytes)

        # input 内存copy到显存
        cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        # 执行推断
        context.execute_v2([int(inputD0), int(outputD0), int(outputD1), int(outputD2), int(outputD3)])

        # output 显存copy到内存
        cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        cudart.cudaMemcpy(outputH1.ctypes.data, outputD1, outputH1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        cudart.cudaMemcpy(outputH2.ctypes.data, outputD2, outputH2.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        cudart.cudaMemcpy(outputH3.ctypes.data, outputD3, outputH3.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        # print(outputH1[:,:outputH0[0],:][0][0])


        pred_box = rescale((INPUT_H,INPUT_W), outputH1[:,:outputH0[0],:][0], image_raw.shape).round()
        pred_score = outputH2[:,:outputH0[0]][0]
        pred_label = outputH3[:,:outputH0[0]][0]



        for i in range(outputH0[0]):

            class_num = int(pred_label[i])  # integer class
            label = f'{class_names[class_num]} {pred_score[i]:.2f}'

            plot_box_and_label(image_raw, max(round(sum(image_raw.shape) / 2 * 0.001), 2), pred_box[i], label,color[class_num])


            # cv2.imwrite(os.path.join("./res",file),image_raw)

        cv2.imshow("res",image_raw)
        # 写入一帧
        videoWriter.write(image_raw)
        cv2.waitKey(30)






        # 释放显存
        cudart.cudaFree(inputD0)
        cudart.cudaFree(outputD0)
        cudart.cudaFree(outputD1)
        cudart.cudaFree(outputD2)
        cudart.cudaFree(outputD3)