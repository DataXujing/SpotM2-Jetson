
import subprocess
import cv2
import time
from datetime import timedelta
from flask import Flask, render_template, Response, request, jsonify, make_response

from base64 import b64decode, b64encode
import os
import numpy as np
import PIL
import io
import time
import matplotlib.pyplot as plt

from deepsocial import *
from sort import *

#pycuda
import pycuda.driver as cuda
import pycuda.autoinit

# from cuda import cudart
import tensorrt as trt
from yolov6 import *



class_names = ["face","mask","person"]


#---实例化sort类，目标跟踪所用
mot_tracker = Sort(max_age=25, min_hits=4, iou_threshold=0.3)

# --deepsocial params
######################## Frame number
StartFrom  = 0 
EndAt      = -1                       #-1 for the end of the video
######################## (0:OFF/ 1:ON) Outputs
CouplesDetection    = 1               # Enable Couple Detection 
DTC                 = 1                # Detection, Tracking and Couples 
SocialDistance      = 1
CrowdMap            = 1
# MoveMap             = 0
# ViolationMap        = 0             
# RiskMap             = 0
######################## Units are Pixel
ViolationDistForIndivisuals = 28 
ViolationDistForCouples     = 31
####
CircleradiusForIndivsual    = 14
CircleradiusForCouples      = 17
######################## 
MembershipDistForCouples    = (16 , 10) # (Forward, Behind) per Pixel
MembershipTimeForCouples    = 35        # Time for considering as a couple (per Frame)
######################## (0:OFF/ 1:ON)
CorrectionShift  = 1                    # Ignore people in the margins of the video
HumanHeightLimit = 200000                 # Ignore people with unusual heights
########################
Transparency        = 0.7

ReductionFactor  = 2 #2
calibration      = [[180,162],[618,0],[552,540],[682,464]]




my_log = []

# btn状态
center_btn = 0
walk_btn = 0

img_id = 0
move2type = {
    'btn_dog_top': 'w',
    'btn_dog_bottom': 's',
    'btn_dog_left': 'a',
    'btn_dog_right': 'd',
    'btn_dog_round': 'u'
}



# 1. roslaunch spot_micro_motion_cmd motion_cmd.launch
# 2. rosrun spot_micro_keyboard_command spotMicroKeyboardMove.py
p_i2cpwd_board = subprocess.Popen(
    'rosrun i2cpwm_board i2cpwm_board', shell=True, cwd='./')
p_motion_cmd_server = subprocess.Popen(
    'roslaunch spot_micro_motion_cmd motion_cmd.launch run_standalone:=true', shell=True, cwd='./')
# p_keyboard_move = subprocess.Popen('rosrun spot_micro_keyboard_command spotMicroKeyboardMove.py', shell=True, stdin=subprocess.PIPE cwd='./')
p_keyboard_move = subprocess.Popen(
    'python3 dog_ctrl.py', shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd='./')


class VideoCamera(object):
    def __init__(self,yolov6):
        
        self.cap = cv2.VideoCapture("./test.MOV")
        self.cap.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        self.height, self.width = self.frame_height // ReductionFactor, self.frame_width // ReductionFactor
        # self.height, self.width = 720 // ReductionFactor, 1280 // ReductionFactor

        # deepsocial
        self.colorPool = ColorGenerator(size = 3000)
        self._centroid_dict = dict()
        self._numberOFpeople = list()
        self._greenZone = list()
        self._redZone = list()
        self._yellowZone = list()
        self._final_redZone = list()
        self._relation = dict()
        self._couples = dict()
        self._trackMap = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._crowdMap = np.zeros((self.height, self.width), dtype=np.int) 
        self._allPeople = 0
        # _counter = 1
        # frame = 0

        self.yolov6 = yolov6


 
    def __del__(self):
        self.cap.release()
        self.yolov6.cfx.pop()


    def ai_func(self,frame):
        '''
        # AI识别，YOLOv6-n 做行人识别和口罩检测，并TensorRT加速
        # 调用deepsocial 做人群距离检测
        '''

        save_tag = 0
        frame_resized = cv2.resize(frame,(self.width, self.height), interpolation=cv2.INTER_LINEAR)
        image = frame_resized
        e = birds_eye(image, calibration)

        # ---------------detect--------------------
        # 在线程中执行
        # pred_box, pred_score, pred_label = get_detect(context,engine,frame_resized)
        pred_box, pred_score, pred_label = self.yolov6.detect(frame_resized)

        # 解析detect
        humans,masks = extract_humans(pred_box, pred_score, pred_label)

        # 跟踪算法
        track_bbs_ids = mot_tracker.update(humans) if len(humans) != 0 else humans

        self._centroid_dict, centroid_dict, self.partImage = centroid(track_bbs_ids, image, calibration, self._centroid_dict, CorrectionShift, HumanHeightLimit)
        redZone, greenZone = find_zone(centroid_dict, self._greenZone, self._redZone, criteria=ViolationDistForIndivisuals)

        if CouplesDetection:
            self._relation, relation = find_relation(e, centroid_dict, MembershipDistForCouples, redZone, self._couples, self._relation)
            self._couples, couples, coupleZone = find_couples(image, self._centroid_dict, relation, MembershipTimeForCouples, self._couples)
            yellowZone, final_redZone, redGroups = find_redGroups(image, centroid_dict, calibration, ViolationDistForCouples, redZone, coupleZone, couples , self._yellowZone, self._final_redZone)
        else:
            couples = []
            coupleZone = []
            yellowZone = []
            redGroups = redZone
            final_redZone = redZone


        if DTC:  # 画detect
            DTC_image = image.copy()
            self._trackMap = Apply_trackmap(centroid_dict, self._trackMap, self.colorPool, 3)
            DTC_image = cv2.add(e.convrt2Image(self._trackMap), image) 
            DTCShow = DTC_image
            for id, box in centroid_dict.items():
                center_bird = box[0], box[1]
                if not id in coupleZone:
                    cv2.rectangle(DTCShow,(box[4], box[5]),(box[6], box[7]),(0,255,0),2)
                    cv2.rectangle(DTCShow,(box[4], box[5]-13),(box[4]+len(str(id))*10, box[5]),(0,200,255),-1)
                    cv2.putText(DTCShow,str(id),(box[4]+2, box[5]-2),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1,cv2.LINE_AA)

            for coupled in couples:
                p1 , p2 = coupled
                couplesID = couples[coupled]['id']
                couplesBox = couples[coupled]['box']
                cv2.rectangle(DTCShow, couplesBox[2:4], couplesBox[4:], (0,150,255), 4)
                loc = couplesBox[0] , couplesBox[3]
                offset = len(str(couplesID)*5)
                captionBox = (loc[0] - offset, loc[1]-13), (loc[0] + offset, loc[1])
                cv2.rectangle(DTCShow,captionBox[0],captionBox[1],(0,200,255),-1)
                wc = captionBox[1][0] - captionBox[0][0]
                hc = captionBox[1][1] - captionBox[0][1]
                cx = captionBox[0][0] + wc // 2
                cy = captionBox[0][1] + hc // 2
                textLoc = (cx - offset, cy + 4)
                cv2.putText(DTCShow, str(couplesID) ,(textLoc),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1,cv2.LINE_AA)
            # 把mask也画出来 [xmin, ymin, xmax, ymax,confidence,label]
            for mask_box in masks:
                save_tag = 1
                cv2.rectangle(DTCShow,(int(mask_box[0]), int(mask_box[1])),(int(mask_box[2]), int(mask_box[3])),(240,32,160),2)
                cv2.rectangle(DTCShow,(int(mask_box[0]), int(mask_box[1]-13)),(int(mask_box[0]+len("face")*10), int(mask_box[1])),(0,200,255),-1)
                cv2.putText(DTCShow,str("face"),(int(mask_box[0]+2), int(mask_box[1]-2)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1,cv2.LINE_AA)

        if SocialDistance:  # social distance
            SDimage, birdSDimage = Apply_ellipticBound(centroid_dict, DTCShow, calibration, redZone, greenZone, yellowZone, final_redZone, coupleZone, couples, CircleradiusForIndivsual, CircleradiusForCouples)
     
        return SDimage, DTCShow, save_tag
        # return frame_resized,frame_resized,0

    def get_frame(self):
        success, image = self.cap.read()
        # 调用ai
        image, detect_img, save_tag = self.ai_func(image)

        if save_tag == 1:
            global img_id
            img_id += 1
            cv2.imwrite("./static/test_img/"+str(img_id%5)+".jpg",detect_img)
       
        # 在这像test_img save图片
        ret, jpeg = cv2.imencode('.jpg', image)
        # if self.start_time == 0:
        #     self.start_time = time.perf_counter()
        # self.end_time = time.perf_counter()
        # print(
        #     f'--- {self.count} --- width: {image.shape[1]}, height: {image.shape[0]}, FPS: {self.count/(self.end_time-self.start_time):.2f}')

        return jpeg.tobytes()


app = Flask(__name__)

# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=0.1)


@app.route('/')
def index():

    global my_log
    context = {"my_log": my_log}

    return render_template('index.html', **context)


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera(TrtYOLOv6("./model/yolov6n.plan"))), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/click_dog_move', methods=["POST"])
def click_dog_move():
    # 实现机器狗的运动控制
    move_type = request.form.get("move_type")
    print(move_type)
    # btn_dog_top  walk  w
    # btn_dog_left left
    # btn_dog_bottom a
    # p_keyboard_move.stdin.write(f'{move2type[move_type]}\n'.encode('utf-8'))
    # p_keyboard_move.stdin.flush()
    # res = p_keyboard_move.stdout.readline().decode('utf-8')
    # print(res)
    # 执行命令行
    # dog_dir = "cd /home/xujing/catkinws/ && source devel/setup.bash"
    # os.system(dog_dir)

    # run_power = "roslaunch spot_micro_motion_cmd motion_cmd.launch debug:=true"
    # os.system(run_power)

    # run_move = "roslaunch spot_micro_keyboard_command keyboard_command.launch run_plot:=true"
    # os.system(run_move)

    # idle 机器狗趴下
    # stand 机器狗站立
    # angle_cmd 调整机器狗身体的方向角
    # walk 机器狗走路模式
    # w,s 控制机器狗前进的速度
    # a,d 控制左右两边腿的摆动速度
    # q,e 控制偏航率
    # u 退出行走模式到专利模式

    global my_log
    my_log = [{"type": "机器狗当前状态：站立", "time": "2022-11-01 11:25 33"}, {"type": "机器狗当前状态：前进", "time": "2022-11-01 11:25 30"},
              {"type": "机器狗当前状态：前进", "time": "2022-11-01 11:25 30"}, {
                  "type": "机器狗当前状态：前进", "time": "2022-11-01 11:25 30"},
              {"type": "机器狗当前状态：前进", "time": "2022-11-01 11:25 30"}]

    # # 只显示最新的5条记录日志
    # if len(my_log) >= 5:
    #     my_log = my_log[-4:]

    # # 获取当前时间
    # curr_time = datetime.datetime.now()
    # time_str = datetime.datetime.strftime(curr_time,'%Y-%m-%d %H:%M:%S')

    # my_log.append({"type":f"机器狗当前状态：{ move2type[move_type] }","time":time_str})

    return jsonify(my_log)



@app.route('/get_img', methods=["GET"])
def get_img():
    my_file = [{"file":"./static/test_img/"+file} for file in os.listdir("./static/test_img")]

    return jsonify(my_file)

    




if __name__ == '__main__':
    # vscode中使用Flask程序调试的时候不会执行main中的代码，此段代码用于部署运行
    app.run(debug=True, host='0.0.0.0', port=5000)
