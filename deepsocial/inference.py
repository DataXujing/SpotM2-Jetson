
from base64 import b64decode, b64encode
import os
import cv2
import numpy as np
import PIL
import io
import time
import matplotlib.pyplot as plt

from deepsocial import *
from sort import *

import tensorrt as trt
from cuda import cudart
from yolov6trt import *

# pip3 install filterpy

class_names = ["face","mask","person"]

# class_names = ["person","cat","dog","horse"]
# 反序列化模型
engine_file = "./model/yolov6n.plan"
# engine_file = "./model/yolov6.engine"

with open(engine_file, 'rb') as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

context = engine.create_execution_context()


#---实例化sort类，目标跟踪所用
mot_tracker = Sort(max_age=25, min_hits=4, iou_threshold=0.3)

# --deepsocial params
######################## Frame number
StartFrom  = 0 
EndAt      = 500                       #-1 for the end of the video
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



Input            =   "OxfordTownCentreDataset.mp4" #"./test.mp4"
ReductionFactor  = 2
calibration      = [[180,162],[618,0],[552,540],[682,464]]

# Output Video's path
Path_For_DTC = os.getcwd() + "/DeepSOCIAL DTC.avi"
Path_For_SocialDistance = os.getcwd() + "/DeepSOCIAL Social Distancing.avi"
Path_For_CrowdMap = os.getcwd() + "/DeepSOCIAL Crowd Map.avi"

# detect res
def extract_humans(boxs,scores,labels):
    detetcted = []
    maskd = []
    if len(boxs) > 0: # At least 1 detection in the image and check detection presence in a frame  
        idList = []
        id = 0
        for i,bbox in enumerate(boxs):
            confidence = scores[i]
            label = labels[i]
            xmin, ymin, xmax, ymax = bbox
            if class_names[int(label)] == 'person': 
                id +=1
                if id not in idList: idList.append(id)
                detetcted.append([int(xmin), int(ymin), int(xmax), int(ymax), idList[-1]])
            if class_names[int(label)] == 'mask':
                maskd.append([xmin, ymin, xmax, ymax,confidence,label])
    return np.array(detetcted), maskd


# 
def centroid(detections, image, calibration, _centroid_dict, CorrectionShift, HumanHeightLimit):
    e = birds_eye(image.copy(), calibration)
    centroid_dict = dict()
    now_present = list()
    if len(detections) > 0:   
        for d in detections:
            p = int(d[4])
            now_present.append(p)
            xmin, ymin, xmax, ymax = d[0], d[1], d[2], d[3]
            w = xmax - xmin
            h = ymax - ymin
            x = xmin + w/2
            y = ymax - h/2

            if h < HumanHeightLimit:
                overley = e.image
                bird_x, bird_y = e.projection_on_bird((x, ymax))
                if CorrectionShift:
                    if checkupArea(overley, 1, 0.25, (x, ymin)):
                        continue
                e.setImage(overley)
                center_bird_x, center_bird_y = e.projection_on_bird((x, ymin))
                centroid_dict[p] = (
                            int(bird_x), int(bird_y),
                            int(x), int(ymax), 
                            int(xmin), int(ymin), int(xmax), int(ymax),
                            int(center_bird_x), int(center_bird_y))

                _centroid_dict[p] = centroid_dict[p]
    return _centroid_dict, centroid_dict, e.image



def ColorGenerator(seed=1, size=10):
    np.random.seed = seed
    color=dict()
    for i in range(size):
        h = int(np.random.uniform() *255)
        color[i]= h
    return color


def VisualiseResult(_Map, e):
    Map = np.uint8(_Map)
    histMap = e.convrt2Image(Map)
    visualBird = cv2.applyColorMap(np.uint8(_Map), cv2.COLORMAP_JET)
    visualMap = e.convrt2Image(visualBird)
    visualShow = cv2.addWeighted(e.original, 0.7, visualMap, 1 - 0.7, 0)
    return visualShow, visualBird, histMap



# main
cap = cv2.VideoCapture(Input)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
height, width = frame_height // ReductionFactor, frame_width // ReductionFactor
print("Video Reolution: ",(width, height))

# 保存识别结果
if DTC: DTCVid = cv2.VideoWriter(Path_For_DTC, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (width, height))
if SocialDistance: SDimageVid = cv2.VideoWriter(Path_For_SocialDistance, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (width, height))
if CrowdMap: CrowdVid = cv2.VideoWriter(Path_For_CrowdMap, cv2.VideoWriter_fourcc(*"XVID"), 30.0, (width, height))

colorPool = ColorGenerator(size = 3000)
_centroid_dict = dict()
_numberOFpeople = list()
_greenZone = list()
_redZone = list()
_yellowZone = list()
_final_redZone = list()
_relation = dict()
_couples = dict()
_trackMap = np.zeros((height, width, 3), dtype=np.uint8)
_crowdMap = np.zeros((height, width), dtype=np.int) 
_allPeople = 0
_counter = 1
frame = 0

while True:
    print('-- Frame : {}'.format(frame))
    prev_time = time.time()
    ret, frame_read = cap.read()
    if not ret: break

    frame += 1
    if frame <= StartFrom: continue
    # if frame != -1:
    #     if frame > EndAt: break
        
    frame_resized = cv2.resize(frame_read,(width, height), interpolation=cv2.INTER_LINEAR)
    image = frame_resized
    e = birds_eye(image, calibration)
    
    # ---------------detect---------------------
    # detections, _, _ = darknet_helper(image, width, height)

    pred_box, pred_score, pred_label = get_detect(context,engine,image)

    # 解析detect
    humans,masks = extract_humans(pred_box, pred_score, pred_label)
    # 跟踪算法
    track_bbs_ids = mot_tracker.update(humans) if len(humans) != 0 else humans

    _centroid_dict, centroid_dict, partImage = centroid(track_bbs_ids, image, calibration, _centroid_dict, CorrectionShift, HumanHeightLimit)
    redZone, greenZone = find_zone(centroid_dict, _greenZone, _redZone, criteria=ViolationDistForIndivisuals)
    
    if CouplesDetection:
        _relation, relation = find_relation(e, centroid_dict, MembershipDistForCouples, redZone, _couples, _relation)
        _couples, couples, coupleZone = find_couples(image, _centroid_dict, relation, MembershipTimeForCouples, _couples)
        yellowZone, final_redZone, redGroups = find_redGroups(image, centroid_dict, calibration, ViolationDistForCouples, redZone, coupleZone, couples , _yellowZone, _final_redZone)
    else:
        couples = []
        coupleZone = []
        yellowZone = []
        redGroups = redZone
        final_redZone = redZone


    if DTC:
        DTC_image = image.copy()
        _trackMap = Apply_trackmap(centroid_dict, _trackMap, colorPool, 3)
        DTC_image = cv2.add(e.convrt2Image(_trackMap), image) 
        DTCShow = DTC_image
        for id, box in centroid_dict.items():
            center_bird = box[0], box[1]
            if not id in coupleZone:
                cv2.rectangle(DTCShow,(box[4], box[5]),(box[6], box[7]),(0,255,0),2)
                cv2.rectangle(DTCShow,(box[4], box[5]-13),(box[4]+len(str(id))*10, box[5]),(0,200,255),-1)
                cv2.putText(DTCShow,str(id),(box[4]+2, box[5]-2),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1,cv2.LINE_AA)
        # 把mask也画出来
        
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
        DTCVid.write(DTCShow)


    if SocialDistance:
  

        SDimage, birdSDimage = Apply_ellipticBound(centroid_dict, image, calibration, redZone, greenZone, yellowZone, final_redZone, coupleZone, couples, CircleradiusForIndivsual, CircleradiusForCouples)
        SDimageVid.write(SDimage)
  


    if CrowdMap:
        _crowdMap, crowdMap = Apply_crowdMap(centroid_dict, image, _crowdMap)
        crowd = (crowdMap - crowdMap.min()) / (crowdMap.max() - crowdMap.min())*255
        crowd_visualShow, crowd_visualBird, crowd_histMap = VisualiseResult(crowd, e)
        CrowdVid.write(crowd_visualShow)
     


    cv2.waitKey(3)
print('::: Analysis Completed')

cap.release()
if DTC: DTCVid.release(); print("::: Video Write Completed : ", Path_For_DTC)
if SocialDistance: SDimageVid.release() ; print("::: Video Write Completed : ", Path_For_SocialDistance)
if CrowdMap: CrowdVid.release() ; print("::: Video Write Completed : ", Path_For_CrowdMap)












