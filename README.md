<div align='center'>
    <img src="./images/logo.png">
</div>



## NVIDIA Jetson Edge AI开发者大赛

### 智能防疫四足机器狗-SpotM2

**美迪康 AI Lab 成员：徐静，邵学军**

### 1.大赛说明

+ 请在12月14日23:59分之前提交参赛作品，作品包括：

1. 项目报告书；
2. DEMO视频（视频比例16：9，高清，视频长度至少2分钟）
3. 发送邮箱：sisiy.gu@gpus.cn, 邮件名标注：Jetson开发大赛

+ 【讲座回访】Jetson Edge AI 开发大赛专属讲

链接：https://pan.baidu.com/s/1a4PqmGWvVwGl8aD7-Hejiw?pwd=NVDA 
提取码：NVDA 

+ 大赛流程

报名初选：即日起-2022年11月20日

初选结果公告：2022年11月21日

创作阶段：2022年11月21日-2022年12月14日

评审阶段：2022年12月15日-2022年12月20日

大赛结果公告：2022年12月21日

具体信息请见：https://jinshuju.net/f/stGD9D

+ Jetson项目参考:
  https://developer.nvidia.com/embedded/community/jetson-projects



### 1.代码结构说明

```shell
./
├─SpotM2教程.pdf  # SpotM2详细教程
├─servo_calibration.py  # 舵机校准程序
├─servo_calibration_spreadsheet.ods #舵机PWM配置计算
├─SpotM2硬件购买清单.docx  #SpotM2硬件清单
├─3D打印件 #SpotM2 3D打印STL文件
│  ├─各打1份
│  └─各打2份
├─deepsocial  #Deepsocial算法
├─DogServer3.0 #SpotM2 Flask服务  
├─spotMicro-ROS-Melodic-Jetson-Nano #spotMicro Jetson Nano版运动控制 ROS程序源码
│  ├─assets
│  ├─docs
│  ├─lcd_monitor
│  │  ├─launch
│  │  ├─scripts
│  │  └─src
│  │      └─lcd_monitor
│  ├─ros-i2cpwmboard
│  ├─servo_move_keyboard
│  │  ├─launch
│  │  └─scripts
│  ├─spot_micro_joy
│  │  ├─launch
│  │  └─scripts
│  ├─spot_micro_keyboard_command
│  │  ├─launch
│  │  └─scripts
│  ├─spot_micro_launch
│  │  └─launch
│  ├─spot_micro_motion_cmd
│  │  ├─config
│  │  ├─data
│  │  ├─include
│  │  │  └─spot_micro_motion_cmd
│  │  ├─launch
│  │  ├─libs
│  │  │  └─spot_micro_kinematics_cpp
│  │  └─src
│  │      ├─rate_limited_first_order_filter
│  │      └─smfsm
│  ├─spot_micro_plot
│  │  ├─launch
│  │  └─scripts
│  └─spot_micro_rviz
│      ├─launch
│      ├─rviz
│      └─urdf
│          └─stl
└─YOLOv6-mask  #YOLOv6行人和口罩检测

# 每个文件夹有详细的代码供参考！

```



### 2.SpotM2教程

详细的SpotM2教程参考：[https://github.com/DataXujing/SpotM2-Jetson/blob/main/SpotM2%E6%95%99%E7%A8%8B.pdf](https://github.com/DataXujing/SpotM2-Jetson/blob/main/SpotM2%E6%95%99%E7%A8%8B.pdf)

<iframe src="https://github.com/DataXujing/SpotM2-Jetson/blob/main/SpotM2%E6%95%99%E7%A8%8B.pdf" style="width:800px; height:500px;" frameborder="0"></iframe>

```text

```

<embed id="pdfPlayer" src="./SpotM2教程.pdf" type="application/pdf" width="100%" height="600" >



### 3.SpotM2 Demo

