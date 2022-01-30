# coding = utf-8
'''

date :  2022-1-13
desc:   获取注视点G的坐标，调用IrisPostion函数

'''
import cv2
import dlib
import numpy as np
import os
import keyboard
from imutils import face_utils
import time
from IrisPostion import createEyeMask
from IrisPostion import findIris
from IrisPostion import findCentroid



def GetGPoint():
    pwd = os.getcwd()  # 获取当前路径
    model_path = os.path.join(pwd, 'model')  # 模型文件夹路径
    shape_detector_path = os.path.join(model_path, 'shape_predictor_68_face_landmarks.dat')  # 人脸特征点检测模型路径
    with open('Centroid.txt', 'r') as f1:
        GHVList  = f1.read()
    j = 0

    GHV = list()
    '''
    for i in range(0, len(GHVList)):
        GHVList[i] = GHVList[i].rstrip('\n')
        
    GHVList = str(GHVList)
    '''

    for i in range(0, len(GHVList)):
        if GHVList[i]==',':
            GHV.append(GHVList[j:i])
            j = i+1
        i += 1

    faceDetector = dlib.get_frontal_face_detector()  # 人脸检测器
    landmarkDetector = dlib.shape_predictor(shape_detector_path)  # 人脸特征点检测器

    # 对应特征点的序号
    RIGHT_EYE_START = 37 - 1
    RIGHT_EYE_END = 42 - 1
    LEFT_EYE_START = 43 - 1
    LEFT_EYE_END = 48 - 1

    a = input("是否颠倒摄像头方向,Y/N ：")
    cap = cv2.VideoCapture(0)
    gaze_x = 0
    gaze_y = 0
    while (1):
        ret, img = cap.read()  # 读取视频流的一帧
        new_img = np.zeros_like(img)
        h, w = img.shape[0], img.shape[1]
        for i in range(h):  # 上下翻转
            if a == "Y":
                new_img[i] = img[h - i - 1]
            else:
                new_img[i] = img[i]

        gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)  # 专程回读图像
        rects = faceDetector(new_img, 0)  # 人脸检测
        for rect in rects:
            # 遍历每一个人脸
            # print('-'*20)
            shape = landmarkDetector(gray, rect)  # 检测特征点
            points = face_utils.shape_to_np(shape)  # convert the facial(x,y)-coordinates to a Nump array
            # print(points)
            leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]  # 取出左眼对应的特征点
            rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]  # 取出右眼对应的特征点
            leftrightEye = points[39:42 + 1]  # 取内眼角对应的特征点坐标
            eyecenterX = round((leftrightEye[0][0] + leftrightEye[3][0]) / 2)  # 计算内眼角X轴中点
            eyecenterY = round((leftrightEye[0][1] + leftrightEye[3][1]) / 2)  # 计算内眼角Y轴中点

            shape_width = new_img.shape[1]


            leftEyeMask = createEyeMask(leftEye, new_img)
            rightEyeMask = createEyeMask(rightEye, new_img)
            # 设定阈值来找到虹膜
            leftIris = findIris(leftEyeMask, new_img, 40)
            rightIris = findIris(rightEyeMask, new_img, 50)

            # 寻找质心
            leftIrisCentroid = findCentroid(leftIris)
            rightIrisCentroid = findCentroid(rightIris)

            if (leftIrisCentroid[0] != 0) and (leftIrisCentroid[1] != 0):  # 如果无法取得当前帧左眼质心，则取前一帧质心
                oldleftIrisCentroid = leftIrisCentroid
            else:
                leftIrisCentroid = oldleftIrisCentroid
            if (rightIrisCentroid[0] != 0) and (rightIrisCentroid[1] != 0):  # 如果无法取得当前帧右眼质心，则取前一帧质心
                oldrightIrisCentroid = rightIrisCentroid
            else:
                rightIrisCentroid = oldrightIrisCentroid

            iriscentroidX = round(leftIrisCentroid[0] + rightIrisCentroid[0] / 2)  # 质心中心点x坐标
            iriscentroidY = round(leftIrisCentroid[1] + rightIrisCentroid[1] / 2)  # 质心中心点y坐标
            irisCentr = (iriscentroidX, iriscentroidY)

            # print(leftIrisCentroid[0])
            leftH = leftIrisCentroid[0] / leftrightEye[0][0]  # 左眼质心坐标x/左眼内眼角坐标x
            leftV = leftIrisCentroid[1] / leftrightEye[0][1]  # 左眼质心坐标y/左眼内眼角坐标y
            rightH = rightIrisCentroid[0] / leftrightEye[3][0]  # 右眼同上
            rightV = rightIrisCentroid[1] / leftrightEye[3][1]

            ratioH = (leftH + rightH) / 2
            ratioV = (leftV + rightV) / 2

            Gh = ( float(GHV[0])- float(ratioH)) * float(GHV[4])   #水平方向
            Gv = ( float(ratioV)- float(GHV[3])) * float(GHV[5])   #垂直方向
            if Gv > float(GHV[3]):                     #映射y坐标
                gaze_y = (Gv-float(GHV[3]))*1080
            if Gh < float(GHV[0]):                     #映射x坐标q
                gaze_x = (Gh-float(GHV[0]))*1920
            print('gaze_y')
            print(gaze_y)
            print('gaze_x')
            print(gaze_x)
            time.sleep(3)
            cv2.imshow("frame", new_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


GetGPoint()

if __name__ == '_main_':
    GetGPoint()