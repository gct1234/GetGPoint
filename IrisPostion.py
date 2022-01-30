'''

date :  2021-12-27
desc:   在视频中定位虹膜，实现跟踪虹膜，获取虹膜质心和内眼角坐标数据

'''
import cv2
import dlib
import numpy as np
import os
import keyboard
from imutils import face_utils
from scipy.spatial import distance
import time

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = distance.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio


    return ear

def createEyeMask(eyeLandmarks, im):
    # 创建眼睛蒙版
    leftEyePoints = eyeLandmarks
    eyeMask = np.zeros_like(im)
    cv2.fillConvexPoly(eyeMask, np.int32(leftEyePoints), (255, 255, 255))
    eyeMask = np.uint8(eyeMask)
    return eyeMask

def findIris(eyeMask, im, thresh):  #定位虹膜
    # 设定阈值来找到虹膜
    r = im[:,:,2]
    _, binaryIm = cv2.threshold(r, thresh, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
    morph = cv2.dilate(binaryIm, kernel, 1)
    morph = cv2.merge((morph, morph, morph))
    morph = morph.astype(float)/255
    eyeMask = eyeMask.astype(float)/255
    iris = cv2.multiply(eyeMask, morph)
    #print("定位虹膜")
    #print(iris)
    return iris

def findCentroid(iris):  #计算质心
    # 寻找质心
    M = cv2.moments(iris[:,:,0])
    try:
        cX = round(M["m10"] / M["m00"])
        cY = round(M["m01"] / M["m00"])

    except:
        cX = 0
        cY = 0

    centroid = (cX, cY)
    return centroid
def createIrisMask(iris, centroid):
    # 生成虹膜蒙版及其反蒙版
    cnts, _ = cv2.findContours(np.uint8(iris[:,:,0]), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    flag = 10000
    final_cnt = None
    for cnt in cnts:
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        distance = abs(centroid[0]-x)+abs(centroid[1]-y)
        if distance < flag :
            flag = distance
            final_cnt = cnt
        else:
            continue
    (x,y),radius = cv2.minEnclosingCircle(final_cnt)
    center = (int(x),int(y))
    radius = int(radius) - 2

    irisMask = np.zeros_like(iris)
    inverseIrisMask = np.ones_like(iris)*255
    cv2.circle(irisMask,center,radius,(255, 255, 255),-1)
    cv2.circle(inverseIrisMask,center,radius,(0, 0, 0),-1)
    irisMask = cv2.GaussianBlur(irisMask, (5,5), cv2.BORDER_DEFAULT)
    inverseIrisMask = cv2.GaussianBlur(inverseIrisMask, (5,5), cv2.BORDER_DEFAULT)

    return irisMask, inverseIrisMask
def changeEyeColor(im, irisMask, inverseIrisMask):
    # 改变眼睛的颜色，并合并到原始图像
    imCopy = cv2.applyColorMap(im, cv2.COLORMAP_TWILIGHT_SHIFTED)
    imCopy = imCopy.astype(float)/255
    irisMask = irisMask.astype(float)/255
    inverseIrisMask = inverseIrisMask.astype(float)/255
    im = im.astype(float)/255
    faceWithoutEye = cv2.multiply(inverseIrisMask, im)
    newIris = cv2.multiply(irisMask, imCopy)
    result = faceWithoutEye + newIris
    #print("changeeyecolor")
    #print(result)
    return result

def float642Uint8(im):
    im2Convert = im.astype(np.float64) / np.amax(im)
    im2Convert = 255 * im2Convert 
    convertedIm = im2Convert.astype(np.uint8)
    return convertedIm

def IrisPostion():
    pwd = os.getcwd() #获取当前路径
    model_path = os.path.join(pwd,'model') #模型文件夹路径
    shape_detector_path = os.path.join(model_path,'shape_predictor_68_face_landmarks.dat') #人脸特征点检测模型路径

    faceDetector = dlib.get_frontal_face_detector() #人脸检测器
    landmarkDetector = dlib.shape_predictor(shape_detector_path) #人脸特征点检测器
    #landmarks = face.getLandmarks(faceDetector, landmarkDetector, im,shape_width,shape_height,1)

    EYE_AR_THRESH = 0.3 #EAR阈值
    EYE_AR_CONSEC_FRAMES = 3 #当EAR小于阈值时，接连多少帧一定发生眨眼动作

    #对应特征点的序号
    RIGHT_EYE_START = 37 - 1
    RIGHT_EYE_END  = 42 - 1
    LEFT_EYE_START  = 43 - 1
    LEFT_EYE_END    = 48 - 1

    a = input("是否颠倒摄像头方向,Y/N ：")
    cap = cv2.VideoCapture(0)
    txt = open('Centroid.txt','a')
    global maxratioH
    global minratioH
    global maxratioV
    global minratioV
    maxratioH = 0.00
    minratioH = 10.00
    maxratioV = 0.00
    minratioV = 10.00
    while(1):
        ret,img = cap.read() #读取视频流的一帧
        new_img=np.zeros_like(img)
        h,w=img.shape[0],img.shape[1]
        for i in range(h): #上下翻转
            if a=="Y":
                new_img[i]=img[h-i-1]
            else:
                new_img[i]=img[i]
     
        gray = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY) #专程回读图像
        rects = faceDetector(new_img,0) #人脸检测
        for rect in rects:
            #遍历每一个人脸
            #print('-'*20)
            shape = landmarkDetector(gray,rect) #检测特征点
            points = face_utils.shape_to_np(shape)  # convert the facial(x,y)-coordinates to a Nump array
            #print(points)
            leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]  # 取出左眼对应的特征点
            rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]  # 取出右眼对应的特征点
            leftrightEye = points[39:42+1]   #取内眼角对应的特征点坐标
            eyecenterX = round((leftrightEye[0][0]+ leftrightEye[3][0])/2)  #计算内眼角X轴中点
            eyecenterY = round((leftrightEye[0][1]+ leftrightEye[3][1])/2)  #计算内眼角Y轴中点

            #eyeCenter = (eyecenterX,eyecenterY)   #内眼角中点坐标赋值



            leftEAR = eye_aspect_ratio(leftEye)  # 计算左眼EAR
            rightEAR = eye_aspect_ratio(rightEye)  # 计算右眼EAR

            # print('leftEAR = {0}'.format(leftEAR))
            # print('rightEAR = {0}'.format(rightEAR))

            ear = (leftEAR + rightEAR) / 2.0  # 求左右眼的均值

            leftEyeHull = cv2.convexHull(leftEye)  # 寻求左眼轮廓
            rightEyeHull = cv2.convexHull(rightEye)  # 寻求右眼轮廓
            cv2.drawContours(new_img, [leftEyeHull], -1, (0, 255, 0), 1)  # 绘制左眼轮廓
            cv2.drawContours(new_img, [rightEyeHull], -1, (0, 255, 0), 1)  # 绘制右眼轮廓

            shape_width = new_img.shape[1]

            #landmarks = face.getLandmarks(faceDetector, landmarkDetector, new_img,w,h,1)
            #landmarks = face_utils.shape_to_np(shape) #convert the facial(x,y)-coordinates to a Nump array
            # 创建眼睛蒙版
         
            #leftEyeMask = createEyeMask(landmarks[36:41], new_img)
            #rightEyeMask = createEyeMask(landmarks[42:47], new_img)

            leftEyeMask = createEyeMask(leftEye, new_img)
            rightEyeMask = createEyeMask(rightEye, new_img)
            # 设定阈值来找到虹膜
            leftIris = findIris(leftEyeMask, new_img, 40)
            rightIris = findIris(rightEyeMask, new_img, 50)

            # 寻找质心
            leftIrisCentroid = findCentroid(leftIris)
            rightIrisCentroid = findCentroid(rightIris)

            if  (leftIrisCentroid[0]!=0) and  (leftIrisCentroid[1]!=0):  #如果无法取得当前帧左眼质心，则取前一帧质心
                oldleftIrisCentroid = leftIrisCentroid
            else:
                leftIrisCentroid = oldleftIrisCentroid
            if  (rightIrisCentroid[0]!=0) and  (rightIrisCentroid[1]!=0): #如果无法取得当前帧右眼质心，则取前一帧质心
                oldrightIrisCentroid = rightIrisCentroid
            else:
                rightIrisCentroid = oldrightIrisCentroid

            iriscentroidX = round(leftIrisCentroid[0] + rightIrisCentroid[0]/2)   #质心中心点x坐标
            iriscentroidY = round(leftIrisCentroid[1] + rightIrisCentroid[1]/2)   #质心中心点y坐标
            irisCentr = (iriscentroidX,iriscentroidY)


            #print(leftIrisCentroid[0])
            leftV = leftIrisCentroid[0] / leftrightEye[0][0]   #左眼质心坐标x/左眼内眼角坐标x
            leftH = leftIrisCentroid[1] / leftrightEye[0][1]   #左眼质心坐标y/左眼内眼角坐标y
            rightV = rightIrisCentroid[0] / leftrightEye[3][0] #右眼同上
            rightH = rightIrisCentroid[1] / leftrightEye[3][1]


            ratioH = (leftH + rightH)/2  
            ratioV = (leftV + rightV)/2

            if maxratioH < ratioH:     #获取rationH最大值
                maxratioH = ratioH
            if maxratioV < ratioV:     #获取rationV最大值
                maxratioV = ratioV

            if minratioH > ratioH:     #获取rationH最小值
                minratioH = ratioH
            if minratioV >ratioV:     #获取rationV最小值
                minratioV = ratioV

            if keyboard.is_pressed('q'):
                RH = (maxratioH-minratioH)/1920   #计算水平方向上眼球坐标改变量与屏幕视点相对于屏幕中心该变量的比值
                RV = (maxratioV-minratioV)/1080   #计算垂直水平方向上眼球坐标改变量与屏幕视点相对于屏幕中心该变量的比值

                #HV = (maxratioH, maxratioV, minratioH, minratioV, RH, RV)
                # V = (maxrationV,minrationV,RV)
                txt.write('{0},'.format(maxratioH))
                txt.write('{0},'.format(maxratioV))
                txt.write('{0},'.format(minratioH))
                txt.write('{0},'.format(minratioV))
                txt.write('{0},'.format(RH))
                txt.write('{0},'.format(RV))

                txt.write('\n')
                '''
                txt.write('{0},{1},{2}'.format(str(maxratioH),str(minratioH),str(RH)))
                txt.write('\n')
                txt.write('{0},{1},{2}'.format(str(maxratioV), str(minratioV), str(RV)))
                txt.write('\n')
                txt.write('\n')
                '''
                #txt.write('{0},{1},{2},{3}'.format(str(eyecenterX),str(eyecenterY),str(iriscentroidX),str(iriscentroidY)))
                #txt.write('\n')

                #txt.write('{0},{1},{2}'.format(str(leftV),str(rightV),str(ratioV)))
                #txt.write('\n')

            # 生成虹膜蒙版及其反蒙版
            #leftIrisMask, leftInverseIrisMask = createIrisMask(leftIris, leftIrisCentroid)
            #rightIrisMask, rightInverseIrisMask = createIrisMask(rightIris, rightIrisCentroid)

            # 改变眼睛的颜色，并合并到原始图像
            #coloredEyesLady = changeEyeColor(new_img, rightIrisMask, rightInverseIrisMask)
            #coloredEyesLady = float642Uint8(coloredEyesLady)
            #qqcoloredEyesLady = changeEyeColor(coloredEyesLady, leftIrisMask, leftInverseIrisMask)
        cv2.imshow("frame",new_img)
    
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    


if __name__ == '_main_' :

    IrisPostion()