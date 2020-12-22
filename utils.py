from PIL import Image
import numpy as np
import random
import copy
import os
import cv2
from keras.models import load_model
import pytesseract
import pandas as pd

random.seed(0)
class_colors = [[0,0,0],[255,0,0]]
NCLASSES = 2
HEIGHT = 416
WIDTH = 416

# 加载模型


def cut(img,boxes):
    """在图片中截图"""
    cut = img[int(boxes[1]):int(boxes[1] + boxes[3]),
          int(boxes[0]):int(boxes[0] + boxes[2])]
    # cv2.imwrite(path+'/{}.jpg'.format(name), cut)
    return cut

def segment_img(img, model):
    # 将opencv图像格式转换为PIL.Image的格式
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

    orininal_h = np.array(img).shape[0]
    orininal_w = np.array(img).shape[1]

    # 将图片resize成416*416
    img = img.resize((WIDTH,HEIGHT))
    img = np.array(img)

    # 归一化，然后reshape成1，416，416，3
    img = img/255
    img = img.reshape(-1,HEIGHT,WIDTH,3)

    # 进行一次正向传播，得到（43264，2)
    pr = model.predict(img)[0]
    # print(pr.shape)

    pr = pr.reshape((int(HEIGHT/2), int(WIDTH/2),NCLASSES)).argmax(axis=-1)

    seg_img = np.zeros((int(HEIGHT/2), int(WIDTH/2),3))
    colors = class_colors

    for c in range(NCLASSES):
        seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
        seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
        seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')

    seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w,orininal_h))
    # 将将PIL.Image图像格式转换为opencv的格式
    seg_img = cv2.cvtColor(np.asarray(seg_img),cv2.COLOR_RGB2BGR)
    return seg_img

# 计算掩膜的面积
def calculate_area(gray_img):
    area = 0
    w,h = gray_img.shape
    for i in range(w):
        for j in range(h):
            if gray_img[i][j] != 0:
                area += 1
    return area

# 识别图片中的数字，主要进行OCR,读取时间参数
def recognition_number(image):
    """识别数字"""
    try:
        text = pytesseract.image_to_string(image, lang="eng")
        return text
    except:
        return None


#主程序，目标跟踪和语义分割
def detect(input_video, output_video, interval, track_number, csv_path, mode_path, debuglog):
    """
    input_video:输入视频的路径
    output_video:输出结果视频的路径
    interval:检测是隔多少帧检测一次
    csv_path:输出csv文件的路径
    """
    # 提示，输入三个要检测的目标
    debuglog('Select {} tracking targets'.format(track_number))

    cv2.namedWindow("tracking")
    # 输入要检测视频的名称和路径
    camera = cv2.VideoCapture(input_video)
    tracker = cv2.MultiTracker_create()
    init_once = False
    model = load_model(mode_path)

    

    ok, image = camera.read()
    # 设置保存视频结果的信息
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video, fourcc, 20, (image.shape[1], image.shape[0]))

    if not ok:
        debuglog('Failed to read video')
        exit()


    # 设置跟踪目标的个数
    track_number = track_number
    # 把目标跟踪的框放入空列表
    bboxes = []    
    for t in range(track_number):
        bbox = cv2.selectROI('tracking', image) #第一个框用来检测时间参数
        bboxes.append(bbox)


    i = 1

    #保存气孔面积的列表
    aliens = []
    while camera.isOpened():
        ok, image = camera.read()
        if not ok:
            debuglog ('no image to read')
            camera.release()
            cv2.destroyAllWindows()
            break
        
        # 每隔指定的间隔帧数才进行语义分割
        if i%interval == 0:

            if not init_once:
                for bbox in bboxes:
                    ok = tracker.add(cv2.TrackerBoosting_create(), image, bbox)

                init_once = True

            ok, boxes = tracker.update(image)
            # print(ok, boxes)

            j = 0
            # 将一帧里的气孔面积保存在一个列表中
            stoma_area = []
            for newbox in boxes:

                #判断，如果是第一个框，则检测时间参数
                if j == 0:
                    p1 = (int(newbox[0]), int(newbox[1]))
                    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                    cut_img = cut(image, newbox)
                    time_text = recognition_number(cut_img)

                # 如果不是第一帧，则进行语义分割
                else:
                    p1 = (int(newbox[0]), int(newbox[1]))
                    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                    cut_img = cut(image, newbox)
                    # print(cut_img.shape)
                    seg_img = segment_img(cut_img, model)
                    # print(seg_img.shape)
                    # 将检测结果与原始图片混合相加
                    dst = cv2.addWeighted(cut_img,0.5,seg_img,0.5,0)
                    image[int(newbox[1]):int(newbox[1] + newbox[3]),
                    # 将混加的结果放到原图上去
                    int(newbox[0]):int(newbox[0] + newbox[2])] = dst
                    # 画矩形
                    cv2.rectangle(image, p1, p2, (200, 0, 0))
                    # 显示数字
                    # 将掩膜变成灰度图像
                    gray_img = cv2.cvtColor(seg_img,cv2.COLOR_BGR2GRAY)
                    area_number = calculate_area(gray_img)
                    stoma_area.append(area_number)
                    text = '{}'.format(area_number)
                    cv2.putText(image,text,(int(newbox[0]), int(newbox[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                j = j + 1


            # stoma_area变成字典
            dict_keys = []
            dict_values = stoma_area
            for s in range(len(dict_values)):
                # 生成keys的列表
                dict_keys.append('stomat{}'.format(s))
            
            # 生成气孔的字典
            new_dict1 = dict(zip(dict_keys,dict_values))
            # 生成序号和时间的字典
            new_dict2 = {'image_number':i, 'time_date':time_text}

            # 合并字典
            new_aliens = dict(new_dict1, **new_dict2)

            aliens.append(new_aliens)
            cv2.imshow('tracking', image)
            out.write(image)
            debuglog('这是第{}张图片！'.format(i))
        else:
            pass
        k = cv2.waitKey(1)
        if k == 27 & 0xFF == ord('q'):
            break  #按q退出
        i = i + 1
    out.release()

    # 将字典转化为pandas
    df = pd.DataFrame(aliens)
    #保存为excel表格
    df.to_csv(csv_path)