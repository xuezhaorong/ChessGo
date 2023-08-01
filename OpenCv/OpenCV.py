import cv2
import numpy as np
from myutils import four_point_transform, Rotate_image, resize, cv_show, img_contrast_bright, min_max_filtering
import Global as glo
import matplotlib.pyplot as plt
# from template_match import chess_recognize
import multiprocessing
from FLANN_match import chess_recognize

def roi(image, blur, S):
    kernel = np.ones((4, 4), np.uint8)
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)
    # cv_show('blackhat', blackhat)
    # 二值化
    thresh = cv2.adaptiveThreshold(
        blackhat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5)
    # cv_show('thresh', thresh)
    # 轮廓检测
    coutrous, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 过滤掉最外框
    coutrous = sorted(coutrous, key=cv2.contourArea, reverse=True)[:5]
    # cv2.drawContours(image,coutrous,-1,(0,255,0),2)
    screenCnt = np.zeros((4, 2))
    for c in coutrous:
        epslion = 0.03 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epslion, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            s = w*h
            # print(S,s)
            if s > 0.4 * S and s < 0.95 * S:
                screenCnt[0] = np.array([x-10, y-10])
                screenCnt[1] = np.array([x-10, y+h+10])
                screenCnt[2] = np.array([x+w+10, y+h+10])
                screenCnt[3] = np.array([x+w+10, y-10])

                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

    return image, screenCnt


# roi处理
# def roipreprocess():
#     camera = cv2.VideoCapture(0)
#     if not camera.isOpened():
#         print("打开摄像头失败")
#         return

#     cv2.namedWindow('Roi', cv2.WINDOW_NORMAL)

#     while True:
#         ret, frame = camera.read()
#         if not ret:
#             print("未读取到图片")
#             break

#         # # 灰度图
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         # # 中值滤波
#         blur = cv2.medianBlur(gray, 3)

#         height, width = frame.shape[:2]

#         # # 取得roi
#         image, screenCnt = roi(frame, blur, width * height)

#         cv2.imshow('Roi', image)

#         key = cv2.waitKey(5)

#         if key & 0xFF == ord('q'):
#             break

#     camera.release()
#     cv2.destroyAllWindows()

#     return screenCnt

def roipreprocess(frame):
    # 灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # # 中值滤波
    blur = cv2.medianBlur(gray, 3)

    height, width = frame.shape[:2]

    # # 取得roi
    image, screenCnt = roi(frame, blur, width * height)

    #
    cv_show('roi', image)

    return screenCnt


# 图像识别线程
def recognize_thread(iq, oq):
    while True:
        if iq.empty():
            break
        (circle, color, x, y) = iq.get()
        chess_text = chess_recognize(circle, color)
        oq.put((chess_text, x, y))


# 滑动条回调函数↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓


def HoughCirclesCallbackFun(self):  # 霍夫圆检测阈值滑动条回调函数
    glo.setValue('houghCirclesDeteThreshold', cv2.getTrackbarPos(
        'HoughCircles', 'ChessBoard') + 1)  # 该值必须大于0


def ChessRecognize(screenCnt, frame):
    # 特征变换 ############################
    # warped = four_point_transform(frame,screenCnt)
    # glo._init()
    # glo.setValue('houghCirclesDeteThreshold', 100
        # 转换成灰度图
    blur = cv2.medianBlur(frame, 3)

    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    # 均衡化
    equ = cv2.equalizeHist(gray)
    # 二值化
    threshold = cv2.adaptiveThreshold(
        equ, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 5)

    cv_show('image',threshold)
    # 输入图像，方法（类型），dp(dp=1时表示霍夫空间与输入图像空间的大小一致，dp=2时霍夫空间是输入图像空间的一半，以此类推)，最短距离-可以分辨是两个圆否 则认为是同心圆 ,边缘检测时使用Canny算子的高阈值，越低对圆的要求越低，中心点累加器阈值—候选圆心（霍夫空间内累加和大于该阈值的点就对应于圆心），检测到圆的最小半径，检测到圆的的最大半径

    circles = cv2.HoughCircles(
        threshold, cv2.HOUGH_GRADIENT, 1, 30, param1=100, param2=20, minRadius=20, maxRadius=35)

    print(len(circles[0]))
    if circles is not None:
        for circle in circles[0]:
            x, y, r = map(int, circle)
            if r < 5:
                continue
            # 截取当个棋子进行识别
            circle = frame[y-(r-10):y+(r-10), x-(r-10):x+(r-10)]
            if 0 in circle.shape:
                print('error')
                continue

            # 颜色识别
            color = ColorClassify(circle)
            circle = resize(circle, 30, 30)
            # cv_show('circle',circle)

            if color == None:
                continue

            # 绘制圆形
            if color == 0:
                cv2.circle(frame, (x, y), r, (0, 0, 255), 3)
            else:
                cv2.circle(frame, (x, y), r, (0, 255, 0), 3)

            # 棋子识别
            chess_text = chess_recognize(circle, color)
            print(chess_text)

            cv2.putText(frame, chess_text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv_show('result', frame)


# 棋子颜色识别线程


def ColorClassify(image):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        hist_max = np.argmax(hist)
        # print(hist_max)
        if hist_max > 180:
            return None

    image = img_contrast_bright(image, 0.3, 0.3, 20)
    # 转换为HSV图
    imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 转换成灰度图
    gray = cv2.cvtColor(imgHSV, cv2.COLOR_BGR2GRAY)
    # 二值化
    _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
    # cv_show('image', thresh)
    # print(np.sum(thresh, axis=(0, 1)))
    if np.sum(thresh, axis=(0, 1)) > 20000:
        return 0
    else:
        return 1


def OpenCamera(name):
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("打开摄像头失败")
        return

    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)

    while True:
        ret, frame = camera.read()
        if not ret:
            print("未读取到图片")
            break

        cv2.imshow('frame', frame)

        key = cv2.waitKey(5)

        if key & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image = cv2.imread('./1.png')
    ChessRecognize(None, image)


# 棋子识别线程


def recognizeThread(screenCnt):

    glo._init()

    glo.setValue('houghCirclesDeteThreshold', 100)
    glo.setValue('recognizethreadFlag', False)

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("打开摄像头失败")
        return

    cv2.namedWindow('ChessBoard', cv2.WINDOW_NORMAL)

    cv2.createTrackbar('HoughCircles', 'ChessBoard', glo.getValue(
        'houghCirclesDeteThreshold'), 1000, HoughCirclesCallbackFun)  # 创建霍夫圆检测阈值滑动条

    # # 棋子几何
    # chess_dir = dict()

    # # 创建输入输出进程队列
    # # 存放输入图像
    # iq = multiprocessing.Queue()
    # oq = multiprocessing.Queue()
    while True:
        ret, frame = camera.read()
        if not ret:
            print("未读取到图片")
            break
        # 特征变换 ############################
        # warped = four_point_transform(frame,screenCnt)

        # 转换成灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 均衡化
        equ = cv2.equalizeHist(gray)
        # 二值化
        threshold = cv2.adaptiveThreshold(
            equ, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 5)
        blur = cv2.medianBlur(threshold, 3)

        # cv_show('blur',blur)
        # 输入图像，方法（类型），dp(dp=1时表示霍夫空间与输入图像空间的大小一致，dp=2时霍夫空间是输入图像空间的一半，以此类推)，最短距离-可以分辨是两个圆否 则认为是同心圆 ,边缘检测时使用Canny算子的高阈值，越低对圆的要求越低，中心点累加器阈值—候选圆心（霍夫空间内累加和大于该阈值的点就对应于圆心），检测到圆的最小半径，检测到圆的的最大半径

        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, 1, 30, param1=glo.getValue('houghCirclesDeteThreshold'), param2=20, minRadius=20, maxRadius=35)
        if circles is not None:
            for j, i in enumerate(circles[0]):  # 遍历矩阵的每一行的数据
                x, y, r = int(i[0]), int(i[1]), int(i[2])
                if r < 5:
                    continue
                # 截取单个棋子进行识别
                circle = frame[y-(r-7):y+(r-7), x-(r-7):x+(r-7)]
                if 0 in circle.shape:
                    print("error")
                    continue
                # 先识别是黑还是红
                color = ColorClassify(circle)
                circle = resize(circle, 30, 30)

                if color == None:
                    continue
                # 存入数据
                # iq.put((circle, color, x, y))
                # cv2.imwrite(f'circle{j}.png',circle)
                # 绘制圆形
                if color == 0:
                    cv2.circle(frame, (x, y), r, (0, 0, 255), 3)
                else:
                    cv2.circle(frame, (x, y), r, (0, 255, 0), 3)

                chess_text = chess_recognize(circle, color)
                cv2.putText(frame, chess_text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            # 多核
            # if oq.empty():
            #     p = multiprocessing.Process(
            #         target=recognize_thread,
            #         args=(iq, oq))
            #     p.daemon = True
            #     p.start()

            # while True:
            #     if oq.empty():
            #         break
            #     (chess_text, x, y) = oq.get()
            #     chess_dir[chess_text] = (x, y)
            #     p.terminate()
            #     p.join()

            # for k, v in chess_dir.items():
            #     chess_text = k
            #     (x, y) = v
            #     cv2.putText(frame, chess_text, (x, y),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.imshow('ChessBoard', frame)

        key = cv2.waitKey(5)

        if key & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
