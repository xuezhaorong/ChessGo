#基于FLANN的匹配器(FLANN based Matcher)定位图片
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from myutils import circle_tr, Rotate_image, resize, cv_show
chess_red = ('shuai', 'ju', 'pao', 'ma', 'bin', 'shi', 'xiang')
chess_black = ('jiang','ju','pao','ma','zu','shi','xiang')
templates = []
chess = []
chess.append(chess_red)
chess.append(chess_black)

# 读取训练数据路径


def read_path(path_name):
    templates = []
    for dir_item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name, dir_item))

        image = cv2.imread(full_path)
        templates.append(image)
    return templates


templates.append(read_path('./Temple/red'))
templates.append(read_path('./Temple/black'))
MIN_MATCH_COUNT = 10 #设置最低特征点匹配数量为10
#创建设置FLANN匹配
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
# Initiate SIFT detector创建sift检测器
sift = cv2.xfeatures2d.SIFT_create()
def chess_recognize(target, color):
    # kernel = np.ones((1,1),np.uint8)
    target = resize(target, 30, 30)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    # target = circle_tr(target)
    # 二值化
    target = cv2.adaptiveThreshold(
        target, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 5)
    # target = cv2.dilate(target,kernel,iterations=1)
    _, des2 = sift.detectAndCompute(target, None)
    temp,indx = 1,0
    # cv_show('target',target)
    for num, template in enumerate(templates[color]):
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        # 二值化
        template = cv2.adaptiveThreshold(
            template, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 5)

        # find the keypoints and descriptors with SIFT
        _, des1 = sift.detectAndCompute(template,None)
        try:
            matches = flann.knnMatch(des1,des2,k=2)
        except:
            print('error')
            return
        # store all the good matches as per Lowe's ratio test.
        good = []
        #舍弃大于0.7的匹配
        for m,n in matches:
            if m.distance < 0.8*n.distance:
                good.append(m)
        if len(good) > temp:
            temp = len(good)
            indx = num
    return chess[color][indx]
        # if len(good)>MIN_MATCH_COUNT:
        #     # 获取关键点的坐标
        #     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        #     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        #     #计算变换矩阵和MASK
        #     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        #     matchesMask = mask.ravel().tolist()
        #     h,w = template.shape
        #     # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标
        #     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        #     dst = cv2.perspectiveTransform(pts,M)
        #     cv2.polylines(target,[np.int32(dst)],True,0,2, cv2.LINE_AA)
        # else:
        #     print( "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        #     matchesMask = None
        # draw_params = dict(matchColor=(0,255,0), 
        #                 singlePointColor=None,
        #                 matchesMask=matchesMask, 
        #                 flags=2)
        # result = cv2.drawMatches(template,kp1,target,kp2,good,None,**draw_params)
        # plt.imshow(result, 'gray')
        # plt.show()
