from PyQt6.QtWidgets import QMainWindow,QApplication
from Ui_MainWindows import Ui_MainWindow
from OpenCV import roipreprocess, ChessRecognize, recognizeThread
import numpy as np
import sys
import cv2
import Global as glo



class main(Ui_MainWindow,QMainWindow):
    def __init__(self):
        super().__init__()

        self.setupUi(self)

        # 设置全局变量
        main.setGlobalValue()

        self.screentCnt = np.zeros((4,2))

        self.roiButton.clicked.connect(
            self.setRoi
        )
        
        self.cameraButton.clicked.connect(
            self.cameraThread
        )
        
        self.ChessButton.clicked.connect(
            self.setRecognize
        )

        self.cutButton.clicked.connect(
            self.setCutScreen
        )
    # 主线程
    def cameraThread(self):
        recognizeThread(None)
        # camera = cv2.VideoCapture(0)
        # if not camera.isOpened():
        #     print("打开摄像头失败")
        #     return

        # cv2.namedWindow('chessboard', cv2.WINDOW_AUTOSIZE)

        # while True:
        #     ret, frame = camera.read()
        #     if not ret:
        #         print("未读取到图片")
        #         break

        #     # 图像处理部分
        #     # roi矫正
        #     if(glo.getValue('roiFlag')):
        #         screenCnt = roipreprocess(frame)
        #         glo.setValue('screenCnt',screenCnt)

        #     # if glo.getValue('screenCnt') == None and glo.getValue('recognizeFlag'):
        #     #     print("请先进行roi校正")            
        #     # # 识别
        #     # elif(glo.getValue('recognizeFlag')):
        #     #     ChessRecognize(glo.getValue('screenCnt'),frame)

        #     if(glo.getValue('recognizeFlag')):
        #         ChessRecognize(glo.getValue('screemCmt'),frame)

        #     # 截图
        #     if(glo.getValue('cutFlag')):
        #         cv2.imwrite('./1.png',frame)
        #         self.setCutScreen()

        #     cv2.imshow('chessboard', frame)

        #     key = cv2.waitKey(5)

        #     if key & 0xFF == ord('q'):
        #         break
        # camera.release()
        # cv2.destroyAllWindows()


    def setRoi(self):
        Flag = glo.getValue('roiFlag')
        glo.setValue('roiFlag',not Flag)
        
    def setRecognize(self):
        Flag = glo.getValue('recognizeFlag')
        glo.setValue('recognizeFlag',not Flag)

    def setCutScreen(self):
        Flag = glo.getValue('cutFlag')
        glo.setValue('cutFlag', not Flag)

    # 全局变量设置
    @staticmethod
    def setGlobalValue():
        glo._init()
        # roi区域标志
        glo.setValue("roiFlag",False)
        # 识别标志
        glo.setValue("recognizeFlag",False)
        # 截图标志
        glo.setValue("curFlag",False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = main()
    main.show()
    sys.exit(app.exec())
