from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
import cv2, os, time
from threading import Thread
import threading
import tempfile
from pathlib import Path
from PyQt5.QtWidgets import QProgressDialog
import numpy as np
import deep_sort.deep_sort.deep_sort as ds
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtWidgets import QLabel,QFileDialog
# 不然每次YOLO处理都会输出调试信息
os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO 
def putTextWithBackground(img, text, origin, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, text_color=(255, 255, 255), bg_color=(0, 0, 0), thickness=1):
    """绘制带有背景的文本。

    :param img: 输入图像。
    :param text: 要绘制的文本。
    :param origin: 文本的左上角坐标。
    :param font: 字体类型。
    :param font_scale: 字体大小。
    :param text_color: 文本的颜色。
    :param bg_color: 背景的颜色。
    :param thickness: 文本的线条厚度。
    """
    # 计算文本的尺寸
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # 绘制背景矩形
    bottom_left = origin
    top_right = (origin[0] + text_width, origin[1] - text_height - 5)  # 减去5以留出一些边距
    cv2.rectangle(img, bottom_left, top_right, bg_color, -1)

    # 在矩形上绘制文本
    text_origin = (origin[0], origin[1] - 5)  # 从左上角的位置减去5来留出一些边距
    cv2.putText(img, text, text_origin, font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)
    
def extract_detections(results, detect_class):
    """
    从模型结果中提取和处理检测信息。
    - results: YoloV8模型预测结果,包含检测到的物体的位置、类别和置信度等信息。
    - detect_class: 需要提取的目标类别的索引。
    参考: https://docs.ultralytics.com/modes/predict/#working-with-results
    """
    
    # 初始化一个空的二维numpy数组，用于存放检测到的目标的位置信息
    # 如果视频中没有需要提取的目标类别，如果不初始化，会导致tracker报错
    detections = np.empty((0, 4)) 
    
    confarray = [] # 初始化一个空列表，用于存放检测到的目标的置信度。
    
    # 遍历检测结果
    # 参考：https://docs.ultralytics.com/modes/predict/#working-with-results
    for r in results:
        for box in r.boxes:
            # 如果检测到的目标类别与指定的目标类别相匹配，提取目标的位置信息和置信度
            if box.cls[0].int() == detect_class:
                x1, y1, x2, y2 = box.xywh[0].int().tolist() # 提取目标的位置信息，并从tensor转换为整数列表。
                conf = round(box.conf[0].item(), 2) # 提取目标的置信度，从tensor中取出浮点数结果，并四舍五入到小数点后两位。
                detections = np.vstack((detections, np.array([x1, y1, x2, y2]))) # 将目标的位置信息添加到detections数组中。
                confarray.append(conf) # 将目标的置信度添加到confarray列表中。
    return detections, confarray # 返回提取出的位置信息和置信度。

# 视频处理
def detect_and_track(input_path: str, output_path: str, detect_class: int, model, tracker) -> Path:
    """
    处理视频，检测并跟踪目标。
    - input_path: 输入视频文件的路径。
    - output_path: 处理后视频保存的路径。
    - detect_class: 需要检测和跟踪的目标类别的索引。
    - model: 用于目标检测的模型。
    - tracker: 用于目标跟踪的模型。
    """
    cap = cv2.VideoCapture(input_path)  # 使用OpenCV打开视频文件。
    if not cap.isOpened():  # 检查视频文件是否成功打开。
        print(f"Error opening video file {input_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) # 获取视频的分辨率（宽度和高度）。
    output_video_path = Path(output_path) / "output.avi" # 设置输出视频的保存路径。

    # 设置视频编码格式为XVID格式的avi文件
    # 如果需要使用h264编码或者需要保存为其他格式，可能需要下载openh264-1.8.0
    # 下载地址：https://github.com/cisco/openh264/releases/tag/v1.8.0
    # 下载完成后将dll文件放在当前文件夹内
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_video = cv2.VideoWriter(output_video_path.as_posix(), fourcc, fps, size, isColor=True) # 创建一个VideoWriter对象用于写视频。

    # 对每一帧图片进行读取和处理
    while True:
        success, frame = cap.read() # 逐帧读取视频。
        
        # 如果读取失败（或者视频已处理完毕），则跳出循环。
        if not (success):
            break

        # 使用YoloV8模型对当前帧进行目标检测。
        results = model(frame, stream=True)

        # 从预测结果中提取检测信息。
        detections, confarray = extract_detections(results, detect_class)

        # 使用deepsort模型对检测到的目标进行跟踪。
        resultsTracker = tracker.update(detections, confarray, frame)
        
        for x1, y1, x2, y2, Id in resultsTracker:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2]) # 将位置信息转换为整数。

            # 绘制bounding box和文本
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            putTextWithBackground(frame, str(int(Id)), (max(-10, x1), max(40, y1)), font_scale=1.5, text_color=(255, 255, 255), bg_color=(255, 0, 255))

        output_video.write(frame)  # 将处理后的帧写入到输出视频文件中。
            
    output_video.release()  # 释放VideoWriter对象。
    cap.release()  # 释放视频文件。
    
    print(f'output dir is: {output_video_path}')
    return output_video_path
class MWindow(QtWidgets.QMainWindow):

    def __init__(self):

        super().__init__()

        # 设置界面
        self.setupUI()

        self.videoBtn.clicked.connect(self.choose_video)
        self.startBtn.clicked.connect(self.play_video)

        # 定义定时器，用于控制显示视频的帧率
        # self.timer_camera = QtCore.QTimer()
        # 定时到了，回调 self.show_camera
        # self.timer_camera.timeout.connect(self.show_camera)


        # 启动处理视频帧独立线程
        # Thread(target=self.frameAnalyzeThreadFunc,daemon=True).start()
                # 创建进度对话框


    def setupUI(self):
        self.resize(1200, 800)
        self.setWindowTitle('基于深度学习的建筑工人视频跟踪技术')
        self.setStyleSheet("QMainWindow { font-size: 16pt; }")  # 设置标题字体大小

        # central Widget
        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)

        # central Widget 里面的 主 layout
        mainLayout = QtWidgets.QVBoxLayout(centralWidget)

        # 界面的上半部分 : 图形展示部分
        topLayout = QtWidgets.QHBoxLayout()
        
        # 视频原始
        videoLabelLayout1 = QtWidgets.QVBoxLayout()
        self.label_ori_video = QtWidgets.QLabel(self)
        self.label_ori_video.setMinimumSize(520, 400)
        self.label_ori_video.setStyleSheet('border: 1px solid #D7E2F9;')
        label_ori_video_title = QtWidgets.QLabel("追踪原视频")
        label_ori_video_title.setAlignment(Qt.AlignCenter)
        label_ori_video_title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        videoLabelLayout1.addWidget(self.label_ori_video)
        videoLabelLayout1.addWidget(label_ori_video_title)
        topLayout.addLayout(videoLabelLayout1)

        # 视频处理后
        videoLabelLayout2 = QtWidgets.QVBoxLayout()
        self.label_ori_video1 = QtWidgets.QLabel(self)
        self.label_ori_video1.setMinimumSize(520, 400)
        self.label_ori_video1.setStyleSheet('border: 1px solid #D7E2F9;')
        label_ori_video1_title = QtWidgets.QLabel("追踪后视频")
        label_ori_video1_title.setAlignment(Qt.AlignCenter)
        label_ori_video1_title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        videoLabelLayout2.addWidget(self.label_ori_video1)
        videoLabelLayout2.addWidget(label_ori_video1_title)
        topLayout.addLayout(videoLabelLayout2)
        
        mainLayout.addLayout(topLayout)

        # 界面下半部分： 输出框 和 按钮
        groupBox = QtWidgets.QGroupBox(self)
        bottomLayout = QtWidgets.QHBoxLayout(groupBox)
        self.textLog = QtWidgets.QTextBrowser()
        self.textLog.setStyleSheet("font-size: 18pt;")  # 设置文本框内字体大小
        bottomLayout.addWidget(self.textLog, 1)  # 改变文本框的比例因子

        mainLayout.addWidget(groupBox)

        btnLayout = QtWidgets.QVBoxLayout()
        self.videoBtn = QtWidgets.QPushButton('🎞️视频文件')
        self.startBtn = QtWidgets.QPushButton('▶开始播放')
        btnLayout.addWidget(self.videoBtn)
        btnLayout.addWidget(self.startBtn)
        bottomLayout.addLayout(btnLayout, 0)  # 减小按钮部分的比例因子
    # def play_video(self):
    #     cap = cv2.VideoCapture('output\output.avi')
    #     if not cap.isOpened():
    #         self.textLog.append("Error: Unable to open video.")
    #         return

    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break

    #         # 将OpenCV的图像格式转换为Qt支持的格式
    #         height, width, channel = frame.shape
    #         bytesPerLine = 3 * width
    #         qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
    #         pixmap = QPixmap.fromImage(qImg)

    #         # 在标签中显示视频帧
    #         self.label_ori_video.setPixmap(pixmap.scaled(self.label_ori_video.size(), Qt.KeepAspectRatio))

    #         # 等待一小段时间，以便可以观察视频帧
    #         cv2.waitKey(30)

    #     cap.release()
    def play_video(self):
        # 打开第一个视频
        cap1 = cv2.VideoCapture('output\output.avi')
        if not cap1.isOpened():
            self.textLog.append("Error: Unable to open first video.")
            return

        # 打开第二个视频
        cap2 = cv2.VideoCapture(self.video_file_path)
        if not cap2.isOpened():
            self.textLog.append("Error: Unable to open second video.")
            return

        while True:
            # 读取第一个视频的帧
            ret1, frame1 = cap1.read()
            if not ret1:
                break

            # 读取第二个视频的帧
            ret2, frame2 = cap2.read()
            if not ret2:
                break

            # 将第一个视频帧转换为Qt支持的格式并在label_ori_video1中显示
            height1, width1, channel1 = frame1.shape
            bytesPerLine1 = 3 * width1
            qImg1 = QImage(frame1.data, width1, height1, bytesPerLine1, QImage.Format_RGB888).rgbSwapped()
            pixmap1 = QPixmap.fromImage(qImg1)
            self.label_ori_video1.setPixmap(pixmap1.scaled(self.label_ori_video1.size(), Qt.KeepAspectRatio))

            # 将第二个视频帧转换为Qt支持的格式并在label_ori_video中显示
            height2, width2, channel2 = frame2.shape
            bytesPerLine2 = 3 * width2
            qImg2 = QImage(frame2.data, width2, height2, bytesPerLine2, QImage.Format_RGB888).rgbSwapped()
            pixmap2 = QPixmap.fromImage(qImg2)
            self.label_ori_video.setPixmap(pixmap2.scaled(self.label_ori_video.size(), Qt.KeepAspectRatio))

            # 等待一小段时间，以便可以观察视频帧
            cv2.waitKey(30)

        # 释放视频对象
        cap1.release()
        cap2.release()

    # def choose_video(self):
    #     # 打开文件对话框
    #     options = QFileDialog.Options()
    #     file_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi)", options=options)
    #     if file_path:
    #         # 如果用户选择了文件，将文件路径显示在输出框中
    #         self.textLog.append(f"选择的视频文件：{file_path}")
    #         self.video_file_path = file_path  # 将文件路径保存在实例变量中
    #         detect_and_track(self.video_file_path, output_path, detect_class, model, tracker)
    #         self.textLog.append("视频处理完成")
    def choose_video(self):
        # 打开文件对话框
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi)", options=options)
        if file_path:
            # 如果用户选择了文件，将文件路径显示在输出框中
            self.textLog.append(f"已选择的视频文件：{file_path}")
            self.video_file_path = file_path  # 将文件路径保存在实例变量中
            # 创建线程来执行 detect_and_track 函数
            thread = threading.Thread(target=self.execute_detect_and_track)
            thread.start()

    def execute_detect_and_track(self):
        # 执行 detect_and_track 函数
        detect_and_track(self.video_file_path, output_path, detect_class, model, tracker)
        # 处理完成后向self.textLog添加日志
        self.textLog.append("程序预处理完成")



if __name__ == "__main__":
        # 指定输入视频的路径。
    ######
    input_path = "test.mp4"
    ######

    # 输出文件夹，默认为系统的临时文件夹路径
    output_path = 'output'  # 创建一个临时目录用于存放输出视频。

    # 加载yoloV8模型权重
    model = YOLO("yolov8n.pt")

    # 设置需要检测和跟踪的目标类别
    # yoloV8官方模型的第一个类别为'person'
    detect_class = 0
    # print(f"detecting {model.names[detect_class]}") # model.names返回模型所支持的所有物体类别

    # 加载DeepSort模型
    tracker = ds.DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")

    app = QtWidgets.QApplication([])
    window = MWindow()
    window.show()
    app.exec()