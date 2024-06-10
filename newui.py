from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtCore import Qt
import cv2, os, time,sys
from threading import Thread
import threading
import tempfile
from pathlib import Path
from PyQt5.QtWidgets import QProgressDialog
import numpy as np
import deep_sort.deep_sort.deep_sort as ds
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtWidgets import QLabel,QFileDialog
from PIL import Image
# 不然每次YOLO处理都会输出调试信息
os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO 
def putTextWithBackground(img, text, origin, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, text_color=(255, 255, 255), bg_color=(0, 0, 0), thickness=2):
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
        self.setupUI()
        self.videoBtn.clicked.connect(self.choose_video)
        self.startBtn.clicked.connect(self.play_video)
        self.stopBtn.clicked.connect(self.stop_video)
        self.pauseBtn.clicked.connect(self.pause_video)
        self.isPlaying = False  # 控制视频播放的状态

        # 在开始时输出日志信息
        self.textLog.append("基于深度学习的建筑工人视频跟踪技术")

        # 初始化时加载默认图片
        self.load_default_image()
    def load_default_image(self):
        self.default_img = QPixmap('R-C.jpg')
        self.set_default_image()
    def pause_video(self):
        # 切换播放状态
        self.isPlaying = not self.isPlaying
        if self.isPlaying:
            self.pauseBtn.setText('⏸️暂停播放')
            self.textLog.append("视频播放已继续")
        else:
            self.pauseBtn.setText('▶继续播放')
            self.textLog.append("视频播放已暂停")

    def set_default_image(self):
        scaled_img = self.default_img.scaled(640, 360, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.label_ori_video.setPixmap(scaled_img)
        self.label_process_video.setPixmap(scaled_img)
    def setupUI(self):
        self.resize(1200, 800)
        self.setWindowTitle('基于深度学习的建筑工人视频跟踪技术')  # 设置窗口的标题



        
        # 设置全局背景颜色，但排除了特定的组件
        self.setStyleSheet("QMainWindow { font-size: 16pt; background-color: #FF7F50; }"
                           "QPushButton { background-color: none; }"  # 保持按钮默认背景
                           "QLabel { background-color: none; }")  # 保持其他标签默认背景

        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)
        mainLayout = QtWidgets.QVBoxLayout(centralWidget)
        topLayout = QtWidgets.QHBoxLayout()
         # 创建并设置标题标签
        titleLabel = QtWidgets.QLabel("基于深度学习的建筑工人视频跟踪技术")
        titleLabel.setAlignment(Qt.AlignCenter)  # 标题居中
        titleLabel.setStyleSheet("font-size: 24pt; font-weight: bold;")  # 设置字体大小和加粗
        mainLayout.addWidget(titleLabel)  # 将标题标签添加到布局中
        videoLabelLayout1 = QtWidgets.QVBoxLayout()
        self.label_ori_video = QtWidgets.QLabel(self)
        self.label_ori_video.setMinimumSize(640, 360)
        # 为label_ori_video设置特定的背景颜色
        self.label_ori_video.setStyleSheet('border: 1px solid #D7E2F9; background-color: #FDF5E6;')  # 确保此标签有白色背景
        label_ori_video_title = QtWidgets.QLabel("追踪原视频")
        label_ori_video_title.setAlignment(Qt.AlignCenter)
        label_ori_video_title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        videoLabelLayout1.addWidget(self.label_ori_video)
        videoLabelLayout1.addWidget(label_ori_video_title)
        topLayout.addLayout(videoLabelLayout1)

        videoLabelLayout2 = QtWidgets.QVBoxLayout()
        self.label_process_video = QtWidgets.QLabel(self)
        self.label_process_video.setMinimumSize(640, 360)
        self.label_process_video.setStyleSheet('border: 1px solid #D7E2F9;background-color: #FDF5E6;')  # 维持此标签特定样式
        label_process_video_title = QtWidgets.QLabel("追踪后视频")
        label_process_video_title.setAlignment(Qt.AlignCenter)
        label_process_video_title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        videoLabelLayout2.addWidget(self.label_process_video)
        videoLabelLayout2.addWidget(label_process_video_title)
        topLayout.addLayout(videoLabelLayout2)
        
        mainLayout.addLayout(topLayout)

        groupBox = QtWidgets.QGroupBox(self)
        bottomLayout = QtWidgets.QHBoxLayout(groupBox)
        self.textLog = QtWidgets.QTextBrowser()
        self.textLog.setStyleSheet("font-size: 18pt;background-color:#F5DEB3")  # 文本框保持特定样式
        bottomLayout.addWidget(self.textLog, 1)

        mainLayout.addWidget(groupBox)

        btnLayout = QtWidgets.QVBoxLayout()
        self.videoBtn = QtWidgets.QPushButton('🎞️选择视频文件')
        self.startBtn = QtWidgets.QPushButton('▶开始播放')
        self.stopBtn = QtWidgets.QPushButton('⏹️停止播放')
        btnStyle = "QPushButton { font-size: 18pt; }"
        self.pauseBtn = QtWidgets.QPushButton('⏸️暂停播放')
        self.pauseBtn.setStyleSheet(btnStyle)
        self.videoBtn.setStyleSheet(btnStyle)
        self.startBtn.setStyleSheet(btnStyle)
        self.stopBtn.setStyleSheet(btnStyle)
        btnLayout.addWidget(self.videoBtn)
        btnLayout.addWidget(self.startBtn)
        btnLayout.addWidget(self.pauseBtn)
        btnLayout.addWidget(self.stopBtn)
        bottomLayout.addLayout(btnLayout, 0)


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
        self.isPlaying = True
        frame_count = 0
        self.cap1 = cv2.VideoCapture('output/output.avi')
        self.cap2 = cv2.VideoCapture(self.video_file_path)
        if not self.cap1.isOpened() or not self.cap2.isOpened():
            self.textLog.append("Error: Unable to open video.")
            return

        while True:  # 主循环
            if self.isPlaying:
                ret1, frame1 = self.cap1.read()
                ret2, frame2 = self.cap2.read()
                if not ret1 or not ret2:
                    break
                putTextWithBackground(frame1, f"Frame: {frame_count}", (10, 50), font_scale=2)
                putTextWithBackground(frame2, f"Frame: {frame_count}", (10, 50), font_scale=2)
                frame_count += 1
                self.update_video_display(frame1, self.label_process_video)
                self.update_video_display(frame2, self.label_ori_video)
                cv2.waitKey(30)
            else:
                cv2.waitKey(500)  # 短暂等待，减少CPU使用率

        self.cap1.release()
        self.cap2.release()
        self.set_default_image()

    def update_video_display(self, frame, label):
        if frame is not None:
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(qImg)
            label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio))
    # def play_video(self):
    #     self.isPlaying = True  # 开始播放视频时设置为True
    #     # 打开视频
    #     self.cap1 = cv2.VideoCapture('output/output.avi')
    #     self.cap2 = cv2.VideoCapture(self.video_file_path)
    #     if not self.cap1.isOpened() or not self.cap2.isOpened():
    #         self.textLog.append("Error: Unable to open video.")
    #         return

    #     while self.isPlaying:  # 使用 isPlaying 控制播放循环
    #         ret1, frame1 = self.cap1.read()
    #         ret2, frame2 = self.cap2.read()
    #         if not ret1 or not ret2:
    #             break

    #         # 更新UI中的视频显示
    #         self.update_video_display(frame1, self.label_process_video)
    #         self.update_video_display(frame2, self.label_ori_video)

    #         cv2.waitKey(30)

    #     self.cap1.release()
    #     self.cap2.release()
    #     self.set_default_image()

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

    def stop_video(self):
        # 设置为False以停止视频播放
        self.isPlaying = False  
        # 关闭视频文件
        if hasattr(self, 'cap1') and self.cap1.isOpened():
            self.cap1.release()
        if hasattr(self, 'cap2') and self.cap2.isOpened():
            self.cap2.release()
        # 重新加载默认图片
        self.load_default_image()  
        # 更新日志信息
        self.textLog.append("视频播放已停止")

    def append_log(self, message):
        # 在主线程中安全地更新日志
        self.textLog.append(message)
        QCoreApplication.processEvents()
    def update_video_display(self, frame, label):
        if frame is not None:
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(qImg)
            label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio))

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
    detect_class = 0
    # print(f"detecting {model.names[detect_class]}") # model.names返回模型所支持的所有物体类别

    # 加载DeepSort模型
    tracker = ds.DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")

    app = QtWidgets.QApplication([])
    window = MWindow()
    window.show()
    sys.exit(app.exec())