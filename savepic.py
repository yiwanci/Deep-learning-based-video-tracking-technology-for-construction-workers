import os
import cv2

def save_frame(video_path, frame_number, output_dir):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 设置视频的帧数
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # 读取帧
    ret, frame = cap.read()
    
    # 如果成功获取到帧，则保存它
    if ret:
        # 从视频路径中提取文件名
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        # 构建保存路径，格式为 文件名_帧数.jpg
        save_path = os.path.join(output_dir, f'{video_name}_{frame_number}.jpg')
        cv2.imwrite(save_path, frame)
        print(f'Frame at position {frame_number} saved to {save_path}')
    else:
        print(f'Failed to retrieve frame at position {frame_number}')
    
    # 释放视频文件
    cap.release()

# 使用示例
video_path = 'yolov9.avi'  # 视频文件路径
frame_number = 201  # 想要保存的帧号
output_dir = 'results'  # 保存目录

save_frame(video_path, frame_number, output_dir)
