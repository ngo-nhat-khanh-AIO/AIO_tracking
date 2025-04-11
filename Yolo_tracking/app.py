import streamlit as st
import cv2
import tempfile
import shutil
import os
import pandas as pd
from tracker import Tracker  # Import Tracker class từ module của bạn
from ultralytics import YOLO  # Import YOLO class từ module của bạn

# Khai báo các biến và đối tượng cần thiết
class_list = ['xe-oto', 'xe-may']
model_yolo = YOLO("best.pt")
tracker = Tracker()
down = {}
up = {}
counter_down = 0
counter_up = 0

# Hàm xử lý video và hiển thị kết quả
def process_video(video_path):
    global counter_down, counter_up  # Declare these as global to modify them within the function
    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, "temp_video.mp4")
    with open(temp_video_path, "wb") as temp_file:
        temp_file.write(video_path.read())
        
    cap = cv2.VideoCapture(temp_video_path)
    count = 0  # Khai báo và gán giá trị ban đầu cho biến count
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1  # Tăng biến count sau mỗi lần đọc khung hình
            frame = cv2.resize(frame, (1020, 500))
            results = model_yolo.predict(frame)
            a = results[0].boxes.data
            a = a.detach().cpu().numpy()
            px = pd.DataFrame(a).astype("float")
            list = []
            for index, row in px.iterrows():
                x1 = int(row[0])
                y1 = int(row[1])
                x2 = int(row[2])
                y2 = int(row[3])
                d = int(row[5])
                c = class_list[d]
                list.append([x1, y1, x2, y2, c])
                # Vẽ bounding box và nhãn trên frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, c, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            bbox_id = tracker.update([item[:4] for item in list])
            for bbox, (x1, y1, x2, y2, c) in zip(bbox_id, list):
                x3, y3, x4, y4, id = bbox
                cx = int(x3 + x4) // 2
                cy = int(y3 + y4) // 2
                red_line_y = 198
                blue_line_y = 268   
                offset = 7
                if red_line_y < (cy + offset) and red_line_y > (cy - offset):
                    down[id] = cy   
                if id in down:
                    if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
                        if down[id] < blue_line_y:  # Ensure that the object has actually moved down
                            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                            cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                            counter_down += 1
                            del down[id]  # Remove the ID to prevent double counting
                if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
                    up[id] = cy   
                if id in up:
                    if red_line_y < (cy + offset) and red_line_y > (cy - offset):
                        if up[id] > red_line_y:  # Ensure that the object has actually moved up
                            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                            cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                            counter_up += 1
                            del up[id]  # Remove the ID to prevent double counting
            cv2.line(frame, (172, 198), (774, 198), (0, 0, 255), 3)
            cv2.line(frame, (8, 268), (927, 268), (255, 0, 0), 3)
            cv2.putText(frame, 'Di xuong - ' + str(counter_down), (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Di len - ' + str(counter_up), (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            yield frame
    finally:
        cap.release()  # Ensure the video capture is released
        shutil.rmtree(temp_dir)  # Delete the temporary directory

# Gọi Streamlit để hiển thị video và kết quả xử lý
st.title("Đếm số lưu lượng xe")
video_path = st.file_uploader("Chọn video", type=['mp4', 'mov'])
if video_path:
    video_stream = process_video(video_path)
    stframe = st.empty()
    for frame in video_stream:
        stframe.image(frame, channels="BGR")
    st.write("Tổng số xe đi xuống:", counter_down)
    st.write("Tổng số xe đi lên:", counter_up)





