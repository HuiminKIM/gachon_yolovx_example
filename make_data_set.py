import cv2
import os

# 캘리브레이션용 프레임을 저장할 새 폴더 생성
calib_dir = "calibration_fhd_frames"
os.makedirs(calib_dir, exist_ok=True)

video_path = "input_video/road_video_2.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = 0
saved_count = 0
frame_interval = 10 # 10 프레임마다 1장씩 저장

print(f"'{video_path}'에서 프레임 추출 중...")

while cap.isOpened() and saved_count < 200: # 최대 200장 저장
    ret, frame = cap.read()
    if not ret:
        break
        
    if frame_count % frame_interval == 0:
        save_path = os.path.join(calib_dir, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(save_path, frame)
        saved_count += 1
        
    frame_count += 1

cap.release()
print(f"'{calib_dir}' 폴더에 총 {saved_count}개의 프레임 저장 완료.")
