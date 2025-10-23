import os; os.environ["NPU_COMPLETION_CYCLES"] = "0"

import glob
from itertools import islice
import time

import cv2
import numpy as np
import onnx
import tqdm

import furiosa.runtime.session
from furiosa.optimizer import optimize_model
from furiosa.quantizer import quantize, Calibrator, CalibrationMethod

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
     
    padded_img = padded_img.transpose(swap)  # line 15
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)  # line 16
    print(padded_img)
    return padded_img, r
    
def warboy_preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
     
    padded_img = padded_img.transpose(swap)  # line 15
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32) # line 16
    return padded_img, r

model = onnx.load_model("yolox_l.onnx")

calibration_dataset = [
    preproc(cv2.imread(image), (640, 640))[0][np.newaxis, ...]  # line 2
    for image in islice(glob.iglob("calibration_fhd_frames/*.jpg"), 100)
]

model = optimize_model(model)

#####test


calibrator = Calibrator(model, CalibrationMethod.MIN_MAX_ASYM)

for calibration_data in tqdm.tqdm(calibration_dataset, desc="Calibration", unit="images", mininterval=0.5):
    calibrator.collect_data([[calibration_data]])

ranges = calibrator.compute_range()

model_quantized = quantize(model, ranges)

compiler_config = {
    "permute_input": 
        [
            [0, 2, 3, 1],
        ],
}

def nms(boxes, scores, iou_threshold=0.2):
    x1 = boxes[:,0] - boxes[:,2]
    y1 = boxes[:,1] - boxes[:,3]
    x2 = boxes[:,0] + boxes[:,2]
    y2 = boxes[:,1] + boxes[:,3]
    
    areas = (x2 - x1 +1) * (y2 -y1 +1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds+1]
    return keep

video_path = "input_video/home_video_1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"오류: 비디오 파일을 열 수 없습니다. 경로를 확인하세요: {video_path}")
    exit()
    
CONF_THRESHOLD = 0.51
IOU_THRESHOLD = 0.3

# 벤치마킹 변수 초기화
total_frames = 0
elapsed_time = 0
real_elapsed_time = 0

print("NPU 세션을 생성 중입니다. 잠시 기다려주세요...")

#with furiosa.runtime.session.create(model_quantized, compiler_config=compiler_config) as session:
#with furiosa.runtime.session.create("yolov8x.onnx") as session:
with furiosa.runtime.session.create(model_quantized) as session:
    print("세션 생성 완료. 비디오 처리를 시작합니다. 'q' 키를 누르면 종료됩니다.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("비디오 스트림이 종료되었습니다.")
            break
        
        display_frame = frame.copy()
        #frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
        # 타이머 시작 (전처리 + 추론 시간 측정)
        start = time.perf_counter_ns()

        # 3. 전처리 (warboy_preproc 사용)
        # NPU 입력에 맞게 리사이즈 및 패딩. 'ratio'가 스케일 팩터.
        input_tensor, ratio = warboy_preproc(frame, (640, 640))
        input_tensor_batched = input_tensor[np.newaxis, ...] # (1, H, W, C)
        print(input_tensor_batched)
        
        # 4. NPU 추론 실행
        outputs = session.run([input_tensor_batched])
        #outputs = session.run(inputs)
        # 타이머 종료 (여기까지가 NPU Latency)
        elapsed_time += time.perf_counter_ns() - start
        total_frames += 1

        # 5. 후처리
        pred = outputs[0].numpy()
        if total_frames == 1:
            print(f"NPU Output shape: {pred.shape}") # (1, 8400, 85)
        pred = np.squeeze(pred)
        input_shape = (640, 640)
        strides = [8, 16, 32]
        grids_all = []
        strides_all = []
        for stride in strides:
            grid_h, grid_w = input_shape[0] // stride, input_shape[1] // stride
            grid_x, grid_y = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
            grid_xy = np.stack((grid_x.flatten(), grid_y.flatten()), axis=1)
            grids_all.append(grid_xy)
            strides_all.append(np.full((grid_h * grid_w, 1), stride))
        grids_all = np.concatenate(grids_all, axis=0)
        strides_all = np.concatenate(strides_all, axis=0)
        raw_box_offsets = pred[:, :4]
        cxcy_640 = (raw_box_offsets[:, :2] + grids_all) * strides_all
        wh_640 = np.exp(raw_box_offsets[:, 2:4]) * strides_all
        boxes_640 = np.concatenate((cxcy_640, wh_640), axis=1)
        obj_conf_logits = pred[:, 4] # (N,)
        scores_logits = pred[:, 5:]  # (N, 80)
        # 로짓(logit)을 확률(0~1)로 변환하기 위해 sigmoid 함수를 수동으로 적용합니다.
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        obj_conf_probs = sigmoid(obj_conf_logits)
        scores_probs = sigmoid(scores_logits)
        scores_all = scores_probs * obj_conf_probs[:, np.newaxis]
        conf = scores_all.max(axis=1)    # (N,) - 0~1 사이의 확률값
        cls = scores_all.argmax(axis=1)  # (N,)
        # 신뢰도 임계값(Confidence Threshold)을 기준으로 필터링
        valid_indices = conf > CONF_THRESHOLD
        valid_boxes_640 = boxes_640[valid_indices]
        valid_conf = conf[valid_indices]
        valid_cls = cls[valid_indices]
        if valid_indices.any():
        # 감지된 박스들의 좌표를 출력
            print("--- Detected Box Coords (raw) ---")
            print(valid_boxes_640)
            print("---------------------------------")
        # 이제 0~1 사이의 올바른 신뢰도가 출력되어야 함
        print(f"Max confidence in this frame: {conf.max():.4f}")
        # 좌표 변환: (cx, cy, w, h) @ 640 -> (x1, y1, x2, y2) @ 원본 프레임
        # warboy_preproc는 비율을 유지하므로 scale_x, scale_y가 동일.
        # scale 팩터 = 1 / ratio
        scale = 1 / ratio
        
        xyxy_boxes_orig = valid_boxes_640 * scale

        # 6. NMS 실행 및 시각화
        # -----------------------------------------------------------------
        keep = nms(xyxy_boxes_orig, valid_conf, iou_threshold=IOU_THRESHOLD)

        for i in keep:
            x1_k, y1_k, x2_k, y2_k = xyxy_boxes_orig[i]
            print("here:",x1_k, y1_k, x2_k, y2_k)
            conf_k = valid_conf[i]
            cls_k = valid_cls[i]
            # 바운딩 박스 그리기
            #cv2.rectangle(display_frame, (int(x1_k-140), int(y1_k-220)), (int(x1_k+x2_k-140), int(y1_k+y2_k-220)), (0, 255, 0), 2)
            cv2.rectangle(display_frame, (int(x1_k-(x2_k/2)), int(y1_k-(y2_k/2))), (int(x1_k+(x2_k/2)), int(y1_k+(y2_k/2))), (0, 255, 0), 2)
            
            # 클래스 및 신뢰도 텍스트 그리기
            label = f"{COCO_CLASSES[cls_k]}: {conf_k:.2f}"
            cv2.putText(display_frame, label, (int(x1_k), int(y1_k) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # -----------------------------------------------------------------
        display_output = cv2.resize(display_frame, (1280, 720))
        cv2.imshow("YOLOv8 NPU Inference (Press 'q' to quit)", display_output)
        real_elapsed_time += time.perf_counter_ns() - start
        #key = cv2.waitKey(0) 
        
        #if key & 0xFF == ord('q'):
            #break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()

# 최종 성능 리포트
if total_frames > 0:
    latency = elapsed_time / total_frames  # ns
    real_latency = real_elapsed_time / total_frames
    fps = (1 / latency) * 1_000_000_000
    real_fps=(1 / real_latency) * 1_000_000_000
    print(f"\n--- 최종 결과 (NPU Latency) ---")
    print(f"처리된 총 프레임: {total_frames}")
    print(f"WARBOY Latency (전처리 + 추론): {latency / 1_000_000:.2f} ms")
    print(f"WARBOY FPS (전처리 + 추론): {fps:.2f} fps")
    print(f"SW + WARBOY Latency (전처리 + 추론): {real_latency / 1_000_000:.2f} ms")
    print(f"SW + WARBOY FPS (전처리 + 추론): {real_fps:.2f} fps")
else:
    print("처리된 프레임이 없습니다.")
