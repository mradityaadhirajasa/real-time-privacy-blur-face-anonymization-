import cv2
from ultralytics import YOLO
import time
import os

input_video_path = 'CCTV.mp4' 
output_video_path = 'cencored_cctv.mp4'

if not os.path.exists(input_video_path):
    print(f"⚠️ Error: File '{input_video_path}' tidak ditemukan!")
    print("Pastikan file ada di folder yang sama dengan script ini.")
    exit()

print("⏳ Memuat model YOLOv8-Face...")
model = YOLO('yolov8n-face.pt')

cap = cv2.VideoCapture(input_video_path)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print(f"🚀 Memulai pemrosesan video: {width}x{height} @ {fps:.2f} FPS")
print("Tekan 'q' untuk menghentikan proses lebih awal.")

while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        print("✅ Video selesai diproses.")
        break

    results = model.predict(frame, conf=0.4, verbose=False)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                w_face = x2 - x1
                k_size = (w_face // 5) | 1 
                if k_size < 3: k_size = 3
                
                roi_blurred = cv2.GaussianBlur(roi, (k_size, k_size), 30)
                frame[y1:y2, x1:x2] = roi_blurred
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    
    out.write(frame)
    
    cv2.imshow('CCTV Privacy Processing', frame)

    # Tekan 'q' untuk stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release() 
cv2.destroyAllWindows()

print(f"💾 File berhasil disimpan sebagai: {output_video_path}")