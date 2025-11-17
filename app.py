import cv2

cap = cv2.VideoCapture(0)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        roi_blurred = cv2.GaussianBlur(roi, (99, 99), 30)
        frame[y:y+h, x:x+w] = roi_blurred
        
    cv2.imshow('Privacy Cam (Tekan Q untuk Keluar)', frame)

    # Tekan 'q' untuk stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()