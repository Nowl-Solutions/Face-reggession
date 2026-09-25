import cv2
import os

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
name = 'Klahan'
name = 'jesda'# ① ชื่อโฟลเดอร์ = ชื่อคน (label ของ KNN)
name = 'santisuk'
i = 1
i =2
i = 3
os.makedirs(name, exist_ok=True)         # ② รันซ้ำได้ ไม่ติด FileExistsError
while True:
    ret, frame = cap.read()
    if not ret:
        break
    face = cv2.cvtColor(frame[120:300, 250:390, :], cv2.COLOR_BGR2GRAY)  # ตัดก่อนวาดกรอบ
    cv2.rectangle(frame, (250,120), (390,300), (0,0,255), 2)
    cv2.imshow('frame', frame)
    cv2.imshow('face', face)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):                  # s = บันทึกรูป
        cv2.imwrite(f'{name}/{i}.jpg', face)
        print('saved', i)
        i += 1
    elif key == ord('q'):                # ③ q = ออก (โค้ดเดิมไม่มีทางออกจากลูป)
        break
cap.release()
cv2.destroyAllWindows()