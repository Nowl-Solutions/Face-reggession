import cv2
import numpy as np
import os

# โฟลเดอร์ที่ไฟล์นี้อยู่ — รันจากที่ไหนก็หารูปเจอ (สำคัญตอนคนอื่น clone ไปรัน)
BASE = os.path.dirname(os.path.abspath(__file__))


def knn(X, y, z, k=3):
    d = np.sum((X - z) ** 2, axis=1)
    idx = np.argsort(d)[:k]
    cls, vote = np.unique(y[idx], return_counts=True)
    return cls[np.argmax(vote)], np.sqrt(d[idx[0]])


def prep(gray):
    # ปรับแสงให้เท่ากัน แล้วแปลงเป็น float (ถ้าเป็น uint8 ตอนลบกันค่าจะวนติดลบผิด)
    return cv2.equalizeHist(gray).astype(np.float32).flatten()


# ---------- โหลดข้อมูล: 1 โฟลเดอร์ = 1 คน ----------
X, y = [], []
for f in sorted(os.listdir(BASE)):
    folder = os.path.join(BASE, f)
    if not os.path.isdir(folder) or f.startswith(('.', '_')):
        continue
    for i in os.listdir(folder):
        if i.endswith('.jpg'):
            img = cv2.imread(os.path.join(folder, i), cv2.IMREAD_GRAYSCALE)
            if img is not None and img.shape == (180, 140):
                X.append(prep(img))
                y.append(f)

if len(set(y)) == 0:
    raise SystemExit('ไม่พบรูปเลย — รัน "Get data.py" เก็บหน้าก่อน')
X = np.array(X)
y = np.array(y)
for name in np.unique(y):
    print(f'{name}: {np.sum(y == name)} รูป')

# ---------- เปิดกล้องแล้วทายหน้า ----------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    face = cv2.cvtColor(frame[120:300, 250:390, :], cv2.COLOR_BGR2GRAY)
    name, dist = knn(X, y, prep(face))
    cv2.rectangle(frame, (250, 120), (390, 300), (0, 255, 0), 2)
    cv2.putText(frame, f'{name} {unknow} ', (250, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('recognize', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):   # q = ออก
        break

cap.release()
cv2.destroyAllWindows()
