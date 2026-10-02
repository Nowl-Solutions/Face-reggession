import numpy as np
import matplotlib.pyplot as plt
import os
import cv2  # เพิ่ม

def knn(X, y, z, k=1):
    d = np.sum((X - z) ** 2, axis=1)
    idx = np.argsort(d)[:k]
    cls, vote = np.unique(y[idx], return_counts=True)
    return cls[np.argmax(vote)]

y = []
X = []
for f in os.listdir():
    if not os.path.isfile(f) and not f.startswith("."):
        for i in os.listdir(f):
            if i.endswith(".jpg"):
                x = plt.imread(f+"/"+i)
                X.append(x.flatten())
                y.append(f)
X = np.array(X, dtype=float)  # แก้
y = np.array(y)               # แก้
print(X.shape)
print(y)

# ===== เพิ่ม: เปิดกล้องแล้วทายชื่อ =====
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    ret, frame = cap.read()
    face = cv2.cvtColor(frame[120:300, 250:390, :], cv2.COLOR_BGR2GRAY)
    name = knn(X, y, face.flatten().astype(float))
    cv2.rectangle(frame, (250,120), (390,300), (0,255,0), 2)
    cv2.putText(frame, name, (250,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
