import cv2
import os

# โฟลเดอร์ที่ไฟล์นี้อยู่ — รูปจะไปอยู่ข้างๆ recognize.py เสมอ
BASE = os.path.dirname(os.path.abspath(__file__))


def people():
    # ทุกโฟลเดอร์ในโปรเจกต์ = รายชื่อคน (ข้าม .idea, .git ฯลฯ)
    return sorted(f for f in os.listdir(BASE)
                  if os.path.isdir(os.path.join(BASE, f)) and not f.startswith(('.', '_')))


def next_number(name):
    # เลขรูปถัดไป = เลขที่มากที่สุดในโฟลเดอร์ + 1 (ไม่เขียนทับรูปเก่า)
    nums = [int(f[:-4]) for f in os.listdir(os.path.join(BASE, name))
            if f.endswith('.jpg') and f[:-4].isdigit()]
    return max(nums, default=0) + 1


# ---------- เลือกคนก่อนเริ่ม ----------
names = people()
for n, p in enumerate(names, 1):
    print(f'  {n}. {p}')
ans = input('พิมพ์เลขคนที่มีอยู่ หรือพิมพ์ชื่อใหม่ (ภาษาอังกฤษ): ').strip()
name = names[int(ans) - 1] if ans.isdigit() else ans
os.makedirs(os.path.join(BASE, name), exist_ok=True)
names = people()
print('ในหน้าต่างกล้อง: s = บันทึก | กดเลข 1-9 = สลับคน | q = ออก')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    face = cv2.cvtColor(frame[120:300, 250:390, :], cv2.COLOR_BGR2GRAY)  # ตัดก่อนวาดกรอบ
    cv2.rectangle(frame, (250, 120), (390, 300), (0, 0, 255), 2)
    cv2.putText(frame, f'{name}  next: {next_number(name)}.jpg', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow('frame', frame)
    cv2.imshow('face', face)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):                                   # s = บันทึกรูปให้คนที่เลือกอยู่
        i = next_number(name)
        cv2.imwrite(os.path.join(BASE, name, f'{i}.jpg'), face)
        print('saved', name, i)
    elif ord('1') <= key <= ord('9') and key - ord('1') < len(names):
        name = names[key - ord('1')]                      # กดเลข = สลับไปคนนั้น
        print('เปลี่ยนเป็น', name)
    elif key == ord('q'):                                 # q = ออก
        break

cap.release()
cv2.destroyAllWindows()
