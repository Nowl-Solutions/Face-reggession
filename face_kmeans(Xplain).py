
# นำเข้า datetime สำหรับสร้างชื่อโฟลเดอร์ด้วยวันที่และเวลา
from datetime import datetime

# นำเข้า Path สำหรับจัดการตำแหน่งไฟล์และโฟลเดอร์
from pathlib import Path

# นำเข้า shutil สำหรับคัดลอกไฟล์ภาพ
import shutil

# นำเข้า OpenCV สำหรับประมวลผลภาพและทำ K-Means
import cv2

# นำเข้า NumPy สำหรับคำนวณตัวเลขและจัดการ Array
import numpy as np


# หาตำแหน่งโฟลเดอร์ที่เก็บไฟล์ Python ปัจจุบัน
ROOT = Path(__file__).resolve().parent

# กำหนดตำแหน่งโฟลเดอร์ที่เก็บรูปภาพ
IMAGES = ROOT / "ML_Data"

# กำหนดจำนวนกลุ่มที่ต้องการแบ่งด้วย K-Means
K = 3


# ฟังก์ชันแปลงภาพให้เป็น Feature Vector ด้วย HOG
def vector(path):

    # อ่านภาพจากตำแหน่ง path และแปลงเป็นภาพขาวดำ
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    # ตรวจสอบว่าอ่านภาพสำเร็จหรือไม่
    if image is None:
        # หากอ่านไม่ได้ ให้ส่งค่า None กลับไป
        return None

    # อ่านความสูง (h) และความกว้าง (w) ของภาพ
    h, w = image.shape

    # ตัดขอบภาพแนวตั้งจาก 6% ถึง 96%
    # ตัดขอบภาพแนวนอนจาก 6% ถึง 94%
    # แล้วปรับภาพให้กว้าง 144 และสูง 180 พิกเซล
    face = cv2.resize(
        image[int(.06*h):int(.96*h), int(.06*w):int(.94*w)],
        (144, 180)
    )

    # คำนวณ Gradient ของภาพตามแกน X ด้วย Sobel
    # 1, 0 หมายถึงหาอนุพันธ์แกน X ลำดับที่ 1
    gx = cv2.Sobel(face, cv2.CV_32F, 1, 0)

    # คำนวณ Gradient ของภาพตามแกน Y ด้วย Sobel
    # 0, 1 หมายถึงหาอนุพันธ์แกน Y ลำดับที่ 1
    gy = cv2.Sobel(face, cv2.CV_32F, 0, 1)

    # แปลง Gradient จากแกน X และ Y เป็นขนาดและมุม
    # magnitude = ความแรงของ Gradient
    # angle = ทิศทางของ Gradient หน่วยเป็นองศา
    magnitude, angle = cv2.cartToPolar(
        gx, gy,
        angleInDegrees=True
    )

    # ปรับมุมให้อยู่ในช่วง 0 ถึงน้อยกว่า 180 องศา
    # เพราะ HOG แบบ Unsigned ไม่แยกทิศที่ต่างกัน 180 องศา
    angle %= 180

    # สร้าง List ว่างสำหรับเก็บ Histogram ของทุก Cell
    cells = []

    # แบ่งภาพตามแนวตั้งออกเป็น 4 ส่วน
    for y in range(4):

        # แบ่งภาพตามแนวนอนออกเป็น 4 ส่วน
        # รวมทั้งหมด 4 x 4 = 16 Cells
        for x in range(4):

            # กำหนดช่วงพิกเซลแนวตั้งของ Cell ปัจจุบัน
            # แต่ละ Cell สูง 45 พิกเซล
            ys = slice(y * 45, (y + 1) * 45)

            # กำหนดช่วงพิกเซลแนวนอนของ Cell ปัจจุบัน
            # แต่ละ Cell กว้าง 36 พิกเซล
            xs = slice(x * 36, (x + 1) * 36)

            # สร้าง Histogram ของมุม Gradient ใน Cell
            # angle[ys, xs] = มุม Gradient ของพิกเซลใน Cell
            # bins=9 = แบ่งออกเป็น 9 ช่วงมุม
            # range=(0,180) = ช่วงมุมที่นำมาพิจารณา
            # weights=magnitude = ให้น้ำหนักตามความแรง Gradient
            # hist = คะแนนรวมของแต่ละช่วงมุม
            # _ = ขอบช่วง Histogram ซึ่งไม่ได้ใช้ต่อ
            hist, _ = np.histogram(
                angle[ys, xs],
                bins=9,
                range=(0, 180),
                weights=magnitude[ys, xs]
            )

            # คำนวณ L2 Norm ของ Histogram
            # บวก 1e-6 เพื่อป้องกันการหารด้วยศูนย์
            # จากนั้น Normalize และเพิ่มค่าเข้า List cells
            cells.extend(
                hist / (np.linalg.norm(hist) + 1e-6)
            )

    # แปลง List เป็น NumPy Array ชนิด float32
    # มี 16 Cells x 9 Bins = 144 Features ต่อหนึ่งภาพ
    return np.array(cells, dtype=np.float32)


# ฟังก์ชันหลักของโปรแกรม
def main():

    # files เก็บตำแหน่งไฟล์ภาพที่อ่านสำเร็จ
    # vectors เก็บ Feature Vector ของแต่ละภาพ
    files, vectors = [], []

    # ค้นหาไฟล์ทั้งหมดใน ML_Data รวมถึงโฟลเดอร์ย่อย
    # sorted() ใช้เรียงลำดับตำแหน่งไฟล์
    for path in sorted(IMAGES.rglob("*")):

        # ตรวจสอบนามสกุลไฟล์ว่าเป็นรูปภาพที่รองรับหรือไม่
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:

            # หากไม่ใช่ไฟล์ภาพที่รองรับ ให้ข้าม
            continue

        # เรียกฟังก์ชัน vector() เพื่อดึง HOG Features
        features = vector(path)

        # ตรวจสอบว่าดึง Features สำเร็จหรือไม่
        if features is not None:

            # เก็บตำแหน่งไฟล์ภาพ
            files.append(path)

            # เก็บ Feature Vector ของภาพนั้น
            vectors.append(features)

    # ตรวจสอบว่ามีภาพเพียงพอต่อจำนวนกลุ่ม K หรือไม่
    if len(files) < K:

        # หากภาพน้อยกว่า K ให้หยุดโปรแกรมพร้อมแสดงข้อความ
        raise SystemExit(
            f"Need at least {K} readable images; found {len(files)}."
        )

    # แปลง Features ทั้งหมดเป็น NumPy Array แบบ float32
    # ถ้ามี N ภาพ จะได้ Matrix ขนาด N x 144
    X = np.asarray(vectors, dtype=np.float32)

    # คำนวณ Standard Deviation ของแต่ละ Feature
    # axis=0 หมายถึงคำนวณแยกตามคอลัมน์
    scale = X.std(axis=0)

    # ป้องกันการหารด้วยศูนย์
    # หาก Standard Deviation เท่ากับ 0 ให้เปลี่ยนเป็น 1
    scale[scale == 0] = 1

    # ทำ Standardization โดยใช้สูตร Z = (X - Mean) / Std
    # X.mean(axis=0) = ค่าเฉลี่ยของแต่ละ Feature
    # np.ascontiguousarray() = จัดข้อมูลให้ต่อเนื่องในหน่วยความจำ
    X = np.ascontiguousarray(
        (X - X.mean(axis=0)) / scale
    )

    # ตั้งค่า Random Seed เพื่อให้การสุ่มทำซ้ำได้
    cv2.setRNGSeed(42)

    # ใช้อัลกอริทึม K-Means แบ่งภาพออกเป็น K กลุ่ม
    # X = Matrix ของ Features
    # K = จำนวนกลุ่ม
    # None = ไม่กำหนด Labels เริ่มต้น
    # MAX_ITER = เงื่อนไขหยุดเมื่อถึงจำนวนรอบสูงสุด
    # EPS = เงื่อนไขหยุดเมื่อศูนย์กลางเปลี่ยนแปลงน้อย
    # 100 = จำนวนรอบสูงสุดต่อการทดลอง
    # 1e-4 = ค่า Threshold สำหรับการหยุด
    # 10 = ทดลองเริ่มต้น K-Means จำนวน 10 ครั้ง
    # KMEANS_PP_CENTERS = เลือกจุดศูนย์กลางเริ่มต้นแบบ K-Means++
    #
    # _ = Compactness หรือผลรวมระยะห่างกำลังสอง
    # labels = หมายเลขกลุ่มของแต่ละภาพ เริ่มจาก 0
    # centers = เวกเตอร์ศูนย์กลางของแต่ละกลุ่ม
    _, labels, centers = cv2.kmeans(
        X,
        K,
        None,
        (
            cv2.TERM_CRITERIA_EPS
            + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            1e-4
        ),
        10,
        cv2.KMEANS_PP_CENTERS
    )

    # กำหนดชื่อโฟลเดอร์ผลลัพธ์พร้อมวันที่และเวลา
    # เช่น Face_Groups_20261009_154600
    output = ROOT / f"Face_Groups_{datetime.now():%Y%m%d_%H%M%S}"

    # คำนวณระยะห่างระหว่างแต่ละภาพกับศูนย์กลางกลุ่ม
    # labels.ravel() = แปลง Labels เป็น Array หนึ่งมิติ
    # centers[labels.ravel()] = เลือก Center ตามกลุ่มของภาพ
    # X - centers[...] = หาผลต่างของ Feature Vectors
    # np.linalg.norm(..., axis=1) = คำนวณ Euclidean Distance
    distances = np.linalg.norm(
        X - centers[labels.ravel()],
        axis=1
    )

    # วนลูปผ่านภาพ หมายเลขกลุ่ม และระยะห่างพร้อมกัน
    for path, label, distance in zip(
        files, labels.ravel(), distances
    ):

        # สร้างตำแหน่งโฟลเดอร์ตามกลุ่ม
        # บวก 1 เพราะ Labels เริ่มนับจาก 0
        # เช่น group_1, group_2, group_3
        group = output / f"group_{label + 1}"

        # สร้างโฟลเดอร์หากยังไม่มี
        # parents=True = สร้างโฟลเดอร์แม่ด้วย
        # exist_ok=True = ไม่ Error หากโฟลเดอร์มีอยู่แล้ว
        group.mkdir(parents=True, exist_ok=True)

        # กำหนดตำแหน่งไฟล์ปลายทางโดยใช้ชื่อภาพเดิม
        destination = group / path.name

        # ตรวจสอบว่ามีไฟล์ชื่อเดียวกันในกลุ่มแล้วหรือไม่
        if destination.exists():

            # หาตำแหน่งสัมพัทธ์ของภาพจากโฟลเดอร์ ML_Data
            relative = str(path.relative_to(IMAGES))

            # นำเข้า Library สำหรับสร้าง Hash
            import hashlib

            # สร้าง Hash จากตำแหน่งไฟล์
            # encode("utf-8") = แปลงข้อความเป็น Bytes
            # sha1() = คำนวณค่า Hash
            # hexdigest() = แปลง Hash เป็นเลขฐาน 16
            # [:8] = ใช้เพียง 8 ตัวอักษรแรก
            suffix = hashlib.sha1(
                relative.encode("utf-8")
            ).hexdigest()[:8]

            # เพิ่ม Hash ต่อท้ายชื่อไฟล์เพื่อลดโอกาสชื่อซ้ำ
            destination = group / f"{path.stem}_{suffix}{path.suffix}"

        # คัดลอกภาพไปยังโฟลเดอร์กลุ่มที่ K-Means กำหนด
        # copy2() พยายามรักษา Metadata ของไฟล์
        shutil.copy2(path, destination)

        # แสดงชื่อภาพ หมายเลขกลุ่ม และระยะห่างจาก Center
        # :.3f หมายถึงแสดงเลขทศนิยม 3 ตำแหน่ง
        print(
            f"{path.name} -> group {label + 1} "
            f"(distance {distance:.3f})"
        )

    # แสดงสรุปจำนวนภาพ จำนวนกลุ่ม และโฟลเดอร์ผลลัพธ์
    # \n หมายถึงขึ้นบรรทัดใหม่
    print(
        f"\nGrouped {len(files)} images "
        f"into {K} groups: {output}"
    )


# ตรวจสอบว่าไฟล์ Python นี้ถูกเรียกใช้งานโดยตรงหรือไม่
if __name__ == "__main__":

    # เรียกฟังก์ชันหลักเพื่อเริ่มทำงาน
    main()
