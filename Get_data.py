"""Collect face images for one person from the webcam.

    python Get_data.py <name>

Look at the camera, press s to save the face in the box, q to quit.
Images go to DATA/<name>/ and numbering continues after existing files.
Turn your head a little between shots so the model sees different angles.
"""
import sys

import cv2

from face_reg import DATA_DIR, FIXED_BOX, crop, find_face

if len(sys.argv) != 2:
    raise SystemExit("Usage: python Get_data.py <name>")

folder = DATA_DIR / sys.argv[1]
folder.mkdir(parents=True, exist_ok=True)
numbers = [int(p.stem) for p in folder.glob("*.jpg") if p.stem.isdigit()]
i = max(numbers, default=0) + 1

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open the webcam.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    box = find_face(frame) or FIXED_BOX   # no face found: fall back to the old fixed box
    x, y, w, h = box
    face = crop(frame, box)
    shown = frame.copy()
    cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(shown, f"{folder.name}: next #{i}  (s = save, q = quit)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("frame", shown)
    cv2.imshow("face", face)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        cv2.imwrite(str(folder / f"{i}.jpg"), face)
        print(f"saved {folder.name}/{i}.jpg")
        i += 1
    elif key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
