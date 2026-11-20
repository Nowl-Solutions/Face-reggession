"""Face recognition with K-Nearest Neighbors on raw pixels.

    python face_reg.py          # open the webcam and label faces live (q = quit)
    python face_reg.py eval     # measure accuracy on DATA/ without a camera
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "DATA"
CROP_W, CROP_H = 140, 180     # size of the images stored in DATA/
FEAT_W, FEAT_H = 35, 45       # images are shrunk to this before comparing
FIXED_BOX = (250, 120, CROP_W, CROP_H)   # x, y, w, h: the old fixed crop, used as fallback
K = 1   # only 4 photos per person, so a 3-way vote is easily outvoted (k=1: 14/24, k=3: 12/24)

_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def find_face(frame):
    """Return (x, y, w, h) of the largest face in a BGR frame, framed like DATA/, or None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    # Haar boxes are tight around eyes-to-mouth; DATA/ crops also include hair and chin,
    # so widen the box and stretch it to the same 140:180 shape.
    cx, cy = x + w / 2, y + h / 2
    new_w = w * 1.15
    new_h = new_w * CROP_H / CROP_W
    H, W = gray.shape
    x0, y0 = int(max(0, cx - new_w / 2)), int(max(0, cy - new_h / 2))
    x1, y1 = int(min(W, cx + new_w / 2)), int(min(H, cy + new_h / 2))
    return x0, y0, x1 - x0, y1 - y0


def crop(frame, box):
    """Cut the box out of a BGR frame and return a CROP_W x CROP_H grayscale image."""
    x, y, w, h = box
    gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (CROP_W, CROP_H))


def features(gray):
    """Turn a grayscale face image into the vector KNN compares."""
    # Shrinking keeps the same accuracy as full size and is 16x less work.
    # (cv2.equalizeHist was tried and lowered accuracy on DATA/ from 14/24 to 12/24.)
    small = cv2.resize(gray, (FEAT_W, FEAT_H), interpolation=cv2.INTER_AREA)
    return small.flatten().astype(float) / 255.0


def load_dataset(folder=DATA_DIR):
    X, y, files = [], [], []
    for person in sorted(p for p in folder.iterdir() if p.is_dir()):
        for path in sorted(person.glob("*.jpg")):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            X.append(features(image))
            y.append(person.name)
            files.append(path)
    if not X:
        raise SystemExit(f"No images found in {folder}. Run Get_data.py first.")
    return np.array(X), np.array(y), files


def neighbors(X, y, z, k=K):
    """Names and distances of the k training images closest to z, closest first."""
    d = np.sqrt(np.sum((X - z) ** 2, axis=1))
    idx = np.argsort(d)[:k]
    return y[idx], d[idx]


def decide_name(names, dists):
    """Pick the final label from the k nearest neighbours.

    names: neighbour names, closest first, e.g. ['Gow', 'Gow', 'Jang']
    dists: their distances, closest first, e.g. [6.1, 6.8, 9.4]
    Return a name from DATA/, or "Unknown" when the face matches nobody well enough.
    """
    # TODO(you): add the "Unknown" rule. Run `python face_reg.py eval` to see
    # the distance ranges for a correct match vs a wrong one before choosing.
    labels, votes = np.unique(names, return_counts=True)
    return labels[np.argmax(votes)]


def evaluate():
    """Leave-one-out: hide each image in turn and predict it from all the others."""
    X, y, files = load_dataset()
    print(f"{len(X)} images, {len(set(y))} people: {', '.join(sorted(set(y)))}\n")
    correct, right_d, wrong_d = 0, [], []
    for i in range(len(X)):
        mask = np.arange(len(X)) != i
        names, dists = neighbors(X[mask], y[mask], X[i])
        guess = decide_name(names, dists)
        ok = guess == y[i]
        correct += ok
        (right_d if names[0] == y[i] else wrong_d).append(dists[0])
        print(f"{'OK ' if ok else 'XX '} {files[i].relative_to(DATA_DIR)!s:<16} -> {guess}")
    print(f"\nAccuracy (k={K}): {correct}/{len(X)} = {correct / len(X):.0%}")
    print(f"Nearest distance when the closest image is the same person: "
          f"{np.min(right_d):.2f}-{np.max(right_d):.2f} (mean {np.mean(right_d):.2f})")
    if wrong_d:
        print(f"Nearest distance when the closest image is someone else:    "
              f"{np.min(wrong_d):.2f}-{np.max(wrong_d):.2f} (mean {np.mean(wrong_d):.2f})")


def webcam():
    X, y, _ = load_dataset()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Cannot open the webcam.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        box = find_face(frame)
        if box is None:
            cv2.putText(frame, "No face", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            names, dists = neighbors(X, y, features(crop(frame, box)))
            name = decide_name(names, dists)
            x, y0, w, h = box
            color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
            cv2.rectangle(frame, (x, y0), (x + w, y0 + h), color, 2)
            cv2.putText(frame, f"{name} ({dists[0]:.1f})", (x, max(20, y0 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("face_reg (q = quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    evaluate() if sys.argv[1:] == ["eval"] else webcam()
