
from datetime import datetime
from pathlib import Path
import shutil
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "ML_Data"
K = 3


def vector(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    h, w = image.shape
    face = cv2.resize(image[int(.06*h):int(.96*h), int(.06*w):int(.94*w)],
                      (144, 180))
    gx = cv2.Sobel(face, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(face, cv2.CV_32F, 0, 1)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    angle %= 180
    cells = []
    for y in range(4):
        for x in range(4):
            ys, xs = slice(y*45, (y+1)*45), slice(x*36, (x+1)*36)
            hist, _ = np.histogram(angle[ys, xs], bins=9, range=(0, 180),
                                   weights=magnitude[ys, xs])
            cells.extend(hist / (np.linalg.norm(hist) + 1e-6))
    return np.array(cells, dtype=np.float32)


def main():
    files, vectors = [], []
    for path in sorted(IMAGES.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        features = vector(path)
        if features is not None:
            files.append(path)
            vectors.append(features)

    if len(files) < K:
        raise SystemExit(f"Need at least {K} readable images; found {len(files)}.")

    X = np.asarray(vectors, dtype=np.float32)
    scale = X.std(axis=0)
    scale[scale == 0] = 1
    X = np.ascontiguousarray((X - X.mean(axis=0)) / scale)
    cv2.setRNGSeed(42)
    _, labels, centers = cv2.kmeans(
        X, K, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4),
        10, cv2.KMEANS_PP_CENTERS,
    )

    output = ROOT / f"Face_Groups_{datetime.now():%Y%m%d_%H%M%S}"
    distances = np.linalg.norm(X - centers[labels.ravel()], axis=1)
    for path, label, distance in zip(files, labels.ravel(), distances):
        group = output / f"group_{label + 1}"
        group.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, group / path.name)
        print(f"{path.name} -> group {label + 1} (distance {distance:.3f})")
    print(f"\nGrouped {len(files)} images into {K} groups: {output}")


if __name__ == "__main__":
    main()
