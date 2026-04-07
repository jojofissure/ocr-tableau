import cv2
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# --- PREPROCESSING ---
img = cv2.imread("tableau1.jpg")
print(f"Image chargée : {img.shape}")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

board_corners = None
for c in contours[:5]:
    epsilon = 0.02 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)
    if len(approx) == 4:
        board_corners = approx
        break

if board_corners is None:
    print("Tableau non détecté")
    exit()

print("Tableau détecté, correction perspective...")
W, H = 3508, 2480
pts = board_corners.reshape(4, 2).astype(np.float32)
s = pts.sum(axis=1)
diff = np.diff(pts, axis=1)
pts_ordered = np.float32([
    pts[np.argmin(s)],
    pts[np.argmin(diff)],
    pts[np.argmax(s)],
    pts[np.argmax(diff)]
])
pts_dst = np.float32([[0,0],[W,0],[W,H],[0,H]])
M = cv2.getPerspectiveTransform(pts_ordered, pts_dst)
warped = cv2.warpPerspective(img, M, (W, H))

gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
binary = cv2.adaptiveThreshold(
    gray_w, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 31, 10
)
cv2.imwrite("output_binary.png", binary)
print("Preprocessing OK")

# --- SEGMENTATION EN LIGNES ---
inverted = cv2.bitwise_not(binary)
h_proj = np.sum(inverted, axis=1)
threshold = h_proj.max() * 0.05
in_line = False
lines = []
start = 0

for i, val in enumerate(h_proj):
    if val > threshold and not in_line:
        start = i
        in_line = True
    elif val <= threshold and in_line:
        if i - start > 10:
            lines.append((start, i))
        in_line = False

print(f"{len(lines)} ligne(s) détectée(s)")

# --- OCR ---
print("Chargement TrOCR...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

full_image = Image.open("output_binary.png").convert("RGB")
img_array = np.array(full_image)

for idx, (y1, y2) in enumerate(lines):
    crop = Image.fromarray(img_array[y1:y2, :])
    pixel_values = processor(crop, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_new_tokens=50)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Ligne {idx+1} : {text}")