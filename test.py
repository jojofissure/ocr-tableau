import cv2
import numpy as np

img = cv2.imread("tableau.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Détection des bords du tableau
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

# Trouver le plus grand contour (= le tableau)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

for c in contours[:5]:
    epsilon = 0.02 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)
    if len(approx) == 4:  # on veut exactement 4 coins
        board_corners = approx
        break

# Correction perspective → A4 portrait (300 dpi)
W, H = 2480, 3508
pts_src = board_corners.reshape(4, 2).astype(np.float32)

# Trier les coins : top-left, top-right, bottom-right, bottom-left
s = pts_src.sum(axis=1)
diff = np.diff(pts_src, axis=1)
pts_ordered = np.float32([
    pts_src[np.argmin(s)],    # top-left
    pts_src[np.argmin(diff)], # top-right
    pts_src[np.argmax(s)],    # bottom-right
    pts_src[np.argmax(diff)]  # bottom-left
])

pts_dst = np.float32([[0,0],[W,0],[W,H],[0,H]])
M = cv2.getPerspectiveTransform(pts_ordered, pts_dst)
warped = cv2.warpPerspective(img, M, (W, H))

# Binarisation adaptative
gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
binary = cv2.adaptiveThreshold(
    gray_w, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 31, 10
)

cv2.imwrite("output_binary.png", binary)
print("Fait — regarde output_binary.png")
