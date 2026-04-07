import cv2
import numpy as np

img = cv2.imread("tableau1.jpg")

if img is None:
    print("Erreur : image non trouvée")
else:
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
        print("Tableau non détecté — essaie avec une photo plus cadrée")
    else:
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
        print("Fait — output_binary.png généré")