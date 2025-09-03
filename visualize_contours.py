import cv2
import numpy as np

# Load image
img = cv2.imread(r"C:\Users\Yoked\Desktop\Rice project\Rice_Image_Dataset\Arborio\Arborio (1).jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold to get binary mask
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(contours, key=cv2.contourArea)  # largest contour

# Draw contour
img_contour = img.copy()
cv2.drawContours(img_contour, [cnt], -1, (0, 255, 0), 2)

# Fit ellipse (needs at least 5 points)
if len(cnt) >= 5:
    ellipse = cv2.fitEllipse(cnt)
    cv2.ellipse(img_contour, ellipse, (255, 0, 0), 2)

# Show results
cv2.imshow("Contour + Ellipse", img_contour)
cv2.waitKey(0)
cv2.destroyAllWindows()