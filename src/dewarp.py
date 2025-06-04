import cv2
import numpy as np


image = cv2.imread("../data/initial_attempts/img028.jpg")
if image is None:
   raise FileNotFoundError("Can't load the image.")


h, w = image.shape[:2]
center_x = w // 2


warp_width = w // 6


half_warp = warp_width // 2
start_x = center_x - half_warp
end_x = center_x + half_warp


# debugging
print(f"Image width: {w}")
print(f"Warp width: {warp_width}")
print(f"Start_x: {start_x}, End_x: {end_x}")


left = image[:, :start_x]
middle = image[:, start_x:end_x]
right = image[:, end_x:]


# warping
src_pts = np.float32([
   [0, 0],
   [warp_width, 0],
   [warp_width, h],
   [0, h]
])


curve = 50

dst_pts = np.float32([
   [-curve, 0],
   [warp_width + curve, 0],
   [warp_width + curve, h],
   [-curve, h]
])


M = cv2.getPerspectiveTransform(src_pts, dst_pts)
middle_warped = cv2.warpPerspective(middle, M, (warp_width, h))



output = np.hstack([left, middle_warped, right])


cv2.imwrite("../data/initial_attempts/dewarped_center_fixed.jpg", output)
print("Saved in dewarped_center_fixed.jpg")