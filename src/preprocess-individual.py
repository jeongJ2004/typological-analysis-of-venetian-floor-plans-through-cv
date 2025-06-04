import os
import cv2
import numpy as np
import easyocr

# OCR init
reader = easyocr.Reader(['en'], gpu=False)

def load_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    return img

def remove_rivers(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 30, 30])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    img[mask == 255] = [255, 255, 255]
    return img

# def remove_text_with_easyocr(img_color):
#     results = reader.readtext(img_color)
#     for (bbox, text, conf) in results:
#         pts = np.array(bbox).astype(np.int32)
#         x_min = min(p[0] for p in pts)
#         x_max = max(p[0] for p in pts)
#         y_min = min(p[1] for p in pts)
#         y_max = max(p[1] for p in pts)
#         width = x_max - x_min
#         height = y_max - y_min
#         aspect_ratio = max(width / (height + 1e-5), height / (width + 1e-5))
#         if width < 3 or height < 3:
#             continue
#         if aspect_ratio > 20:
#             continue
#         if conf < 0.3:
#             continue
#         cv2.rectangle(img_color, (x_min, y_min), (x_max, y_max), (255, 255, 255), -1)
#     return img_color

def preprocess_image(img, center_margin=0):
    # img_no_text = remove_text_with_easyocr(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_enhanced = cv2.convertScaleAbs(gray, alpha=1.2, beta=0)
    height, width = gray.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    center_x_start = max(0, width // 2 - center_margin)
    center_x_end = min(width, width // 2 + center_margin)
    mask[:, center_x_start:center_x_end] = 255
    center_region = cv2.convertScaleAbs(gray_enhanced, alpha=1.5, beta=50)
    result = gray_enhanced.copy()
    result[mask == 255] = center_region[mask == 255]
    blurred = cv2.GaussianBlur(result, (5, 5), 0)
    return blurred

def apply_threshold(img, center_margin=0, center_threshold=140, default_threshold=140):
    height, width = img.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    center_x_start = max(0, width // 2 - center_margin)
    center_x_end = min(width, width // 2 + center_margin)
    mask[:, center_x_start:center_x_end] = 255
    _, binary_default = cv2.threshold(img, default_threshold, 255, cv2.THRESH_BINARY)
    _, binary_center = cv2.threshold(img, center_threshold, 255, cv2.THRESH_BINARY)
    binary = binary_default.copy()
    binary[mask == 255] = binary_center[mask == 255]
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return binary

def process_one_image(img_path, output_path):
    print(f"🔍 Processing: {img_path}")
    img = load_image(img_path)
    img = remove_rivers(img)
    img = preprocess_image(img)
    binary_img = apply_threshold(img)
    cv2.imwrite(output_path, binary_img)
    print(f"✅ Saved result to: {output_path}")

if __name__ == '__main__':
    img_path = "../data/FLOORPLANS_Cut/img0131.jpg"  # input path
    output_path = "../results/thresholded-imgs-cut/img0131.jpg"  # output path
    process_one_image(img_path, output_path)
