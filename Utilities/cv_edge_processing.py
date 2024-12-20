import cv2
import numpy as np


def extract_blue(screen_in):
    # 将图像从 BGR 颜色空间转换为 HSV 颜色空间
    hsv = cv2.cvtColor(screen_in, cv2.COLOR_BGR2HSV)

    # 定义蓝色的 HSV 范围
    lower_blue = np.array([100, 10, 0])
    upper_blue = np.array([240, 200, 180])
    

    # 创建蓝色的掩码
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 使用掩码提取蓝色部分
    blue_extracted = cv2.bitwise_and(screen_in, screen_in, mask=mask)

    return blue_extracted  

def edge_processing(screen_in, resize_width=160, resize_height=90):
    # resize the screen at first to reduce the performance pressure
    screen_resized = cv2.resize(screen_in, (resize_width, resize_height))
    # screen_resized = extract_blue(screen_resized)

    # apply erode and dilate to remove noise
    kernel = np.ones((2, 2), np.uint8)
    # screen_eroded = cv2.erode(screen_resized, kernel, iterations=1)
    # screen_dilated = cv2.dilate(screen_eroded, kernel, iterations=1)

    # edge detection
    edges = cv2.Canny(screen_resized, 200, 255)
    # edges = cv2.cvtColor(screen_resized, cv2.COLOR_BGR2GRAY)

    # define the ROI axis
    left_mid = (0, resize_height // 2)
    right_mid = (resize_width - 1, resize_height // 2)
    top_mid = (resize_width // 2, 0)
    bottom_left = (0, resize_height - 1)
    bottom_right = (resize_width - 1, resize_height - 1)

    # creating mask for masking ROI
    mask = np.zeros(screen_resized.shape[:2], dtype=np.uint8)
    roi_corners = np.array([[left_mid, top_mid, right_mid, bottom_right, bottom_left]], dtype=np.int32)
    cv2.fillPoly(mask, roi_corners, 255)

    # apply the ROI mask
    screen_out = cv2.bitwise_and(edges, edges, mask=mask)

    return screen_out
