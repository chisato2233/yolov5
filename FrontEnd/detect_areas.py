import cv2
import numpy as np

def rgb_to_hsv(r, g, b):
    rgb = np.array([[[r, g, b]]], dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return hsv[0, 0]

def detect_red_areas(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义几种不同的红色HSV范围
    red_ranges = [
        {"name": "light_red",  "lower": np.array([0, 50, 200]), "upper": np.array([80, 127, 255]), "color": (225, 228, 255)},
        {"name": "medium_red",  "lower": rgb_to_hsv(163,97,81)+np.array([0, 50, 100]), "upper": np.array([10, 255, 200]),  "color": (71,99,255)},
        {"name": "dark_red", "lower": np.array([0, 30, 30]), "upper": rgb_to_hsv(155, 123, 115) + np.array([10, 30,30]), "color":(92,92,205)},
    ]

    masks = {}
    for red_range in red_ranges:
        mask = cv2.inRange(hsv, red_range["lower"], red_range["upper"])
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        masks[red_range["name"]] = {"mask": mask, "color": red_range["color"]}

    return masks

def detect_yellow_areas(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义几种不同的黄色HSV范围
    yellow_ranges = [
        {"name": "light_yellow",  "lower": np.array([20, 100, 100]), "upper": np.array([30, 255, 255]), "color": (0, 255, 255)},
        {"name": "medium_yellow",  "lower": np.array([10, 100, 100]), "upper": np.array([30, 255, 255]),  "color": (0, 255, 255)},
        {"name": "dark_yellow", "lower": np.array([10, 100, 100]), "upper": np.array([30, 255, 255]), "color":(0, 255, 255)},
    ]

    masks = {}
    for yellow_range in yellow_ranges:
        mask = cv2.inRange(hsv, yellow_range["lower"], yellow_range["upper"])
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        masks[yellow_range["name"]] = {"mask": mask, "color": yellow_range["color"]}

    return masks