from PIL import Image
import numpy as np
import matplotlib.colors

def rgb_to_hsv(r, g, b):
    # 使用numpy来计算HSV值
    rgb = np.uint8([[[r, g, b]]])
    hsv = matplotlib.colors.rgb_to_hsv(rgb/255.0)*255
    return hsv[0, 0].astype(int)

red_ranges = [
    {"name": "light_red",  "lower": np.array([0, 50, 200]), "upper": np.array([80, 127, 255]), "color": (225, 228, 255)},
    {"name": "medium_red",  "lower": rgb_to_hsv(163,97,81)+np.array([0, 50, 100]), "upper": np.array([10, 255, 200]),  "color": (71,99,255)},
    {"name": "dark_red", "lower": np.array([0, 30, 30]), "upper": rgb_to_hsv(155, 123, 115) + np.array([10, 30,30]), "color":(92,92,205)},
]


#定义几种不同的黄色HSV范围
yellow_ranges = [
    {"name": "light_yellow",  "lower": np.array([20, 100, 100]), "upper": np.array([30, 255, 255]), "color": (0, 255, 255)},
    {"name": "medium_yellow",  "lower": np.array([10, 100, 100]), "upper": np.array([30, 255, 255]),  "color": (0, 255, 255)},
    {"name": "dark_yellow", "lower": np.array([10, 100, 100]), "upper": np.array([30, 255, 255]), "color":(0, 255, 255)},
]

def detect_red_areas(image):
    image = image.convert('RGB')
    image_np = np.array(image)
    hsv = matplotlib.colors.rgb_to_hsv(image_np/255.0)*255
    masks = {}

    for red_range in red_ranges:
        lower, upper = red_range["lower"], red_range["upper"]
        mask = ((hsv[:,:,0] >= lower[0]) & (hsv[:,:,0] <= upper[0]) &
                (hsv[:,:,1] >= lower[1]) & (hsv[:,:,1] <= upper[1]) &
                (hsv[:,:,2] >= lower[2]) & (hsv[:,:,2] <= upper[2]))
        mask = np.uint8(mask) * 255
        # 使用Pillow将遮罩转换为图像
        mask_image = Image.fromarray(mask)
    return masks

def detect_yellow_areas(image):
    image = image.convert('RGB')
    image_np = np.array(image)
    hsv = matplotlib.colors.rgb_to_hsv(image_np/255.0)*255
    masks = {}


    for red_range in yellow_ranges:
        lower, upper = red_range["lower"], red_range["upper"]
        mask = ((hsv[:,:,0] >= lower[0]) & (hsv[:,:,0] <= upper[0]) &
                (hsv[:,:,1] >= lower[1]) & (hsv[:,:,1] <= upper[1]) &
                (hsv[:,:,2] >= lower[2]) & (hsv[:,:,2] <= upper[2]))
        mask = np.uint8(mask) * 255
        # 使用Pillow将遮罩转换为图像
        mask_image = Image.fromarray(mask)
    return masks