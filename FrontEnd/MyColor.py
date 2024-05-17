import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from detect_areas import detect_red_areas


model_color = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model_color.eval()

def detect_peach(image, threshold=0.1):
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image)
    with torch.no_grad():
        predictions = model_color([img_tensor])
    boxes = predictions[0]['boxes'].numpy()
    labels = predictions[0]['labels'].numpy()
    scores = predictions[0]['scores'].numpy()
    peach_boxes = boxes[(labels == 53) & (scores >= threshold)]
    return peach_boxes


def draw_text_with_background(draw, text, position, font, text_color, bg_color):
    """在带有背景的图像上绘制文本"""
    text_width, text_height = draw.textsize(text, font=font)
    x, y = position
    # 绘制背景框
    draw.rectangle([x, y, x + text_width, y + text_height], fill=bg_color)
    # 绘制文本
    draw.text((x, y), text, fill=text_color, font=font)


def color_detect(uploaded_file):
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    # 转换图像以适应模型输入
    peach_boxes = detect_peach(image_pil)
    for box in peach_boxes:
        x1, y1, x2, y2 = box.astype(int)
        peach_roi = image_cv2[y1:y2, x1:x2]
        red_area_masks = detect_red_areas(peach_roi)

        total_pixels = peach_roi.shape[0] * peach_roi.shape[1]
        total_red_pixels = 0

        text_offset_y = 0
        for name, mask_info in red_area_masks.items():
            mask = mask_info["mask"]
            color = mask_info["color"]
            red_pixels = np.sum(mask > 0)
            total_red_pixels += red_pixels
            red_area_ratio = (red_pixels / total_pixels) * 100

            result = cv2.addWeighted(peach_roi, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
            result[mask > 0] = color
            image_cv2[y1:y2, x1:x2] = result

            text = f"{name}占比: {red_area_ratio:.2f}%"
            img_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.truetype('simhei.ttf', 30)
            draw.text((x1, y1 - 40 - text_offset_y), text, font=font, fill=color)
            image_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            text_offset_y += 40
        
        total_red_area_ratio = (total_red_pixels / total_pixels) * 100

        text = f"是否成熟：成熟\n总红色区域占比: {total_red_area_ratio:.2f}%"
        img_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text((x1, y1 - 60 - text_offset_y), text, font=font, fill=(0, 0, 255, 0))
        image_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    return Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))