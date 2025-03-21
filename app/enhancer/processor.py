import cv2
import numpy as np
from PIL import Image
from .upscaler import upscale_image

def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def adjust_saturation(img, factor=1.2):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def sharpen(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def enhance_image(input_path, output_path, scale=2):
    img = preprocess_image(input_path)
    img = enhance_contrast(img)
    img = adjust_saturation(img)
    img = sharpen(img)
    img = upscale_image(img, scale=scale)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Image.fromarray(img_rgb).save(output_path)
