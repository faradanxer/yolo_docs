from ultralytics import YOLO
import numpy as np
from math import atan, pi
from PIL import Image
import easyocr
import re

def extract_digits(image):
    numbers=''
    count=1
    reader = easyocr.Reader(['ru'], gpu=False,)
    while len(numbers) < 9:
        results = reader.readtext(np.array(image), detail=0)
        numbers = re.sub(r'\s+', '', (re.sub(r'[^a-zA-ZА-Яа-я0-9 ]', '', ' '.join(results))))
        if (len(numbers) > 9) or (count == 3):
            break
        else:
            image = image.rotate(90, expand=True)
            count += 1
            numbers = ''
    return numbers


def predict_text(im):
    model = YOLO('../models/text.pt')
    results = model(im)
    prediction = results[0].boxes
    coord = np.array(prediction.xyxy)
    x1, y1, x2, y2 = int(coord[0][0]), int(coord[0][1]), int(coord[0][2]), int(coord[0][3])
    crop_rectangle = (x1, y1, x2, y2)
    im = im.crop(crop_rectangle)
    text = extract_digits(im)
    
    return text

def split_to_seria_and_number(text):
    if len(text) == 0:
        return "", ""
    else:
        seria = text[:4]
        number = text[4:]
        return seria, number
