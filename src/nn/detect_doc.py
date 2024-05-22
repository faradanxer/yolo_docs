from ultralytics import YOLO
import numpy as np

classes = {
    0: ("driver_license", 1),
    1: ("driver_license", 2),
    3: ("personal_passport", 1),
    4: ("personal_passport", 2),
    6: ("vehicle_passport", 0),
    8: ("vehicle_certificate", 1),
    9: ("vehicle_certificate", 2)
}

    

def predict_class_proba_page_box(im):
    model = YOLO('../models/final.pt')
    results = model(im)
    prediction = results[0].boxes[0]
    print(int(prediction.cls))
    proba = prediction.conf[0].item()
    pair = classes.get(int(prediction.cls))

    cls = pair[0]
    page = pair[1]
    box = np.array(prediction.xyxy)

    return cls, proba, page, box

def crop_box(im, box):
    x1, y1, x2, y2 = int(box[0][0]), int(box[0][1]), int(box[0][2]), int(box[0][3])
    crop_rectangle = (x1, y1, x2, y2)
    im = im.crop(crop_rectangle)
    return im