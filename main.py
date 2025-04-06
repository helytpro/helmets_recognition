import json
from PIL import Image, ImageDraw
import os, shutil
import cv2
import time
import re
from ultralytics import YOLO


def distance(pt1, pt2):
    return int(((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**(0.5))


base_directory = r"C:\Users\Vladimir\helmets_recognition\helmets"
images_folder = "images_yolo"
labels_folder = "labels_yolo"

model = YOLO("yolo11m.pt")

if (os.path.exists(images_folder)):
    shutil.rmtree(images_folder)
os.mkdir(images_folder) # TODO: исправить этот колхоз


if (os.path.exists(labels_folder)):
    shutil.rmtree(labels_folder)
os.mkdir(labels_folder)

counter = 0
skip_counter = 0

for root, dirs, files in os.walk(base_directory):
    for file in files:
        if file.endswith('.json'):
            filename = re.sub('.json', '', file)
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    img = cv2.imread(os.path.join(root, data['imagePath'].split("\\")[-1])) # os.path.join(root, (filename + '.jpg'))
                    if img is not None:
                        model_result = model(img)
                    else:
                        skip_counter += 1
                        continue
                    for crop_dict in data['shapes']:
                        x1 = int(crop_dict['points'][0][0])
                        x2 = int(crop_dict['points'][1][0])
                        y1 = int(crop_dict['points'][0][1])
                        y2 = int(crop_dict['points'][1][1])
                        center_crop_orig = (int((x1 + x2)/2), int((y1 + y2)/2))
                        dist_inter = 1000.0
                        for result in model_result: # МОЖНО СКИПНУТЬ, ЕСЛИ РЕЗУЛЬТАТ ПО ОДНОЙ КАРТИНКЕ, В ТЕОРИИ
                            boxes = result.boxes.xyxy
                            confidences = result.boxes.conf
                            class_ids = result.boxes.cls
                            for box, conf, cls in zip(boxes, confidences, class_ids):
                                if not cls and conf < 0.5: # метка класса - не 0 ("person")
                                    continue
                                x3 = int(box[0] + (box[2] - box[0]) / 2)
                                y3 = int(box[1] + (box[3] - box[1]) / 6)
                                center_crop_yolo = (x3, y3)
                                centers_distance = distance(center_crop_orig, center_crop_yolo)
                                if (centers_distance < dist_inter):
                                    box_target = box
                                    dist_inter = centers_distance
                        xx1, xx2 = int(box_target[0]), int(box_target[2])
                        yy1, yy2 = int(box_target[1]), int(box_target[1] + (box_target[3] - box_target[1])/3)
                        crop_yolo = img[yy1:yy2, xx1:xx2]
                        cv2.imwrite(os.path.join(images_folder, f'{counter}.jpg'), crop_yolo)
                        with open (os.path.join(labels_folder, f'{counter}.txt'), 'w') as markup:
                            label = crop_dict['label'].split('_')[-1]
                            markup.write(str(1)) if label == "on" else markup.write(str(0))
                        counter += 1
            except Exception as e:
                print(f"Ошибка при чтении файла {file_path}: {e}")

print(counter)
print(skip_counter)