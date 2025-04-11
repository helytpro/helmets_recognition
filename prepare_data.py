import json
import os
import shutil
import cv2
import re
from ultralytics import YOLO
import numpy as np

base_directory = "helmets"
images_folder = "images_yolo"
labels_folder = "labels_yolo"

DIST_THRESHOLD = 50

def get_box_from_kpt(key_points, width):
    key_points = key_points[:7, :]
    key_points = key_points[~np.all(key_points == 0, axis=1)]

    if len(key_points) == 0:
        return (0, 0, 0, 0)

    x_max = np.max(key_points[:, 0])
    x_min = np.min(key_points[:, 0])
    y_max = np.max(key_points[:, 1])
    y_min = np.min(key_points[:, 1])

    w = x_max - x_min
    h = y_max - y_min

    x_min = max(0, x_min - w / 3)
    x_max = min(width, x_max + w / 3)
    y_min = max(0, y_min - h)

    return tuple(map(int, (x_min, y_min, x_max, y_max)))


def distance(pt1, pt2):
    return int(((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** (0.5))


model = YOLO("yolo11m.pt")
model_pose = YOLO("yolov8n-pose.pt")

if (os.path.exists(images_folder)):
    shutil.rmtree(images_folder)
os.mkdir(images_folder)  # TODO: исправить этот колхоз

if (os.path.exists(labels_folder)):
    shutil.rmtree(labels_folder)
os.mkdir(labels_folder)

counter = 0

for root, dirs, files in os.walk(base_directory):
    for file in files:
        if file.endswith('.json'):
            filename = re.sub('.json', '', file)
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    img = cv2.imread(os.path.join(root, data['imagePath'].split("\\")[-1]))  # os.path.join(root, (filename + '.jpg'))
                    h, w, _ = img.shape
                    model_result = model_pose(img, verbose=False)[0].keypoints.xy.cpu().numpy()
                    if (not len(model_result) or (len(model_result) < len(data['shapes']))):
                        continue
                    for crop_dict in data['shapes']:
                        x1 = int(crop_dict['points'][0][0])
                        x2 = int(crop_dict['points'][1][0])
                        y1 = int(crop_dict['points'][0][1])
                        y2 = int(crop_dict['points'][1][1])
                        center_crop_orig = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        dist_inter = 1000.0
                        for result in model_result: 
                            box = get_box_from_kpt(result, w)
                            x3 = int(box[0] + abs(box[2] - box[0]) / 2)
                            y3 = int(box[1] + abs(box[3] - box[1]) / 2)
                            center_crop_yolo = (x3, y3)
                            centers_distance = distance(center_crop_orig, center_crop_yolo)
                            if (centers_distance < dist_inter):
                                box_target = box
                                dist_inter = centers_distance

                        with open(os.path.join(labels_folder, f'{counter}.txt'), 'w') as markup:
                            crop_yolo = img[box_target[1]:box_target[3], box_target[0]:box_target[2]]
                            cv2.imwrite(os.path.join(images_folder, f'{counter}.jpg'), crop_yolo)
                            label = crop_dict['label'].split('_')[-1]
                            markup.write(str(1)) if label == "on" else markup.write(str(0))
                        counter += 1
            except Exception as e:
                print(f"Ошибка при чтении файла {file_path}: {e}")

print(counter)

# повторный прогон (очищение от артефактов обработки)

counter_refact = 0
while counter_refact < counter:
    name_img = os.path.join(images_folder, f'{counter_refact}.jpg')
    name_lbl = os.path.join(labels_folder, f'{counter_refact}.txt')
    check_img = os.path.isfile(name_img)
    check_lbl = os.path.isfile(name_lbl)
    if not (check_img and check_lbl):
        os.remove(name_img) if check_img else print("img is already delete")
        os.remove(name_lbl) if check_lbl else print("lbl is already delete")

    counter_refact += 1