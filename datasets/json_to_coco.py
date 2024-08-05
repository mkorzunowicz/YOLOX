import os
import json
import cv2
from pathlib import Path
from collections import defaultdict

# CUDA_VISIBLE_DEVICES="" python tools/train.py -f yolox_config.py -d 1 -b 8 --no-aug
# set CUDA_VISIBLE_DEVICES=""
# python tools/train.py -f exps/example/custom/yolox_s.py -d 1 -b 8 -o -c trained/yolox_s_custom.pth
def convert_to_coco_format(dataset_path):
    coco = {"images": [], "annotations": [], "categories": []}
    category_id = 1
    annotation_id = 1
    categories = {}

    for date_folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, date_folder)
        if not os.path.isdir(folder_path):
            continue
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):
                image_path = os.path.join(folder_path, filename)
                json_path = image_path.replace(".jpg", ".json")

                if not os.path.exists(json_path):
                    continue

                image = cv2.imread(image_path)
                height, width, _ = image.shape

                image_info = {
                    "id": len(coco["images"]) + 1,
                    "file_name": os.path.relpath(image_path, dataset_path),
                    "height": height,
                    "width": width,
                }
                coco["images"].append(image_info)

                with open(json_path) as f:
                    annotations = json.load(f)
                    for annotation in annotations:
                        category_name = annotation["label"]
                        if category_name not in categories:
                            categories[category_name] = category_id
                            coco["categories"].append(
                                {"id": category_id, "name": category_name}
                            )
                            category_id += 1
                        category_id = categories[category_name]

                        x, y, w, h = annotation["bbox"]
                        annotation_info = {
                            "id": annotation_id,
                            "image_id": image_info["id"],
                            "category_id": category_id,
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "iscrowd": 0,
                        }
                        coco["annotations"].append(annotation_info)
                        annotation_id += 1

    return coco


dataset_path = "path/to/your/dataset"
coco_format_data = convert_to_coco_format(dataset_path)

with open("coco_annotations.json", "w") as f:
    json.dump(coco_format_data, f, indent=4)
