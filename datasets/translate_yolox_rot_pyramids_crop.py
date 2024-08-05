import os
import shutil
import json
import cv2
import numpy as np
import random
from pathlib import Path
from datetime import datetime
from PIL import Image

base_source_dir = "datasets/poop"

# Define source directories
train_dirs = [
    "poop-2021-09-20",
    "poop-2021-11-26",
    "poop-2021-12-27",
    "poop-2022-03-13-T152627",
    "poop-2022-04-02-T145512",
    "poop-2022-04-16-T135257",
    "poop-2022-05-26-T173650",
    "poop-2022-06-20-T235340",
    "poop-2022-07-16-T215017",
    "poop-2022-09-19-T153414",
    "poop-2022-11-23-T182537",
    "poop-2023-01-01-T171030",
    "poop-2023-03-11-T165018",
    "poop-2023-07-01-T160318",
    "poop-2023-08-22-T202656",
    "poop-2023-09-22-T180825",
    "poop-2023-10-15-T193631",
    "poop-2023-10-19-T212018",
    "poop-2023-12-19-T190904",
    "poop-2023-12-31-T214950",
    "poop-2024-01-31-T224154",
]
val_dirs = [
    "poop-2021-06-05",
    "poop-2021-06-20",
    "poop-2021-11-11",
    "poop-2022-01-27"
    "poop-2022-06-08-T132910",
    "poop-2023-04-16-T175739",
    "poop-2023-11-16-T154909",
    ]


output_dir = "datasets/coco_zoom"
train_output_dir = os.path.join(output_dir, "train2017")
val_output_dir = os.path.join(output_dir, "val2017")
annotations_dir = os.path.join(output_dir, "annotations")

os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)
os.makedirs(annotations_dir, exist_ok=True)

def create_coco_annotation(images, annotations):
    coco = {
        "info": {
            "description": "Poop Detection Dataset",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "",
            "date_created": datetime.now().strftime("%Y/%m/%d"),
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "poop", "supercategory": "object"},
        ],
    }
    return coco

def augment_image(image, points):
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        image_width = image.shape[1]
        points = [[image_width - point[0], point[1]] for point in points]
    return image, points

def crop_and_resize(image, points, padding=50, target_size=(416, 416)):
    x_points = [int(p[0]) for p in points]
    y_points = [int(p[1]) for p in points]
    xmin, xmax = min(x_points), max(x_points)
    ymin, ymax = min(y_points), max(y_points)
    
    center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2
    crop_size = max(xmax - xmin, ymax - ymin, target_size[0])

    start_x = max(0, center_x - crop_size // 2 - padding)
    start_y = max(0, center_y - crop_size // 2 - padding)
    end_x = min(image.shape[1], center_x + crop_size // 2 + padding)
    end_y = min(image.shape[0], center_y + crop_size // 2 + padding)

    if end_x - start_x < crop_size + 2 * padding:
        start_x = max(0, start_x - (crop_size + 2 * padding - (end_x - start_x)))
        end_x = min(image.shape[1], start_x + crop_size + 2 * padding)
    if end_y - start_y < crop_size + 2 * padding:
        start_y = max(0, start_y - (crop_size + 2 * padding - (end_y - start_y)))
        end_y = min(image.shape[0], start_y + crop_size + 2 * padding)

    crop = image[start_y:end_y, start_x:end_x]
    crop_resized = cv2.resize(crop, target_size)

    new_points = []
    for point in points:
        new_x = (point[0] - start_x) * target_size[0] / crop.shape[1]
        new_y = (point[1] - start_y) * target_size[1] / crop.shape[0]
        new_points.append([new_x, new_y])

    return crop_resized, new_points

def process_directory(source_dirs, output_dir):
    image_id = 1
    annotation_id = 1
    images = []
    annotations = []
    original_image_count = 0
    processed_image_count = 0

    for source_dir in source_dirs:
        source_dir = os.path.join(base_source_dir, source_dir)
        json_files = list(Path(source_dir).glob("*.json"))
        print(f"Processing directory {source_dir} with {len(json_files)} images")

        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)

            image_filename = data["imagePath"]
            source_image_path = Path(source_dir) / image_filename

            if not source_image_path.exists():
                continue

            image = cv2.imread(str(source_image_path))
            original_image_count += 1
            
            for shape in data["shapes"]:
                if shape["label"] == "poop":
                    points = shape["points"]

                    for padding in [0, 50, 300]:
                        if padding == 0:
                            crop_resized, new_points = image, points
                        else:
                            crop_resized, new_points = crop_and_resize(image, points, padding=padding)
                        
                        crop_resized, new_points = augment_image(crop_resized, new_points)

                        dest_image_path = Path(output_dir) / f"{image_id:012d}.jpg"
                        cv2.imwrite(str(dest_image_path), crop_resized)

                        image_info = {
                            "id": image_id,
                            "file_name": dest_image_path.name,
                            "height": crop_resized.shape[0],
                            "width": crop_resized.shape[1],
                            "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "license": 0,
                            "coco_url": "",
                            "flickr_url": "",
                        }
                        images.append(image_info)

                        x_points = [p[0] for p in new_points]
                        y_points = [p[1] for p in new_points]
                        xmin, xmax = min(x_points), max(x_points)
                        ymin, ymax = min(y_points), max(y_points)
                        bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

                        annotation_info = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "segmentation": [sum(new_points, [])],
                            "area": (xmax - xmin) * (ymax - ymin),
                            "bbox": bbox,
                            "iscrowd": 0,
                        }
                        annotations.append(annotation_info)
                        annotation_id += 1

                        image_id += 1
                        processed_image_count += 1

    return images, annotations, original_image_count, processed_image_count

train_images, train_annotations, train_original_count, train_processed_count = process_directory(train_dirs, train_output_dir)
train_coco = create_coco_annotation(train_images, train_annotations)

with open(os.path.join(annotations_dir, "instances_train2017.json"), "w") as f:
    json.dump(train_coco, f)

val_images, val_annotations, val_original_count, val_processed_count = process_directory(val_dirs, val_output_dir)
val_coco = create_coco_annotation(val_images, val_annotations)

with open(os.path.join(annotations_dir, "instances_val2017.json"), "w") as f:
    json.dump(val_coco, f)

print(f"Processing complete. {train_original_count} original train images processed to produce {train_processed_count} images.")
print(f"Processing complete. {val_original_count} original validation images processed to produce {val_processed_count} images.")
