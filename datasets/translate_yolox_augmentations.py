import os
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image
import albumentations as A
from shapely.geometry import Polygon, box
from shapely.validation import explain_validity
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

# Base directory for the datasets
base_source_dir = "datasets/poop"

# Training and validation directories
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

# Output directories for processed data
output_dir = "datasets/coco"
train_output_dir = os.path.join(output_dir, "train2017")
val_output_dir = os.path.join(output_dir, "val2017")
annotations_dir = os.path.join(output_dir, "annotations")

os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)
os.makedirs(annotations_dir, exist_ok=True)

# Define the size of the resulting cropped image
cropped_image_size = (640, 640)

# Switch to turn augmentations on/off
augmentations_enabled = True

def create_coco_annotation(images, annotations):
    """Create COCO-style annotation dictionary."""
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

def augment_image(image, keypoints):
    """Apply augmentations to the image and keypoints."""
    if not augmentations_enabled:
        return image, keypoints

    # Define a set of augmentations
    # The scale and transform augmentations distort the polygons and break them
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.2),
        A.HueSaturationValue(p=0.2),
        # A.ElasticTransform(p=0.2)
    ], keypoint_params=A.KeypointParams(format='xy'))

    # Flatten the keypoints list to apply transformations
    flat_keypoints = [point for polygon in keypoints for point in polygon]

    # Apply augmentations
    augmented = transform(image=image, keypoints=flat_keypoints)

    # Reshape keypoints back into the original structure
    new_keypoints = []
    start = 0
    for polygon in keypoints:
        new_keypoints.append(augmented['keypoints'][start:start+len(polygon)])
        start += len(polygon)

    return augmented['image'], new_keypoints

def crop_and_resize(image, keypoints, padding=50, target_size=(416, 416)):
    """Crop and resize the image around the keypoints."""
    x_points = [int(p[0]) for polygon in keypoints for p in polygon]
    y_points = [int(p[1]) for polygon in keypoints for p in polygon]
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

    new_keypoints = []
    for polygon in keypoints:
        new_polygon = []
        for point in polygon:
            new_x = (point[0] - start_x) * target_size[0] / crop.shape[1]
            new_y = (point[1] - start_y) * target_size[1] / crop.shape[0]
            new_polygon.append([new_x, new_y])
        new_keypoints.append(new_polygon)

    return crop_resized, new_keypoints

def clip_polygons(polygons, crop_box):
    """Clip polygons to the crop box."""
    clipped_polygons = []
    for polygon in polygons:
        poly = Polygon(polygon)
        
        # Check if the polygon is valid
        if not poly.is_valid:
            print(f"Invalid polygon: {explain_validity(poly)}")
            poly = poly.buffer(0)  # Attempt to fix the polygon

        clipped_poly = poly.intersection(crop_box)
        
        if not clipped_poly.is_empty:
            if clipped_poly.geom_type == 'Polygon':
                clipped_polygons.append(list(clipped_poly.exterior.coords)[:-1])
            elif clipped_poly.geom_type == 'MultiPolygon':
                for p in clipped_poly:
                    clipped_polygons.append(list(p.exterior.coords)[:-1])
    return clipped_polygons

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    else:
        return obj

def process_image(json_file, output_dir, start_image_id, start_annotation_id):
    """Process a single image file and apply augmentations."""
    with open(json_file) as f:
        data = json.load(f)

    image_filename = data["imagePath"]
    source_image_path = Path(json_file).parent / image_filename

    if not source_image_path.exists():
        return None, start_image_id, start_annotation_id

    image = cv2.imread(str(source_image_path))
    polygons = [shape["points"] for shape in data["shapes"] if shape["label"] == "poop"]

    results = []
    image_id = start_image_id
    annotation_id = start_annotation_id

    for padding in [0, 50, 300]:
        if padding == 0:
            crop_resized, new_polygons = image, polygons
        else:
            crop_resized, new_polygons = crop_and_resize(image, polygons, padding=padding, target_size=cropped_image_size)
            crop_box = box(0, 0, cropped_image_size[0], cropped_image_size[1])
            new_polygons = clip_polygons(new_polygons, crop_box)

        crop_resized, new_polygons = augment_image(crop_resized, new_polygons)
        dest_image_path = Path(output_dir) / f"{image_id:012d}.jpg"
        cv2.imwrite(str(dest_image_path), crop_resized)

        image_info = {
            "id": int(image_id),
            "file_name": dest_image_path.name,
            "height": int(crop_resized.shape[0]),
            "width": int(crop_resized.shape[1]),
            "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "license": 0,
            "coco_url": "",
            "flickr_url": "",
        }
        results.append(image_info)

        for polygon in new_polygons:
            if len(polygon) < 3:
                continue
            x_points = [float(p[0]) for p in polygon]
            y_points = [float(p[1]) for p in polygon]
            xmin, xmax = min(x_points), max(x_points)
            ymin, ymax = min(y_points), max(y_points)
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

            annotation_info = {
                "id": int(annotation_id),
                "image_id": int(image_id),
                "category_id": 1,
                "segmentation": [list(np.concatenate(polygon).flat)],
                "area": float((xmax - xmin) * (ymax - ymin)),
                "bbox": [float(x) for x in bbox],
                "iscrowd": 0,
            }
            results.append(annotation_info)
            annotation_id += 1

        image_id += 1

    return results, image_id, annotation_id

def process_directory(source_dirs, output_dir):
    """Process all images in the specified directories."""
    image_id = 1
    annotation_id = 1
    images = []
    annotations = []
    original_image_count = 0

    json_files = []
    for source_dir in source_dirs:
        source_dir = os.path.join(base_source_dir, source_dir)
        json_files.extend(list(Path(source_dir).glob("*.json")))

    total_files = len(json_files)
    start_time = time.time()

    with ThreadPoolExecutor() as executor:
        futures = []
        chunk_size = len(json_files) // os.cpu_count()
        if chunk_size == 0:
            chunk_size = 1

        progress_bar = tqdm(total=total_files, desc="Processing images")
        for i in range(0, total_files, chunk_size):
            chunk = json_files[i:i + chunk_size]
            start_image_id = image_id + i * 3
            start_annotation_id = annotation_id + i * 3 * 2  # Rough estimation
            futures.append(executor.submit(process_chunk, chunk, output_dir, start_image_id, start_annotation_id, progress_bar))

        for future in as_completed(futures):
            result = future.result()
            if result:
                for r in result:
                    if "file_name" in r:
                        images.append(r)
                    else:
                        annotations.append(r)
                original_image_count += len(chunk)

        progress_bar.close()

    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_image = total_time / total_files
    print(f"Processing complete. {total_files} original images processed to produce {len(images)} images.")
    print(f"Total time: {total_time:.2f} seconds, Average time per image: {avg_time_per_image:.2f} seconds")

    return images, annotations, total_files, total_files

def process_chunk(chunk, output_dir, start_image_id, start_annotation_id, progress_bar):
    """Process a chunk of images in parallel."""
    results = []
    image_id = start_image_id
    annotation_id = start_annotation_id
    for json_file in chunk:
        result, new_image_id, new_annotation_id = process_image(json_file, output_dir, image_id, annotation_id)
        if result:
            results.extend(result)
            image_id = new_image_id
            annotation_id = new_annotation_id
        progress_bar.update(1)
    return results

# Process training directories
train_images, train_annotations, train_original_count, train_processed_count = process_directory(train_dirs, train_output_dir)
train_coco = create_coco_annotation(train_images, train_annotations)

# Save training annotations
with open(os.path.join(annotations_dir, "instances_train2017.json"), "w") as f:
    json.dump(convert_numpy_types(train_coco), f, ensure_ascii=False, indent=2)

# Process validation directory
val_images, val_annotations, val_original_count, val_processed_count = process_directory(val_dirs, val_output_dir)
val_coco = create_coco_annotation(val_images, val_annotations)

# Save validation annotations
with open(os.path.join(annotations_dir, "instances_val2017.json"), "w") as f:
    json.dump(convert_numpy_types(val_coco), f, ensure_ascii=False, indent=2)

# Print final summary
print(f"Processing complete. {train_original_count} original train images processed to produce {len(train_images)} images.")
print(f"Processing complete. {val_original_count} original validation images processed to produce {len(val_images)} images.")
