import os
import fiftyone as fo
import fiftyone.zoo as foz

# Base directory for your COCO dataset
base_dir = "datasets/coco"

# Paths for the training dataset
train_dataset_dir = os.path.join(base_dir, "train2017")
train_annotations_path = os.path.join(base_dir, "annotations/instances_train2017.json")

# Paths for the validation dataset
val_dataset_dir = os.path.join(base_dir, "val2017")
val_annotations_path = os.path.join(base_dir, "annotations/instances_val2017.json")

# Load the training dataset
train_dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=train_dataset_dir,
    labels_path=train_annotations_path,
)

# Load the validation dataset
val_dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=val_dataset_dir,
    labels_path=val_annotations_path,
)

# Merge the datasets
combined_dataset = train_dataset.clone(name="combined_dataset")
combined_dataset.add_samples(val_dataset)

# Launch the FiftyOne app
session = fo.launch_app(combined_dataset, desktop=True)

# Keep the session open
session.wait()
