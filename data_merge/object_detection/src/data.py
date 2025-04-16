import os
import yaml
import random
import json
import shutil
import zipfile
import glob
import cv2
from PIL import Image
import pybboxes as pbx
from collections import OrderedDict
from typing import List, Dict
from collections import defaultdict


def remove_unlisted_images(image_dir, filtered_dataset):
    """
    Removes all images from the specified directory that are not listed in the filtered dataset (after class filtration).

    Parameters:
    - image_dir (str): The directory containing the images to be filtered.
    - filtered_dataset (dict): Dictionary containing filtered 'images', 'annotations', and 'categories'.

    Description:
    This function scans the provided image directory and removes any image files that are not
    present in the `filtered_dataset['images']` list.
    """

    retained_images = {image_info['file_name'] for image_info in filtered_dataset['images']}

    all_images = set(os.listdir(image_dir))

    unlisted_images = all_images - retained_images

    for image_filename in unlisted_images:
        image_path = os.path.join(image_dir, image_filename)
        os.remove(image_path)

    print(f"Cleanup complete. {len(unlisted_images)} unlisted images have been removed.")


# def coco_to_yolo(coco_annotation_path, images_dir, output_dir, yaml_output_path=None):
#     """
#     Converts COCO format annotations to YOLO format and generates a YAML file describing the dataset.

#     Parameters:
#     - coco_annotation_path (str): Path to the COCO annotation JSON file.
#     - images_dir (str): Directory containing the images referenced in the COCO annotations.
#     - output_dir (str): Directory where YOLO format annotation files will be saved.
#     - yaml_output_path (str, optional): Path to save the generated YAML file. If None, it will save in the `output_dir`.

#     Description:
#     This function reads annotations from a COCO format JSON file and converts them to the YOLO format.
#     It also generates a `data.yaml` file that describes the dataset.
#     In YOLO format, each image has a corresponding .txt file containing the bounding box annotations.
#     Each line in the .txt file follows the format:

#     class_id center_x center_y width height

#     Where:
#     - `class_id` is the ID of the object class (starting from 0).
#     - `center_x`, `center_y` are the normalized coordinates of the bounding box center.
#     - `width`, `height` are the normalized dimensions of the bounding box.

#     The coordinates and dimensions are normalized with respect to the image width and height.

#     The function processes each annotation in the COCO dataset, converts the bounding box to YOLO format
#     using the `pybboxes` library, and saves the results in a .txt file named after the corresponding image.

#     """


#     with open(coco_annotation_path, 'r') as f:
#         coco_data = json.load(f)

#     class_names = [category['name'] for category in coco_data['categories']]

#     os.makedirs(output_dir, exist_ok=True)

#     category_id_to_yolo_id = {category['id']: idx for idx, category in enumerate(coco_data['categories'])}

#     # Process each annotation
#     for annotation in coco_data['annotations']:
#         image_id = annotation['image_id']
#         category_id = annotation['category_id']
#         bbox = annotation['bbox']

#         # Get image file name and size
#         image_info = next(image for image in coco_data['images'] if image['id'] == image_id)
#         image_filename = image_info['file_name']
#         image_path = os.path.join(images_dir, image_filename)

#         if not os.path.exists(image_path):
#             print(f"Warning: Label file not found for image: {image_path}")
#             continue

#         with Image.open(image_path) as img:
#             image_size = img.size  # (width, height)

#         yolo_bbox = pbx.convert_bbox(bbox, from_type="coco", to_type="yolo", image_size=image_size)

#         yolo_class_id = category_id_to_yolo_id[category_id]

#         yolo_annotation_line = f"{yolo_class_id} {' '.join(map(str, yolo_bbox))}\n"

#         yolo_annotation_file = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}.txt")

#         with open(yolo_annotation_file, 'a') as yolo_file:
#             yolo_file.write(yolo_annotation_line)

#     yaml_content = {
#         'nc': len(class_names),                     # Number of classes
#         'names': class_names                        # List of class names
#     }

#     # Determine where to save the YAML file
#     if yaml_output_path is None:
#         yaml_output_path = os.path.join(output_dir, 'data.yaml')

#     # Save the YAML file
#     with open(yaml_output_path, 'w') as yaml_file:
#         yaml.dump(yaml_content, yaml_file)

#     print(f"YOLO annotations and data.yaml have been saved to {output_dir}")


def create_filtered_coco_dataset(original_annotation_path, original_image_dir, filtered_annotation_path, filtered_image_dir):
    """
    Creates a filtered COCO dataset containing only specified categories.

    Parameters:
    - original_annotation_path (str): Path to the original COCO annotation JSON file.
    - original_image_dir (str): Path to the original image directory.
    - filtered_annotation_path (str): Path where the filtered annotation JSON will be saved.
    - filtered_image_dir (str): Path to the directory where filtered images will be copied.

    Description:
    This function filters the original COCO annotations to keep only specified categories,
    copies the corresponding images to a new directory, and saves the filtered annotations.
    """
    
    # Ensure the directory for filtered annotations exists
    filtered_annotation_dir = os.path.dirname(filtered_annotation_path)
    os.makedirs(filtered_annotation_dir, exist_ok=True)

    # Ensure the filtered image directory exists
    os.makedirs(filtered_image_dir, exist_ok=True)

    # Load the filtered annotations
    with open(original_annotation_path, 'r') as f:
        filtered_data = json.load(f)

    # Extract relevant image file names
    retained_images = {image_info['file_name'] for image_info in filtered_data['images']}

    # Copy retained images to the filtered image directory
    for image_info in filtered_data['images']:
        src_image_path = os.path.join(original_image_dir, image_info['file_name'])
        dst_image_path = os.path.join(filtered_image_dir, image_info['file_name'])

        if os.path.exists(src_image_path):
            shutil.copy2(src_image_path, dst_image_path)
        else:
            print(f"Warning: Image {image_info['file_name']} not found in {original_image_dir}")

    # Save the filtered annotations
    with open(filtered_annotation_path, 'w') as f:
        json.dump(filtered_data, f)

    print(f"Filtered dataset created successfully at {filtered_image_dir} and {filtered_annotation_path}")


def coco_to_yolo(coco_annotation_path, images_dir, output_dir, yaml_output_path=None):
    """
    Converts COCO format annotations to YOLO format and generates a YAML file describing the dataset.
    
    Parameters:
    - coco_annotation_path (str): Path to the COCO annotation JSON file.
    - images_dir (str): Directory containing the images referenced in the COCO annotations.
    - output_dir (str): Directory where YOLO format annotation files will be saved.
    - yaml_output_path (str, optional): Path to save the generated YAML file. If None, it will save in the `output_dir`.
    
    Description:
    This function reads annotations from a COCO format JSON file and converts them to the YOLO format.
    It also generates a `data.yaml` file that describes the dataset.
    In YOLO format, each image has a corresponding .txt file containing the bounding box annotations.
    Each line in the .txt file follows the format:
    
    class_id center_x center_y width height
    
    Where:
    - `class_id` is the ID of the object class (starting from 0).
    - `center_x`, `center_y` are the normalized coordinates of the bounding box center.
    - `width`, `height` are the normalized dimensions of the bounding box.
    
    The coordinates and dimensions are normalized with respect to the image width and height.
    
    The function processes each image in the COCO dataset, converts its bounding boxes to YOLO format,
    and saves the results in a .txt file named after the corresponding image.
    """
    
    # Load COCO annotations
    try:
        with open(coco_annotation_path, 'r') as f:
            coco_data = json.load(f)
    except Exception as e:
        print(f"Error: Unable to load COCO annotations from {coco_annotation_path}. Error: {e}")
        return
    
    # Extract class names and create a mapping from category_id to yolo_class_id
    class_names = [category['name'] for category in coco_data['categories']]
    category_id_to_yolo_id = {category['id']: idx for idx, category in enumerate(coco_data['categories'])}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a mapping from image_id to its annotations for faster lookup
    image_id_to_annotations = defaultdict(list)
    for annotation in coco_data['annotations']:
        image_id_to_annotations[annotation['image_id']].append(annotation)
    
    # Initialize a list to track removed images
    removed_images = []
    
    # Iterate over each image in the dataset
    for image_info in coco_data['images']:
        image_id = image_info['id']
        image_filename = image_info['file_name']
        image_path = os.path.join(images_dir, image_filename)
    
        # Define the path for the YOLO annotation file
        yolo_annotation_file = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}.txt")
    
        # Check if the image file exists
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}. Removing image entry.")
            removed_images.append(image_filename)
            continue
    
        # Open the image to get its size
        try:
            with Image.open(image_path) as img:
                image_width, image_height = img.size
        except Exception as e:
            print(f"Error: Unable to open image {image_path}. Error: {e}. Removing image.")
            removed_images.append(image_filename)
            # Remove the problematic image file if it's partially corrupted
            try:
                os.remove(image_path)
                print(f"Removed corrupted image file: {image_path}")
            except Exception as remove_e:
                print(f"Error: Failed to remove corrupted image {image_path}. Error: {remove_e}")
            continue
    
        # Retrieve annotations for this image
        annotations = image_id_to_annotations.get(image_id, [])
    
        # Initialize a list to hold YOLO annotations for this image
        yolo_annotations = []
        problematic_bbox_found = False
    
        for annotation in annotations:
            category_id = annotation['category_id']
            bbox = annotation['bbox']  # [x, y, width, height] in COCO format
    
            # Validate bbox dimensions
            if bbox[2] <= 0 or bbox[3] <= 0:
                print(f"Warning: Invalid bbox with non-positive width or height in image {image_filename}. Removing image.")
                problematic_bbox_found = True
                break  # Exit the annotation loop to remove the image
    
            # Convert COCO bbox to YOLO format using pybboxes
            try:
                yolo_bbox = pbx.convert_bbox(
                    bbox, 
                    from_type="coco", 
                    to_type="yolo", 
                    image_size=(image_width, image_height)
                )
            except Exception as e:
                print(f"Warning: Failed to convert bbox for image {image_filename}. Error: {e}. Removing image.")
                problematic_bbox_found = True
                break  # Exit the annotation loop to remove the image
    
            yolo_class_id = category_id_to_yolo_id.get(category_id)
            if yolo_class_id is None:
                print(f"Warning: Category ID {category_id} not found in categories for image {image_filename}. Removing image.")
                problematic_bbox_found = True
                break  # Exit the annotation loop to remove the image
    
            # Prepare the annotation line with six decimal places for precision
            yolo_annotation_line = f"{yolo_class_id} {' '.join([f'{coord:.6f}' for coord in yolo_bbox])}\n"
            yolo_annotations.append(yolo_annotation_line)
    
        if problematic_bbox_found:
            # Remove the problematic image file
            try:
                os.remove(image_path)
                print(f"Removed problematic image file: {image_path}")
                removed_images.append(image_filename)
            except Exception as remove_e:
                print(f"Error: Failed to remove image {image_path}. Error: {remove_e}")
            continue  # Skip writing annotations for this image
    
        # Write the YOLO annotations to the .txt file
        try:
            with open(yolo_annotation_file, 'w') as yolo_file:
                yolo_file.writelines(yolo_annotations)
        except Exception as e:
            print(f"Error: Failed to write YOLO annotation file {yolo_annotation_file}. Error: {e}")
            # Optionally, remove the image if annotations can't be written
            try:
                os.remove(image_path)
                print(f"Removed image due to annotation write failure: {image_path}")
                removed_images.append(image_filename)
            except Exception as remove_e:
                print(f"Error: Failed to remove image {image_path}. Error: {remove_e}")
    
    # Generate the YAML content
    yaml_content = {
        'nc': len(class_names),                     # Number of classes
        'names': class_names,                       # List of class names
        'removed_images': removed_images            # List of removed image filenames
    }
    
    # Determine where to save the YAML file
    if yaml_output_path is None:
        yaml_output_path = os.path.join(output_dir, 'data.yaml')
    
    # Save the YAML file
    try:
        with open(yaml_output_path, 'w') as yaml_file:
            yaml.dump(yaml_content, yaml_file)
        print(f"YOLO annotations and data.yaml have been saved to {output_dir}")
        if removed_images:
            print(f"Total removed images: {len(removed_images)}")
    except Exception as e:
        print(f"Error: Failed to write YAML file {yaml_output_path}. Error: {e}")
        

def merge_data_yamls(yaml_paths, merged_yaml_path, merged_train_path, merged_val_path, merged_test_path=None, mappings_yaml_path=None):
    """
    Merges multiple data.yaml files into a single data.yaml with unified class indices.
    Also generates a separate YAML file containing the class mappings for verification.

    Parameters:
    - yaml_paths (list): List of paths to the original data.yaml files.
    - merged_yaml_path (str): Path where the merged data.yaml will be saved.
    - merged_train_path (str): Path to the merged train images directory.
    - merged_val_path (str): Path to the merged validation images directory.
    - merged_test_path (str, optional): Path to the merged test images directory.
    - mappings_yaml_path (str, optional): Path where the class mappings YAML will be saved.

    Returns:
    - merged_classes (list): List of unified class names.
    - mappings (dict): Dictionary mapping each original yaml to its class index mapping.
    """
    # OrderedDict to maintain insertion order and ensure uniqueness
    unified_classes = OrderedDict()
    mappings = {}  # To store mapping for each dataset

    for yaml_path in yaml_paths:
        if not os.path.exists(yaml_path):
            print(f"Error: The file {yaml_path} does not exist. Skipping.")
            continue

        with open(yaml_path, 'r') as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(f"Error parsing {yaml_path}: {exc}")
                continue

        original_classes = data.get('names', [])
        if not original_classes:
            print(f"Warning: No classes found in {yaml_path}. Skipping.")
            continue

        # Use the full yaml path as the dataset name for uniqueness
        dataset_name = yaml_path

        mappings[dataset_name] = {}

        for idx, class_name in enumerate(original_classes):
            normalized_name = class_name.lower()
            if normalized_name not in unified_classes:
                unified_classes[normalized_name] = len(unified_classes)  # Assign next available index
            # Map original index to unified index
            mappings[dataset_name][idx] = unified_classes[normalized_name]

    # Create the merged data.yaml structure
    merged_data = {
        'train': merged_train_path,
        'val': merged_val_path,
        'nc': len(unified_classes),
        'names': list(unified_classes.keys())
    }

    if merged_test_path and os.path.exists(merged_test_path):
        merged_data['test'] = merged_test_path

    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(merged_yaml_path), exist_ok=True)

    # Write the merged data.yaml
    with open(merged_yaml_path, 'w') as f:
        yaml.dump(merged_data, f)

    # Print the merged classes
    print("\nMerged Classes:")
    for idx, class_name in enumerate(merged_data['names']):
        print(f"{idx}: {class_name}")

    # Print the class mappings per dataset
    print("\nClass Mappings per Dataset:")

    for dataset, mapping in mappings.items():
        print(f"\nDataset: {dataset}")
        for original_idx, new_idx in mapping.items():
            print(f"  Original Index {original_idx} -> New Index {new_idx}")

    # Optionally, save the mappings to a separate YAML file
    if mappings_yaml_path:
        mappings_data = {}
        for dataset, mapping in mappings.items():
            mappings_data[dataset] = {str(k): v for k, v in mapping.items()}
        with open(mappings_yaml_path, 'w') as f:
            yaml.dump(mappings_data, f)
        print(f"\nClass mappings have been saved to {mappings_yaml_path}")

    return list(unified_classes.keys()), mappings

def merge_datasets(merge_datasets: List[Dict], mapping_yaml: str, merge_dataset_params: Dict):
    """
    Merges multiple YOLO-formatted datasets into a single dataset.

    Parameters:
    - merge_datasets (list): List of dictionaries, each containing paths for a dataset.
    - mapping_yaml (str): Path to the class_mappings.yaml file.
    - merge_dataset_params (dict): Dictionary specifying paths for the merged dataset.
    
    Returns:
    - None
    """

    # Load class mappings
    if not os.path.exists(mapping_yaml):
        print(f"Error: Mapping YAML file '{mapping_yaml}' does not exist.")
        return

    with open(mapping_yaml, 'r') as f:
        try:
            class_mappings = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(f"Error parsing mapping YAML file: {exc}")
            return

    # Initialize counters for verification
    total_images = 0
    total_labels = 0
    merged_images_count = 0
    merged_labels_count = 0

    # Iterate over each dataset to merge
    for dataset in merge_datasets:
        data_dir = dataset.get('data_dir')
        if not data_dir:
            print("Error: 'data_dir' not found in dataset parameters. Skipping this dataset.")
            continue

        # Derive data.yaml path
        possible_data_yaml_paths = [
            os.path.join(data_dir, 'data.yaml'),
            os.path.join(data_dir, 'annotations', 'data.yaml')
        ]
        data_yaml_path = None
        for path in possible_data_yaml_paths:
            if os.path.exists(path):
                data_yaml_path = path
                break
        if not data_yaml_path:
            print(f"Error: data.yaml not found for dataset at '{data_dir}'. Skipping this dataset.")
            continue

        # Get class mapping for this dataset
        mapping_key = data_yaml_path
        if mapping_key not in class_mappings:
            print(f"Error: No class mapping found for '{data_yaml_path}' in '{mapping_yaml}'. Skipping this dataset.")
            continue
        dataset_mapping = class_mappings[mapping_key]

        # Process each split: train, val, test
        splits = ['train', 'val', 'test']
        for split in splits:
            # Get source image and label directories
            src_images_dir = os.path.join(data_dir, dataset.get(f'{split}_images_dir', ''))
            src_labels_dir = os.path.join(data_dir, dataset.get(f'{split}_labels_dir', ''))

            # Get target image and label directories
            tgt_images_dir = os.path.join(merge_dataset_params['data_dir'], merge_dataset_params.get(f'{split}_images_dir', ''))
            tgt_labels_dir = os.path.join(merge_dataset_params['data_dir'], merge_dataset_params.get(f'{split}_labels_dir', ''))

            # Create target directories if they don't exist
            os.makedirs(tgt_images_dir, exist_ok=True)
            os.makedirs(tgt_labels_dir, exist_ok=True)

            if not os.path.exists(src_images_dir):
                print(f"Warning: Source images directory '{src_images_dir}' does not exist. Skipping this split.")
                continue
            if not os.path.exists(src_labels_dir):
                print(f"Warning: Source labels directory '{src_labels_dir}' does not exist. Skipping this split.")
                continue

            # List all image files
            image_extensions = ['.jpg', '.jpeg', '.png']
            src_images = [f for f in os.listdir(src_images_dir) if os.path.splitext(f)[1].lower() in image_extensions]
            src_labels = [f for f in os.listdir(src_labels_dir) if os.path.splitext(f)[1].lower() == '.txt']

            # Ensure that for every image, there is a corresponding label
            src_images_set = set(os.path.splitext(f)[0] for f in src_images)
            src_labels_set = set(os.path.splitext(f)[0] for f in src_labels)
            common_set = src_images_set.intersection(src_labels_set)
            missing_labels = src_images_set - src_labels_set
            missing_images = src_labels_set - src_images_set

            if missing_labels:
                print(f"Warning: {len(missing_labels)} images in '{split}' split of dataset '{data_dir}' do not have corresponding label files.")
            if missing_images:
                print(f"Warning: {len(missing_images)} label files in '{split}' split of dataset '{data_dir}' do not have corresponding images.")

            # Update total counts
            total_images += len(common_set)
            total_labels += len(common_set)

            for img_name in common_set:
                src_img_file = os.path.join(src_images_dir, img_name + os.path.splitext([f for f in src_images if os.path.splitext(f)[0]==img_name][0])[1])
                src_label_file = os.path.join(src_labels_dir, img_name + '.txt')

                tgt_img_file = os.path.join(tgt_images_dir, os.path.basename(src_img_file))
                tgt_label_file = os.path.join(tgt_labels_dir, os.path.basename(src_label_file))

                # Check for duplicate images
                if os.path.exists(tgt_img_file):
                    print(f"Warning: Image '{tgt_img_file}' already exists in the merged dataset. Skipping this image.")
                    continue

                # Copy image
                try:
                    shutil.copy2(src_img_file, tgt_img_file)
                    merged_images_count += 1
                except Exception as e:
                    print(f"Error copying image '{src_img_file}' to '{tgt_img_file}': {e}")
                    continue

                # Read label file and update class indices
                try:
                    with open(src_label_file, 'r') as lf:
                        label_lines = lf.readlines()
                except Exception as e:
                    print(f"Error reading label file '{src_label_file}': {e}")
                    continue

                updated_label_lines = []
                for line in label_lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        print(f"Warning: Invalid label format in file '{src_label_file}'. Skipping this line.")
                        continue
                    try:
                        original_class_id = parts[0]
                        new_class_id = dataset_mapping.get(original_class_id, None)
                        if new_class_id is None:
                            print(f"Warning: Class ID '{original_class_id}' not found in mapping for dataset '{data_dir}'. Skipping this label.")
                            continue
                        updated_line = ' '.join([str(new_class_id)] + parts[1:])
                        updated_label_lines.append(updated_line + '\n')
                    except Exception as e:
                        print(f"Error processing line '{line}' in file '{src_label_file}': {e}")
                        continue

                # Write updated label file
                try:
                    with open(tgt_label_file, 'w') as lf:
                        lf.writelines(updated_label_lines)
                    merged_labels_count += 1
                except Exception as e:
                    print(f"Error writing updated label file '{tgt_label_file}': {e}")
                    continue

    # After merging all datasets, verify counts
    print("\n--- Merging Completed ---")
    print(f"Total images to merge: {total_images}")
    print(f"Total labels to merge: {total_labels}")
    print(f"Total images merged: {merged_images_count}")
    print(f"Total labels merged: {merged_labels_count}")

    if merged_images_count == total_images and merged_labels_count == total_labels:
        print("Verification Passed: All images and labels have been successfully merged.")
    else:
        print("Verification Failed: Some images or labels were not merged successfully.")


def extract_zip_datasets(rf_dir):
    """
    Extracts all .zip files within the Roboflow directory.
    """
    zip_files = glob.glob(os.path.join(rf_dir, '*', '*.zip'))
    for zip_file in zip_files:
        print(f"Extracting {zip_file}")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Extract to the same directory as the zip file
            extract_path = os.path.splitext(zip_file)[0]
            zip_ref.extractall(extract_path)
        print(f"Extracted to {extract_path}")

        # Optionally, remove the zip file after extraction
        os.remove(zip_file)
        print(f"Removed zip file: {zip_file}")

def generate_ground_truth(image_dir,label_dir, output_dir, class_mapping, model_predictions, image_paths, input_format_truth='yolo', input_format_pred='voc'):
    """
    Converts YOLOv5 annotations and model predictions to the required format for mAP evaluation.

    Args:
        image_dir (str): Directory containing image files (e.g., JPG).
        label_dir (str): Directory containing YOLOv5 TXT label files (class_id center_x center_y width height).
                         The label files should have the same base name as the corresponding image
                         files, with the extension .txt.
        output_dir (str): Directory to save the ground-truth and detection-results files.
                          The ground-truth files will be saved in `output_dir/ground-truth/` and
                          the detection-results files will be saved in `output_dir/detection-results/`.
        class_mapping (dict): A dictionary mapping class IDs (int) to class names (str).
                              For example: {0: 'Dime', 1: 'Nickel', 2: 'Penny', 3: 'Quarter'}.
        model_predictions (yolo-nas objects): model predictions 
        image_paths (list[str]): paths to images in order as yolo-nas list them.               
        input_format_truth (str, optional): The format of the input ground-truth bounding boxes.
                                            Default is 'yolo', assuming YOLOv5 format.
        input_format_pred (str, optional): The format of the input prediction bounding boxes.
                                           Default is 'voc', assuming VOC format.

    Returns:
        None: The function writes the converted ground-truth and detection-results files to the specified
              `output_dir` in a format compatible with the Cartucho/mAP evaluation tool.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gt_dir = os.path.join(output_dir, 'ground-truth')
    dr_dir = os.path.join(output_dir, 'detection-results')
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(dr_dir, exist_ok=True)


    for image_name in os.listdir(image_dir):
        if image_name.endswith('.jpg'):
            # Extract the base name without extension to find the corresponding label file
            base_name = os.path.splitext(image_name)[0]
            label_file = os.path.join(label_dir, f"{base_name}.txt")

            if not os.path.exists(label_file):
                print(f"Warning: Label file not found for image: {image_name}")
                continue

            # Load the image to get its size
            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image: {image_name}")
                continue

            image_height, image_width = image.shape[:2]
            image_size = (image_width, image_height)

            output_file = os.path.join(gt_dir, f"{base_name}.txt")

            with open(label_file, 'r') as lf, open(output_file, 'w') as of:
                for line in lf:
                    try:
                        class_id, center_x, center_y, width, height = map(float, line.strip().split())
                    except:
                        continue

                    bbox = [center_x, center_y, width, height]

                    # Convert YOLO format (center_x, center_y, width, height) to (xmin, ymin, xmax, ymax)
                    converted_bbox = pbx.convert_bbox(bbox, from_type=input_format_truth, to_type='coco', image_size=image_size)

                    xmin, ymin, w, h = converted_bbox
                    xmax = xmin + w
                    ymax = ymin + h

                    # Assuming class_id maps directly to category name
                    class_name = class_mapping[int(class_id)]

                    of.write(f"{class_name} {xmin} {ymin} {xmax} {ymax}\n")

    # Process model predictions
    for image_path, predictions in zip(image_paths, model_predictions):
        image_file = os.path.basename(image_path)

        if not os.path.exists(label_file):
            print(f"Warning: Label file not found for image: {image_name}")
            continue

        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image: {image_name}")
            continue

        image_height, image_width = image.shape[:2]
        image_size = (image_width, image_height)

        output_file = os.path.join(dr_dir, image_file.replace('.jpg', '.txt'))

        with open(output_file, 'w') as f:
            for j, score in enumerate(predictions.prediction.confidence):
                
                category_name = predictions.class_names[int(predictions.prediction.labels[j])].replace(' ', '_')
                bbox = predictions.prediction.bboxes_xyxy[j]

                converted_back_bbox = pbx.convert_bbox(bbox, from_type=input_format_pred, to_type="coco", image_size=image_size)

                xmin, ymin, width, height = converted_back_bbox
                xmax = xmin + width
                ymax = ymin + height

                f.write(f"{category_name} {score} {xmin} {ymin} {xmax} {ymax}\n")
