from roboflow import Roboflow
from getpass import getpass
import os
import yaml
from collections import OrderedDict
import shutil
import random
from typing import List, Dict

def analyze_datasets(merge_datasets_list: List[Dict], mapping_yaml: str, merge_dataset_params: Dict):
    """
    Analyzes multiple YOLO-formatted datasets and their merged counterpart to ensure data integrity.

    Parameters:
    - merge_datasets_list (list): List of dictionaries, each containing paths for a dataset.
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

    # Initialize counters for merged dataset verification
    total_source_images = {'train': 0, 'val': 0, 'test': 0}
    total_source_labels = {'train': 0, 'val': 0, 'test': 0}

    # Analyze each dataset
    for dataset in merge_datasets_list:
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

        print(f"\nAnalyzing Dataset: {data_dir}")
        print(f"  Using class mapping from: {data_yaml_path}")

        splits = ['train', 'val', 'test']
        for split in splits:
            # Get source image and label directories
            src_images_dir = os.path.join(data_dir, dataset.get(f'{split}_images_dir', ''))
            src_labels_dir = os.path.join(data_dir, dataset.get(f'{split}_labels_dir', ''))

            # Get target image and label directories
            tgt_images_dir = os.path.join(merge_dataset_params['data_dir'], merge_dataset_params.get(f'{split}_images_dir', ''))
            tgt_labels_dir = os.path.join(merge_dataset_params['data_dir'], merge_dataset_params.get(f'{split}_labels_dir', ''))

            # Check if source directories exist
            if not os.path.exists(src_images_dir):
                print(f"  Warning: Source images directory '{src_images_dir}' does not exist for split '{split}'. Skipping this split.")
                continue
            if not os.path.exists(src_labels_dir):
                print(f"  Warning: Source labels directory '{src_labels_dir}' does not exist for split '{split}'. Skipping this split.")
                continue

            # List all image and label files
            image_extensions = ['.jpg', '.jpeg', '.png']
            try:
                src_images = [f for f in os.listdir(src_images_dir) if os.path.splitext(f)[1].lower() in image_extensions]
            except Exception as e:
                print(f"  Error accessing images in '{src_images_dir}': {e}")
                continue

            try:
                src_labels = [f for f in os.listdir(src_labels_dir) if os.path.splitext(f)[1].lower() == '.txt']
            except Exception as e:
                print(f"  Error accessing labels in '{src_labels_dir}': {e}")
                continue

            # Create sets of base names
            src_images_set = set(os.path.splitext(f)[0] for f in src_images)
            src_labels_set = set(os.path.splitext(f)[0] for f in src_labels)

            # Find common base names
            common_set = src_images_set.intersection(src_labels_set)
            missing_labels = src_images_set - src_labels_set
            missing_images = src_labels_set - src_images_set

            # Update total counts
            total_source_images[split] += len(common_set)
            total_source_labels[split] += len(common_set)

            # Print discrepancies
            if missing_labels:
                print(f"  Warning: {len(missing_labels)} images in '{split}' split do not have corresponding label files.")
            if missing_images:
                print(f"  Warning: {len(missing_images)} label files in '{split}' split do not have corresponding images.")

            print(f"  Split: {split}")
            print(f"    Total Common Images and Labels: {len(common_set)}")
            print(f"    Images without Labels: {len(missing_labels)}")
            print(f"    Labels without Images: {len(missing_images)}")

            # If there are common images, proceed to print sample
            if common_set:
                sample_image_base = random.choice(list(common_set))
                # Determine original and merged label file paths
                src_label_file = os.path.join(src_labels_dir, sample_image_base + '.txt')
                tgt_label_file = os.path.join(tgt_labels_dir, sample_image_base + '.txt')

                # Check if merged label file exists
                if os.path.exists(tgt_label_file):
                    try:
                        with open(src_label_file, 'r') as f:
                            src_label_content = f.read()
                        with open(tgt_label_file, 'r') as f:
                            tgt_label_content = f.read()

                        print(f"    Sample Label File: {sample_image_base}.txt")
                        print(f"      Original Label Content:\n{src_label_content}")
                        print(f"      Merged Label Content:\n{tgt_label_content}")
                    except Exception as e:
                        print(f"    Error reading sample label files: {e}")
                else:
                    print(f"    Warning: Merged label file '{tgt_label_file}' does not exist.")
            else:
                print(f"    No common images and labels found for split '{split}'.")

    # Analyze the merged dataset
    merged_train_images_dir = os.path.join(merge_dataset_params['data_dir'], merge_dataset_params.get('train_images_dir', ''))
    merged_train_labels_dir = os.path.join(merge_dataset_params['data_dir'], merge_dataset_params.get('train_labels_dir', ''))
    merged_val_images_dir = os.path.join(merge_dataset_params['data_dir'], merge_dataset_params.get('val_images_dir', ''))
    merged_val_labels_dir = os.path.join(merge_dataset_params['data_dir'], merge_dataset_params.get('val_labels_dir', ''))
    merged_test_images_dir = os.path.join(merge_dataset_params['data_dir'], merge_dataset_params.get('test_images_dir', ''))
    merged_test_labels_dir = os.path.join(merge_dataset_params['data_dir'], merge_dataset_params.get('test_labels_dir', ''))

    splits = ['train', 'val', 'test']
    merged_counts = {'train': 0, 'val': 0, 'test': 0}
    for split in splits:
        merged_images_dir = os.path.join(merge_dataset_params['data_dir'], merge_dataset_params.get(f'{split}_images_dir', ''))
        merged_labels_dir = os.path.join(merge_dataset_params['data_dir'], merge_dataset_params.get(f'{split}_labels_dir', ''))

        if not os.path.exists(merged_images_dir):
            print(f"\nWarning: Merged images directory '{merged_images_dir}' does not exist for split '{split}'.")
            continue
        if not os.path.exists(merged_labels_dir):
            print(f"\nWarning: Merged labels directory '{merged_labels_dir}' does not exist for split '{split}'.")
            continue

        try:
            merged_images = [f for f in os.listdir(merged_images_dir) if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']]
        except Exception as e:
            print(f"\nError accessing images in '{merged_images_dir}': {e}")
            continue

        try:
            merged_labels = [f for f in os.listdir(merged_labels_dir) if os.path.splitext(f)[1].lower() == '.txt']
        except Exception as e:
            print(f"\nError accessing labels in '{merged_labels_dir}': {e}")
            continue

        merged_images_set = set(os.path.splitext(f)[0] for f in merged_images)
        merged_labels_set = set(os.path.splitext(f)[0] for f in merged_labels)

        common_set = merged_images_set.intersection(merged_labels_set)
        missing_labels = merged_images_set - merged_labels_set
        missing_images = merged_labels_set - merged_images_set

        merged_counts[split] += len(common_set)

        if missing_labels:
            print(f"\nWarning: {len(missing_labels)} images in merged '{split}' split do not have corresponding label files.")
        if missing_images:
            print(f"Warning: {len(missing_images)} label files in merged '{split}' split do not have corresponding images.")

        print(f"\nMerged Split: {split}")
        print(f"  Total Common Images and Labels: {len(common_set)}")
        print(f"  Images without Labels: {len(missing_labels)}")
        print(f"  Labels without Images: {len(missing_images)}")

    # Verify counts
    print("\n--- Verification Summary ---")
    for split in splits:
        source_count = total_source_images.get(split, 0)
        merged_count = merged_counts.get(split, 0)
        if source_count == merged_count:
            print(f"  {split.capitalize()} Split: OK (Merged: {merged_count}, Source: {source_count})")
        else:
            print(f"  {split.capitalize()} Split: Mismatch! (Merged: {merged_count}, Source: {source_count})")

    # Example Visualization
    print("\n--- Sample Label Verification ---")
    for dataset in merge_datasets_list:
        data_dir = dataset.get('data_dir')
        if not data_dir:
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
            continue

        mapping_key = data_yaml_path
        dataset_mapping = class_mappings.get(mapping_key, {})
        if not dataset_mapping:
            continue

        splits = ['train', 'val', 'test']
        for split in splits:
            src_labels_dir = os.path.join(data_dir, dataset.get(f'{split}_labels_dir', ''))
            if not os.path.exists(src_labels_dir):
                continue

            src_labels = [f for f in os.listdir(src_labels_dir) if os.path.splitext(f)[1].lower() == '.txt']
            if not src_labels:
                continue

            sample_label = random.choice(src_labels)
            src_label_path = os.path.join(src_labels_dir, sample_label)

            # Determine merged label path
            merged_labels_dir = os.path.join(merge_dataset_params['data_dir'], merge_dataset_params.get(f'{split}_labels_dir', ''))
            merged_label_path = os.path.join(merged_labels_dir, sample_label)

            if not os.path.exists(merged_label_path):
                print(f"\nWarning: Merged label file '{merged_label_path}' does not exist for sample '{sample_label}'.")
                continue

            try:
                with open(src_label_path, 'r') as f:
                    src_label_content = f.read()
                with open(merged_label_path, 'r') as f:
                    merged_label_content = f.read()

                print(f"\nDataset: {data_dir}, Split: {split}, Sample: {sample_label}")
                print(f"  Original Label Content:\n{src_label_content}")
                print(f"  Merged Label Content:\n{merged_label_content}")
            except Exception as e:
                print(f"\nError reading label files for sample '{sample_label}': {e}")
                continue

    print("\n--- Analysis Completed ---")