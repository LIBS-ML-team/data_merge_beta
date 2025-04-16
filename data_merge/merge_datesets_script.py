import os
import yaml
import argparse
from object_detection.src.data import merge_data_yamls, merge_datasets
from object_detection.src.analyzer import analyze_datasets
from object_detection.src.parser import parse_config
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Merge multiple YOLO-formatted datasets into a single dataset.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    
    args = parser.parse_args()
    config_path = args.config

    # Parse the configuration file
    config = parse_config(config_path)

    # Extract dataset list and merged dataset parameters
    datasets = config.get('datasets', [])
    merged_dataset = config.get('merged_dataset', {})

    if not datasets:
        print("Error: No datasets found in the configuration file.")
        return
    if not merged_dataset:
        print("Error: Merged dataset parameters not found in the configuration file.")
        return

    # Collect original data.yaml paths
    original_yamls = []
    for dataset in datasets:
        data_dir = dataset.get('data_dir')
        if not data_dir:
            print("Warning: 'data_dir' not specified for a dataset. Skipping.")
            continue

        # Possible locations for data.yaml
        possible_data_yaml_paths = [
            os.path.join(data_dir, 'data.yaml'),
            os.path.join(data_dir, 'annotations', 'data.yaml')
        ]
        data_yaml_path = next((path for path in possible_data_yaml_paths if os.path.exists(path)), None)
        if data_yaml_path:
            original_yamls.append(data_yaml_path)
        else:
            print(f"Warning: data.yaml not found for dataset '{data_dir}'. Skipping.")

    if not original_yamls:
        print("Error: No valid data.yaml files found for any dataset.")
        return

    # Define paths for the merged dataset
    merged_data_dir = merged_dataset.get('data_dir')
    if not merged_data_dir:
        print("Error: 'data_dir' not specified for the merged dataset.")
        return

    merged_yaml = os.path.join(merged_data_dir, 'data.yaml')
    mappings_yaml = os.path.join(merged_data_dir, 'class_mappings.yaml')

    merged_train = os.path.join(merged_data_dir, merged_dataset.get('train_images_dir', 'train/images'))
    merged_val = os.path.join(merged_data_dir, merged_dataset.get('val_images_dir', 'valid/images'))
    merged_test = os.path.join(merged_data_dir, merged_dataset.get('test_images_dir', 'test/images'))

    # Create merged directories
    os.makedirs(merged_train, exist_ok=True)
    os.makedirs(merged_val, exist_ok=True)
    os.makedirs(merged_test, exist_ok=True)

    # Merge data.yaml files with progress
    print("\n--- Merging data.yaml Files ---")
    merged_classes, class_mappings = merge_data_yamls(
        yaml_paths=original_yamls,
        merged_yaml_path=merged_yaml,
        merged_train_path=merged_train,
        merged_val_path=merged_val,
        merged_test_path=merged_test,
        mappings_yaml_path=mappings_yaml
    )

    # Merge datasets with progress
    print("\n--- Merging Datasets ---")
    total_datasets = len(datasets)
    for i, dataset in enumerate(datasets, 1):
        print(f"\nProcessing dataset {i}/{total_datasets}: {dataset.get('name', 'unnamed')}")
        merge_datasets(
            merge_datasets=[dataset],  # Process one at a time
            mapping_yaml=mappings_yaml,
            merge_dataset_params=merged_dataset
        )

    print("\n--- Analyzing Merged Datasets ---")
    analyze_datasets(
        merge_datasets_list=datasets,
        mapping_yaml=mappings_yaml,
        merge_dataset_params=merged_dataset
    )

if __name__ == "__main__":
    main()