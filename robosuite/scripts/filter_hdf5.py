
import argparse
import json
import os
import random

import h5py
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to your demonstration folder that contains the demo.hdf5 file, e.g.: "
        "'path_to_assets_dir/demonstrations/YOUR_DEMONSTRATION'",
    ),
    parser.add_argument(
        "--use-actions",
        action="store_true",
    )
    args = parser.parse_args()

    mug_handle_dataset = [1,10,13,21,26,29,34,41,47,48,5,51,55,57]
    mug_rim_dataset = [2,16,17,3,30,32,35,37,4,42,44,49,56,59,60]
    mug_rim_dataset_not_optimal = [2,14,16,17,18,20,22,23,24,25,28,3,30,32,33,35,36,37,38,39,4,42,44,49,53,54,55,56,59,60,61,7,8]

    demo_path = args.folder
    hdf5_path = os.path.join(demo_path, "demo.hdf5")

    # Output HDF5 file paths
    mug_handle_path = os.path.join(demo_path, "mug_handle_dataset.hdf5")
    mug_rim_path = os.path.join(demo_path, "mug_rim_dataset.hdf5")

    with h5py.File(hdf5_path, "r") as f:
        # Get the list of demos
        demos = list(f["data"].keys())

        # Create output HDF5 files
        with h5py.File(mug_handle_path, "w") as handle_file, h5py.File(mug_rim_path, "w") as rim_file:
            # Create data groups for each new dataset
            handle_group = handle_file.create_group("data")
            rim_group = rim_file.create_group("data")

            for demo in demos:
                demo_index = int(demo.split("_")[-1])  # Extract the demo index from the key name

                # Copy to mug_handle_dataset if the index matches
                if demo_index in mug_handle_dataset:
                    f.copy(f["data"][demo], handle_group, name=demo)

                # Copy to mug_rim_dataset if the index matches
                elif demo_index in mug_rim_dataset:
                    f.copy(f["data"][demo], rim_group, name=demo)

            print("handle_group is:", type(handle_group))
            print("rim_group is:", type(rim_group))
            
            # Copy all attributes from source_group to target_group
            source_group = f["data"]
            for key, value in source_group.attrs.items():
                handle_group.attrs[key] = value
                rim_group.attrs[key] = value

            print("Attributes in mug_handle_dataset:")
            for key, value in handle_group.attrs.items():
                print(f"{key}: {value}")

            print("\nAttributes in mug_rim_dataset:")
            for key, value in rim_group.attrs.items():
                print(f"{key}: {value}")
