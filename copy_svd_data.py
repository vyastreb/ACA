import os
import numpy as np
import json

# Directories
folder_1 = "FullTestCycle/"
folder_2 = "FullTestCycle_Optimized/"

# Helper function to load JSON parameters
def load_json_parameters(json_file):
    with open(json_file, 'r') as f:
        params = json.load(f)
    return params['xi'], params['target_distance'], params['rank3treatment']

# Helper function to load SVD data from Folder 1
def load_svd_data(npz_file):
    with np.load(npz_file) as data:
        if 'SVD' in data:
            return data['SVD']
        else:
            return None

# Helper function to update npz files in Folder 2 and 3 with SVD data
def update_npz_file(npz_file, svd_data):
    with np.load(npz_file) as data:
        # Load existing arrays
        arrays = dict(data)

    # Update or add the SVD data
    arrays['SVD'] = svd_data

    # Save the updated npz file
    np.savez(npz_file, **arrays)
    print(f"Updated {npz_file} with SVD data")

# Process all files in Folder 1
for filename in os.listdir(folder_1):
    if filename.endswith(".json"):
        # Get corresponding .npz and .json filenames
        json_file_1 = os.path.join(folder_1, filename)
        npz_file_1 = os.path.join(folder_1, filename.replace('.json', '.npz'))

        # Load parameters from the json file
        xi_1, dist_1, rank3_1 = load_json_parameters(json_file_1)

        # Load the SVD data from Folder 1
        svd_data = load_svd_data(npz_file_1)
        if svd_data is None:
            print(f"No SVD data found in {npz_file_1}")
            continue

        # Iterate over Folders 2 and 3 to find matching files
        for folder in [folder_2]:
            for filename_other in os.listdir(folder):
                if filename_other.endswith(".json"):
                    # Get corresponding .npz and .json filenames in Folder 2 or 3
                    json_file_other = os.path.join(folder, filename_other)
                    npz_file_other = os.path.join(folder, filename_other.replace('.json', '.npz'))

                    # Load parameters from the json file
                    xi_other, dist_other, rank3_other = load_json_parameters(json_file_other)

                    # Compare the parameters
                    if xi_1 == xi_other and dist_1 == dist_other and rank3_1 == rank3_other:
                        # Update the .npz file in Folder 2 or 3 with the SVD data
                        update_npz_file(npz_file_other, svd_data)

print("Process completed.")
