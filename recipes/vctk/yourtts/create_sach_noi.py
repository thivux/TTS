'''
- move audio files from oneshot/big_processed_data to TTS/recipes/vctk/yourtts/SACH_NOI
- resample audio from 32-bit integer to 16-bit integer
'''

import os
import pandas as pd
import subprocess
from tqdm import tqdm


# Define the paths
source_base_path = '/lustre/scratch/client/vinai/users/thivt1/code/oneshot'
destination_base_path = '/lustre/scratch/client/vinai/users/thivt1/code/TTS/recipes/vctk/yourtts/SACH_NOI/wavs'

# Read the CSV file
metadata_file = 'SACH_NOI/metadata_big_processed_data.txt'
metadata = pd.read_csv(metadata_file)

# Function to copy and resample files
def copy_and_resample_files(source_base, destination_base, paths):
    for path in tqdm(paths):
        # Construct full source and destination paths
        full_source_path = os.path.join(source_base, path)
        full_destination_path = os.path.join(destination_base, path.replace("big_processed_data/", ''))
        
        # Create the destination directory if it doesn't exist
        os.makedirs(os.path.dirname(full_destination_path), exist_ok=True)
        
        # Resample and copy the file using sox
        command = f'sox "{full_source_path}" -b 16 "{full_destination_path}"'
        subprocess.run(command, shell=True, check=True)
        # print(f"Copied and resampled {full_source_path} to {full_destination_path}")

# Extract the 'path' column from the metadata
paths_to_copy = metadata['path']

# Copy and resample the files
copy_and_resample_files(source_base_path, destination_base_path, paths_to_copy)
