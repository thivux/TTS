'''
resample vin27 audio from 16khz to 22050hz for xtts training 
'''

import os
import json
import subprocess
from tqdm.contrib.concurrent import thread_map
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Path to the input file
input_file = '/workspace/code/TTS/recipes/vctk/yourtts/VIN27/full_metadata.csv' 

# Function to convert a single file using SoX


def resample_22k(input_path):
    output_path = input_path.replace('vin27_16k', 'vin27_22k')
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        command = f'sox "{input_path}" -r 22050 "{output_path}"'
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_path}: {e}")


# Read the input file and collect all file paths
file_paths = []
# root_dir = '/lustre/scratch/client/vinai/users/linhnt140/zero-shot-tts/preprocess_audio/vin27_16k'
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

for sample in data:
    file_paths.append(sample['path'])

print(f'found {len(file_paths)} files to resample')
print(file_paths[:3])

# Use ThreadPoolExecutor to convert files in parallel
# with ThreadPoolExecutor(max_workers=16) as executor:
#     list(tqdm(executor.map(resample_22k, file_paths),
#          total=len(file_paths), desc="resampling files..."))
num_threads = 64 
list(thread_map(resample_22k, file_paths, desc="Resampling files", total=len(file_paths), max_workers=num_threads))


# output_file = './VIN27/44hours_22khz.txt'
# with open(output_file, 'w', encoding='utf-8') as out_file, open(input_file, 'r') as in_file:
#     for line in in_file:
#         path, province, speaker_id, duration, gender, transcript = line.strip().split('|')
#         new_file_path = path.replace('vin27_16k', 'vin27_22k')
#         output_file.write(
#             f'{new_file_path}|{province}|{speaker_id}|{duration}|{gender}|{transcript}\n')

print('successfully resampled audio to 22khz and save the new metadata file')
