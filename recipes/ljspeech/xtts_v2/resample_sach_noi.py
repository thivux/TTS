'''
resample sach_noi audio from 16khz to 22050hz for xtts training 
'''

import os
import subprocess
# from concurrent.futures import ThreadPoolExecutor
from tqdm.contrib.concurrent import thread_map
from tqdm import tqdm

# Path to the input file
input_file = '/workspace/code/oneshot/artifacts/step14_tone_norm_transcript_no_multispeaker.txt'

# Function to convert a single file using SoX
def resample_22k(input_path):
    output_path = input_path.replace('.wav', '_22k.wav')
    try:
        command = f'sox "{input_path}" -r 22050 "{output_path}"'
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_path}: {e}")


# Read the input file and collect all file paths
file_paths = []
root_dir = '/workspace/code/oneshot' 
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('|')
        file_paths.append(os.path.join(root_dir, parts[0]))

print(f'found {len(file_paths)} files to resample')
print(file_paths[:3])

# Use ThreadPoolExecutor to convert files in parallel
# with ThreadPoolExecutor(max_workers=8) as executor:
#     list(tqdm(executor.map(resample_24k, file_paths),
#          total=len(file_paths), desc="resampling files..."))

num_threads = 8
list(thread_map(resample_22k, file_paths, desc="Resampling files", total=len(file_paths), max_workers=num_threads))