import json
import os
from tqdm import tqdm
import pandas as pd
import csv
import subprocess
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict


def group_and_sort_by_speaker(input_csv, output_csv):
    """Groups data by speaker, calculates total duration, and sorts by duration.

    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to the output CSV file.
    """
    speaker_durations = defaultdict(float)

    with open(input_csv, 'r', newline='') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            speaker = row['speaker']
            duration = float(row['duration'])
            speaker_durations[speaker] += duration

    # Sort by duration in descending order
    sorted_durations = sorted(speaker_durations.items(), key=lambda item: item[1], reverse=True)

    with open(output_csv, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['speaker', 'total_duration'])
        writer.writerows(sorted_durations)


def get_duration(filepath):
    """Get the duration of an audio file using ffprobe."""
    cmd = ["ffprobe", "-i", filepath, "-show_entries",
           "format=duration", "-v", "quiet", "-of", "csv=p=0"]
    try:
        dur = subprocess.check_output(
            cmd, stderr=subprocess.PIPE).decode('utf-8').strip()
        return float(dur)
    except subprocess.CalledProcessError as e:
        print(f"Error processing file {filepath}: {e.stderr.decode('utf-8')}")
        return 0


def process_file(filepath, province, speaker_id, writer):
    """Processes a single file to extract information and write to CSV."""
    duration = get_duration(filepath)
    writer.writerow([os.path.relpath(filepath, root_directory),
                    province, f"{province}_{speaker_id}", duration])


def create_audio_csv(root_dir, csv_file):
    """Creates a CSV file containing information about audio files."""
    wav_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file_path', 'province', 'speaker', 'duration'])

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            for filepath in tqdm(wav_files, total=len(wav_files), desc="Processing files"):
                speaker_id = os.path.basename(os.path.dirname(filepath))
                province = os.path.basename(
                    os.path.dirname(os.path.dirname(filepath)))
                future = executor.submit(
                    process_file, filepath, province, speaker_id, writer)
                futures.append(future)

            # Collect results (optional, mainly for error handling)
            for future in tqdm(futures, desc="Collecting results"):
                future.result()  # Will raise an exception if any occurred during processing


def add_region_column(input_csv, output_csv):
    """Adds a 'region' column to the CSV based on province."""
    list_vietnamese_north_provinces = "bac-giang bac-kan bac-ninh cao-bang ha-giang ha-nam ha-noi ha-tay hai-duong hai-phong hoa-binh hung-yen lai-chau lang-son lao-cai nam-đinh ninh-binh phu-tho quang-ninh son-la thai-binh thai-nguyen tuyen-quang vinh-phuc yen-bai đien-bien".split(" ")
    list_vietnamese_center_provinces = "binh-thuan binh-đinh gia-lai ha-tinh khanh-hoa kon-tum lam-đong nghe-an ninh-thuan phu-yen quang-binh quang-nam quang-ngai quang-tri thanh-hoa thua-thien---hue đa-nang đak-lak đak-nong".split(" ")
    list_vietnamese_south_provinces = "an-giang ba-ria-vung-tau bac-lieu ben-tre binh-duong binh-phuoc ca-mau can-tho hau-giang ho-chi-minh kien-giang long-an soc-trang tay-ninh tien-giang tp.-ho-chi-minh tra-vinh vinh-long đong-nai đong-thap".split(" ")
    with open(input_csv, 'r', newline='') as infile, \
         open(output_csv, 'w', newline='') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader)  # Read the header row
        header.append('region')  # Add 'region' to the header
        writer.writerow(header)

        for row in tqdm(reader, desc="Adding region"):
            province = row[1]  # Assuming province is the second column (index 1)

            if province in list_vietnamese_north_provinces:
                region = 'north'
            elif province in list_vietnamese_center_provinces:
                region = 'center'
            elif province in list_vietnamese_south_provinces:
                region = 'south'
            else:
                print(f'Unknown province: {province}')
                print(f'row: {row}')
                region = 'unknown'

            row.append(region)
            writer.writerow(row)
            

def separate_csv_by_region(input_csv, output_dir):
    """Separates the CSV into multiple CSVs based on the 'region' column.

    Args:
        input_csv (str): Path to the input CSV file.
        output_dir (str): Path to the directory where output CSV files will be saved.
    """

    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    with open(input_csv, 'r', newline='') as infile:
        reader = csv.DictReader(infile)  # Use DictReader for easier column access

        # Create separate writers for each region
        region_writers = {}
        for row in reader:
            region = row['region']
            if region not in region_writers:
                output_path = os.path.join(output_dir, f'{region}.csv')
                outfile = open(output_path, 'w', newline='')
                writer = csv.writer(outfile)
                writer.writerow(reader.fieldnames)  # Write header row
                region_writers[region] = (outfile, writer)

        # Reset the reader to the beginning of the file
        infile.seek(0)
        next(reader)  # Skip the header row

        # Write data to respective region files
        for row in tqdm(reader, desc="Separating by region"):
            region = row['region']
            _, writer = region_writers[region]
            writer.writerow(row.values())

        # Close all output files
        for outfile, _ in region_writers.values():
            outfile.close()


def get_top_speakers(input_csv, top_n=150):
    """Gets the top N speakers from the sorted CSV.

    Args:
        input_csv (str): Path to the input CSV file.
        top_n (int): Number of top speakers to retrieve.

    Returns:
        list: List of top speaker names.
    """
    df = pd.read_csv(input_csv)
    top_speakers = df['speaker'].head(top_n).tolist()
    return top_speakers

    
def create_region_metadata(region, region_dir, top_speakers_file):
    """Creates metadata for a region, filtering by top speakers.

    Args:
        region (str): Region name ('north', 'south', 'center').
        region_dir (str): Directory containing region CSV files.
        top_speakers_file (str): Path to the file with top speaker names.
    """

    # Load top speakers
    with open(top_speakers_file, 'r') as f:
        top_speakers = set(line.strip() for line in f)

    # Load region CSV
    region_csv = os.path.join(region_dir, f"{region}.csv")
    df = pd.read_csv(region_csv)

    # Filter DataFrame for top speakers
    filtered_df = df[df['speaker'].isin(top_speakers)]

    # Save filtered metadata
    output_csv = os.path.join(region_dir, f"metadata_{region}.csv")
    filtered_df.to_csv(output_csv, index=False)
    print(f"Region: {region}, Metadata saved to: {output_csv}")


def concat_metadata_files(region_dir, output_file):
    """Concatenates metadata CSV files into a single file.

    Args:
        region_dir (str): Directory containing the metadata CSV files.
        output_file (str): Path to the output concatenated CSV file.
    """
    metadata_files = [
        f for f in os.listdir(region_dir) if f.startswith("metadata_") and f.endswith(".csv")
    ]

    all_dfs = []
    for filename in metadata_files:
        filepath = os.path.join(region_dir, filename)
        df = pd.read_csv(filepath)
        all_dfs.append(df)

    concatenated_df = pd.concat(all_dfs, ignore_index=True)
    concatenated_df.to_csv(output_file, index=False)
    print(f"Concatenated metadata saved to: {output_file}")
    

def add_transcript_gender():
    # Load the JSON transcript data
    with open("VIN27/transcription.json", encoding='utf-8') as f:
        transcript = json.load(f)

    # Output CSV filename
    output_csv = 'VIN27/updated_metadata.csv'

    # Function to extract data from transcript
    def _get_transcript_data(row):
        """
        Extracts 'gender' and 'normalized_text' from the 'transcript' dictionary
        based on matching keys from the CSV row.
        """
        province, speaker_id, wav_filename = row['file_path'].split("/")

        try:
            gender = transcript[province][speaker_id][wav_filename]['gender']
            normalized_text = transcript[province][speaker_id][wav_filename]['normalized_text']
        except KeyError:
            print(f"Warning: Data not found for province: {province}, speaker_id: {speaker_id}, wav_filename: {wav_filename}")
            return None, None  # Return None if data is not found

        return gender, normalized_text

    # Process the CSV file
    with open('VIN27/concatenated_metadata.csv', 'r', encoding='utf-8') as infile, \
        open(output_csv, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['gender', 'normalized_text']  # Add new columns
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter='|')
        writer.writeheader()  # Write the header row

        for row in tqdm(reader):
            gender, normalized_text = _get_transcript_data(row)
            if gender and normalized_text:
                row['gender'] = gender
                row['normalized_text'] = normalized_text
                writer.writerow(row) 

    print(f"Updated data written to: {output_csv}")
    
if __name__ == "__main__":
    # create audio_data.csv (relative_file_path, province, province_speakerid, duration)
    # root_directory = '/lustre/scratch/client/vinai/users/linhnt140/zero-shot-tts/preprocess_audio/vin27_16k'
    # csv_file_path = './VIN27/audio_data.csv'
    # create_audio_csv(root_directory, csv_file_path)

    # add region (north, central, south) to audio_data.csv -> audio_data_region.csv
    # csv_file_path = './VIN27/audio_data.csv'
    # csv_file_with_region = './VIN27/audio_data_region.csv'
    # add_region_column(csv_file_path, csv_file_with_region)

    # seperate audio_data_region.csv based on region
    # input_csv_file = './VIN27/audio_data_region.csv'
    # output_directory = './VIN27'
    # separate_csv_by_region(input_csv_file, output_directory) 
    
    # for each region, sort speaker by duration descendingly
    # region_dir = './VIN27'
    # for region in ['north', 'south', 'center']:
    #     input_csv = os.path.join(region_dir, f"{region}.csv")
    #     output_csv = os.path.join(region_dir, f"sorted_{region}.csv")
    #     group_and_sort_by_speaker(input_csv, output_csv)

    # get top 150 speakers for each region
    region2n_speakers = {
        'north': 150,
        'south': 75,
        'center': 125 
    }
    region_dir = './VIN27'
    for region in ['north', 'south', 'center']:
        input_csv = os.path.join(region_dir, f"sorted_{region}.csv")
        top_n = 100
        top_speakers = get_top_speakers(input_csv, top_n=region2n_speakers[region])

        output_file = os.path.join(region_dir, f"top_{region2n_speakers[region]}_speakers_{region}.txt")
        with open(output_file, 'w') as f:
            for speaker in top_speakers:
                f.write(f"{speaker}\n")
        print(f"Region: {region}, Top 150 speaker names saved to: {output_file}")

    # create metadata.csv for each region
    region_dir = './VIN27'

    for region in ['north', 'south', 'center']:
        top_speakers_file = os.path.join(region_dir, f"top_{region2n_speakers[region]}_speakers_{region}.txt")
        create_region_metadata(region, region_dir, top_speakers_file)

    # concat metadata into 1 file
    region_dir = './VIN27'
    output_file = './VIN27/concatenated_metadata.csv'
    concat_metadata_files(region_dir, output_file)

    # add transcript and gender to the metadata list
    add_transcript_gender()
    
