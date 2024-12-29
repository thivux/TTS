import json
import pandas as pd
import os
from tqdm import tqdm
import pandas as pd
import csv
import subprocess
from collections import defaultdict
from tqdm.contrib.concurrent import thread_map


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
    sorted_durations = sorted(
        speaker_durations.items(), key=lambda item: item[1], reverse=True)

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
    except ValueError:
        print(f"Error converting duration to float for file {filepath}, got '{dur}'")
    return 0


def process_file(filepath, province, speaker_id):
    """Process a single file to extract information and return it."""
    duration = get_duration(filepath)
    return [os.path.relpath(filepath, root_directory), province, f"{province}_{speaker_id}", duration]

def write_to_csv(data, csv_file):
    """Writes data to the CSV file."""
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file_path', 'province', 'speaker', 'duration'])
        writer.writerows(data)

def create_audio_csv(root_directory, input_file_path, output_file_path):
    """Creates a CSV file containing information about audio files."""

    wav_files = [os.path.join(root_directory, line.strip())
                 for line in open(input_file_path, 'r')]
    wav_files = wav_files[int(len(wav_files)/2):]
    print(f'There are {len(wav_files)} wav files in the dataset')

    def process_wrapper(filepath):
        speaker_id = os.path.basename(os.path.dirname(filepath))
        province = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
        return process_file(filepath, province, speaker_id)

    max_workers = 8 
    print(f'max_workers: {max_workers}')

    # Use thread_map to process files with a progress bar
    results = thread_map(process_wrapper, wav_files, max_workers=max_workers, desc="Processing files")

    # Write all results to CSV
    write_to_csv(results, output_file_path)
    

def add_region_column(input_csv, output_csv):
    """Adds a 'region' column to the CSV based on province."""
    list_vietnamese_north_provinces = "bac-giang bac-kan bac-ninh cao-bang ha-giang ha-nam ha-noi ha-tay hai-duong hai-phong hoa-binh hung-yen lai-chau lang-son lao-cai nam-đinh ninh-binh phu-tho quang-ninh son-la thai-binh thai-nguyen tuyen-quang vinh-phuc yen-bai đien-bien".split(
        " ")
    list_vietnamese_center_provinces = "binh-thuan binh-đinh gia-lai ha-tinh khanh-hoa kon-tum lam-đong nghe-an ninh-thuan phu-yen quang-binh quang-nam quang-ngai quang-tri thanh-hoa thua-thien---hue đa-nang đak-lak đak-nong".split(
        " ")
    list_vietnamese_south_provinces = "an-giang ba-ria-vung-tau bac-lieu ben-tre binh-duong binh-phuoc ca-mau can-tho hau-giang ho-chi-minh kien-giang long-an soc-trang tay-ninh tien-giang tp.-ho-chi-minh tra-vinh vinh-long đong-nai đong-thap".split(
        " ")
    with open(input_csv, 'r', newline='') as infile, \
            open(output_csv, 'w', newline='') as outfile:

        reader = csv.reader(infile, delimiter='|')
        writer = csv.writer(outfile)

        header = next(reader)  # Read the header row
        header.append('region')  # Add 'region' to the header
        writer.writerow(header)

        for row in tqdm(reader, desc="Adding region"):
            # Assuming province is the second column (index 1)
            province = row[1]

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

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    with open(input_csv, 'r', newline='') as infile:
        # Use DictReader for easier column access
        reader = csv.DictReader(infile)

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
    output_csv = 'VIN27/full_metadata.csv'

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
            print(
                f"Warning: Data not found for province: {province}, speaker_id: {speaker_id}, wav_filename: {wav_filename}")
            return None, None  # Return None if data is not found

        return gender, normalized_text

    # Process the CSV file
    with open('VIN27/combined_audio_data_region.csv', 'r', encoding='utf-8') as infile, \
            open(output_csv, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile, delimiter=',')
        fieldnames = reader.fieldnames + \
            ['gender', 'normalized_text']  # Add new columns
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter='|')
        writer.writeheader()  # Write the header row

        for row in tqdm(reader):
            gender, normalized_text = _get_transcript_data(row)
            if gender and normalized_text:
                row['gender'] = gender
                row['normalized_text'] = normalized_text
                writer.writerow(row)

    print(f"Updated data written to: {output_csv}")


def filter_files_not_cal_dur(root_dir, csv_file_path):
    '''
    load files in VIN27 dataset and save files whose duration are not calculated 
    '''

    list_vietnamese_north_provinces = "bac-giang bac-kan bac-ninh cao-bang ha-giang ha-nam ha-noi ha-tay hai-duong hai-phong hoa-binh hung-yen lai-chau lang-son lao-cai nam-đinh ninh-binh phu-tho quang-ninh son-la thai-binh thai-nguyen tuyen-quang vinh-phuc yen-bai đien-bien".split(
        " ")
    list_vietnamese_center_provinces = "binh-thuan binh-đinh gia-lai ha-tinh khanh-hoa kon-tum lam-đong nghe-an ninh-thuan phu-yen quang-binh quang-nam quang-ngai quang-tri thanh-hoa thua-thien---hue đa-nang đak-lak đak-nong".split(
        " ")
    list_vietnamese_south_provinces = "an-giang ba-ria-vung-tau bac-lieu ben-tre binh-duong binh-phuoc ca-mau can-tho hau-giang ho-chi-minh kien-giang long-an soc-trang tay-ninh tien-giang tp.-ho-chi-minh tra-vinh vinh-long đong-nai đong-thap".split(
        " ")
    processed_files = []
    with open(csv_file_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i == 0:
                continue
            else:
                parts = line.split(',')
                if len(parts) == 4:
                    processed_files.append(parts[0])
                    if parts[-2] == "hau-giang_3511048" and parts[-1].strip() == "6.112": 
                        print(f'line number: {i}, parts: {parts}')
                else:
                    print(f'line number: {i}, number of parts: {len(parts)}')
                    print(f'line: {line}')

                if parts[1] in list_vietnamese_center_provinces or parts[1] in list_vietnamese_north_provinces or parts[1] in list_vietnamese_south_provinces:
                    pass
                else:
                    print(
                        f'houston we have a problem at line {i}, province: {parts[1]}')

    print(f'there are {len(processed_files)} processed files')
    print(processed_files[:2])

    # load all wav files in the dataset
    wav_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.relpath(
                    os.path.join(root, file), root_dir))

    print(f'there are {len(wav_files)} wav files in the dataset')
    print(wav_files[:2])

    # save files that are not processed to txt file
    not_processed_files = list(set(wav_files) - set(processed_files))
    print(f'files not processed: {len(not_processed_files)}')
    # with open('VIN27/not_processed_files.txt', 'w') as f:
    #     for file in not_processed_files:
    #         f.write(f"{file}\n")


def filter_short_audio(root_directory, metafile, threshold):
    # Load the metadata file
    df = pd.read_csv(metafile, delimiter=',', names=["file_path", "province", 'speaker', 'duration'])

    # Filter out audio files with duration less than threshold
    df = df[df['duration'] >= threshold]

    # Save the filtered metadata
    output_file = metafile.replace('.csv', f'_filtered_{threshold}s.csv')
    df.to_csv(output_file, index=False, sep='|')
    print(f"Filtered metadata saved to: {output_file}")
    
    return output_file

            
def create_json_file(root_directory, input_csv, output_json): 
    json_data = []

    # Open and read the CSV file
    with open(input_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='|')
    
        for row in tqdm(reader):
            # Create a dictionary for each row with the required structure
            entry = {
                "path": os.path.join(root_directory, row["file_path"]),
                "transcript": row["normalized_text"],
                "speaker": row["speaker"],
                "duration": float(row["duration"]),
                "dialect": row["region"],
                "segment_id": row["file_path"].replace("/", "_").replace(".wav", ""),
            }
            # Append the dictionary to the list
            json_data.append(entry)

    # Write the list of dictionaries to a JSON file
    with open(output_json, 'w', encoding='utf-8') as jsonfile:
        json.dump(json_data, jsonfile, ensure_ascii=False, indent=4)

    print(f"JSON file has been created at {output_json}")


if __name__ == "__main__":
    # get short audio files
    file1 = 'VIN27/combined_audio_data.csv'
    file2 = 'VIN27/combined_audio_data_filtered_3s.csv'
    output_file = 'VIN27/short_files.csv'
    full_df = pd.read_csv(file1, names=['path', 'province', 'speaker', 'duration'])
    short = full_df[full_df['duration'] <= 3]
    short.to_csv(output_file, index=False, sep='|', header=False)

    # root_directory = '/lustre/scratch/client/vinai/users/linhnt140/zero-shot-tts/preprocess_audio/vin27_16k_denoised'
    
    # # filter > 3s audio files
    # filtered_length_path = 'VIN27/short_files.csv'

    # # add region (north, central, south) to audio_data.csv -> audio_data_region.csv
    # csv_file_with_region = './VIN27/short_files_region.csv'
    # add_region_column(filtered_length_path, csv_file_with_region)

    # # add transcript and gender to the metadata list
    # add_transcript_gender()
    
    # # create json file for phonemize_encode
    # create_json_file(root_directory, "VIN27/full_metadata.csv", "VIN27/metadata.json")
