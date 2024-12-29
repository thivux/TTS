'''
create path-text pairs of full data, including short audio files (1-2s)
'''

import json

# Load the JSON data
with open('VIN27/transcription.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Open a new file to write the pairs
with open('VIN27/path_text_pairs.txt', 'w', encoding='utf-8') as outfile:
    # Iterate through each province
    for province, speakers in data.items():
        # Iterate through each speaker
        for speaker, recordings in speakers.items():
            # Iterate through each recording
            for recording, info in recordings.items():
                # Create the path
                path = f"{province}/{speaker}/{recording}"
                # Get the transcript (origin_text)
                transcript = info['normalized_text']
                # Write the pair to the file
                outfile.write(f"{path}|{transcript}\n")

print("Transcript pairs have been written to path_text_pairs.txt")