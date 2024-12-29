import time
import subprocess
from lhotse import CutSet

def get_duration_ffprobe(filepath):
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

def get_duration_lhotse(filepath):
    """Get the duration of an audio file using lhotse."""
    try:
        # Create a CutSet on-the-fly for the single audio file
        cut = CutSet.from_dicts(
            [{'id': 'audio', 'start': 0, 'duration': None, 'channel': 0, 'recording': {'id': 'audio', 'sources': [{'type': 'file', 'channels': [0], 'source': filepath}]}}]
        )
        return cut.duration
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return 0

# Example usage and comparison:
# audio_file = 'path/to/your/audio.wav'  # Replace with your audio file
import os
root_directory = '/lustre/scratch/client/vinai/users/linhnt140/zero-shot-tts/preprocess_audio/vin27_16k'
audio_file = os.path.join(root_directory, "binh-đinh/0971972/125.wav")

# Measure ffprobe execution time
start_time = time.time()
duration_ffprobe = get_duration_ffprobe(audio_file)
end_time = time.time()
ffprobe_time = end_time - start_time

# Measure lhotse execution time
start_time = time.time()
duration_lhotse = get_duration_lhotse(audio_file)
end_time = time.time()
lhotse_time = end_time - start_time

print(f"Duration using ffprobe: {duration_ffprobe:.2f} seconds (took {ffprobe_time:.4f} seconds)")
print(f"Duration using lhotse: {duration_lhotse:.2f} seconds (took {lhotse_time:.4f} seconds)")