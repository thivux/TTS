from tqdm import tqdm
import numpy as np
import json
import glob
import os
import re
import time
import torch
import torchaudio
from huggingface_hub import snapshot_download
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


# Define paths and model settings
checkpoint_dir = "./vivoice_model/"
repo_id = "capleaf/viXTTS"
use_deepspeed = False

# Download model files if not available
os.makedirs(checkpoint_dir, exist_ok=True)
snapshot_download(repo_id=repo_id, repo_type="model", local_dir=checkpoint_dir)

# Load model configuration
xtts_config = os.path.join(checkpoint_dir, "config.json")
config = XttsConfig()
config.load_json(xtts_config)
MODEL = Xtts.init_from_config(config)
MODEL.load_checkpoint(config, checkpoint_dir=checkpoint_dir, use_deepspeed=use_deepspeed)

if torch.cuda.is_available():
    MODEL.cuda()

# Inference function
def run_inference(target_text, ref_audio_path, output_path):
    # if len(target_text) < 2 or len(target_text) > 250:
    #     raise ValueError("Prompt length must be between 2 and 250 characters.")
    #
    # Load reference audio
    speaker_wav = ref_audio_path
    (
        gpt_cond_latent,
        speaker_embedding,
    ) = MODEL.get_conditioning_latents(
        audio_path=speaker_wav,
        gpt_cond_len=30,
        gpt_cond_chunk_len=4,
        max_ref_length=60,
    )

    # Generate audio
    out = MODEL.inference(
        target_text,
        'vi',
        gpt_cond_latent,
        speaker_embedding,
        repetition_penalty=5.0,
        temperature=0.75,
        enable_text_splitting=True,
    )

    # Save output
    torchaudio.save(output_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        
def infer_test_set():
    # ===========infer all samples in test set===========
    set_seed(42)

    # Get all JSON files in the test_set_5k folder
    test_set_path = "/lustre/scratch/client/vinai/users/thivt1/code/VoiceCraft/test_set_5k"
    json_files = sorted(glob.glob(f"{test_set_path}/*.json"))
    json_files = [file for file in json_files if "VIVOS" in file] 
    print(json_files)
    print(f'there are {len(json_files)} speakers in test set')

    for i, speaker in enumerate(json_files):
        # output_speaker_dir = os.path.join("/lustre/scratch/client/vinai/users/thivt1/code/VoiceCraft/test_set_results", os.path.basename(speaker).split('.')[0])
        # print(f'output_speaker_dir: {output_speaker_dir}')
        #
        # # Check if all samples for this speaker have already been processed
        # all_processed = True
        # for root, dirs, files in os.walk(output_speaker_dir):
        #     if not any(file.endswith("xtts_vivoice.wav") for file in files):
        #         all_processed = False
        #         break
        #
        # # If all samples are already processed, skip this speaker
        # if all_processed:
        #     print(f"Skipping speaker {i}: {speaker} - All samples already processed")
        #     continue
        
        print(f'processing speaker {i}: {speaker}')
        with open(speaker, 'r') as f:
            speaker_samples = json.load(f)

        # Use the first sample as reference speaker
        prompt_data = speaker_samples[0]
        reference_audio = prompt_data['path']
        if 'vin27' in speaker: 
            reference_audio = reference_audio.replace("linhnt140/zero-shot-tts/preprocess_audio/vin27_16k", "thivt1/code/VoiceCraft/vin27_testset")

        # Process the rest of the samples
        for sample in tqdm(speaker_samples[1:], desc=f"Processing {speaker}", leave=False):
            original_audio_path = sample['path']
            if 'vin27' in speaker:
                original_audio_path = original_audio_path.replace("thivt1/code/VoiceCraft/vin27_testset", "thivt1/code/VoiceCraft/vin27_testset")
            original_audio_name = os.path.basename(original_audio_path)
            output_dir = os.path.join("/lustre/scratch/client/vinai/users/thivt1/code/VoiceCraft/test_set_final_results", os.path.basename(speaker).split('.')[0], original_audio_name.split('.')[0])
            os.makedirs(output_dir, exist_ok=True)
            text = sample['transcript']
            output_path = os.path.join(output_dir, "xtts_vivoice.wav")
            print(f'output_path: {output_path}')
            if not os.path.exists(output_path):
                run_inference(text, reference_audio, output_path)


if __name__ == '__main__':
    # infer 1 sample
    output_path = 'test_results/xtts_vivoice_long_ver.wav'
    run_inference(
        "Đến hơn mười hai giờ, trao đổi với phóng viên Dân trí, một lãnh đạo phường Dương Đông cho biết nước đang rút dần. Do tình hình thời tiết xấu, mưa to nên tàu đi Phú Quốc và các đảo khác ở Kiên Giang đều tạm ngưng hoạt động. Hơn một trăm khách du lịch đang bị kẹt trên đảo. Một người dân ở đường Cách mạng tháng Tám cho hay, mưa lớn kéo dài, các cống không thoát nước kịp khiến nước mưa tràn vào nhà dân.",
        "test_results/quang-nam_3512618_127.wav",
        output_path
    )
    # print(f"Generated audio saved to: {output_path}")

    # infer_test_set()

