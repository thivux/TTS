import argparse
import os
import json
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import glob


def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_model(xtts_checkpoint, xtts_config, xtts_vocab):
    clear_gpu_cache()
    config = XttsConfig()
    config.load_json(xtts_config)
    xtts_model = Xtts.init_from_config(config)
    print("Loading XTTS model...")
    xtts_model.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False)
    if torch.cuda.is_available():
        xtts_model.cuda()
    print("Model loaded!")
    return xtts_model

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_tts(model, lang, tts_text, speaker_audio_file, output_path, seed=None):
    if seed is not None:
        set_seed(seed)

    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=speaker_audio_file,
        gpt_cond_len=model.config.gpt_cond_len,
        max_ref_length=model.config.max_ref_len,
        sound_norm_refs=model.config.sound_norm_refs
    )
    out = model.inference(
        text=tts_text,
        language=lang,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=model.config.temperature,
        length_penalty=model.config.length_penalty,
        repetition_penalty=model.config.repetition_penalty,
        top_k=model.config.top_k,
        top_p=model.config.top_p,
        enable_text_splitting=True,
    )

    out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
    torchaudio.save(output_path, out["wav"], 24000)

def get_sublist(lst, index):
    # Ensure the index is within the allowed range
    if index < 0 or index > 3:
        raise ValueError("Index must be between 0 and 3.")
    
    # Determine the length of each part
    part_size = len(lst) // 4
    remainder = len(lst) % 4  # To account for uneven split
    
    # Calculate start and end positions for the sublist at the given index
    start = sum(part_size + (1 if i < remainder else 0) for i in range(index))
    end = start + part_size + (1 if index < remainder else 0)
    
    return lst[start:end]

def main(args):
    seed = 1 

    model = load_model(args.xtts_checkpoint, args.xtts_config, args.xtts_vocab)
    # models = load_model(args.xtts_checkpoint, args.xtts_config, args.xtts_vocab)

    # Get all JSON files in the test_set_5k folder
    json_files = sorted(glob.glob(f"{args.test_set_path}/*.json"))
    json_files = [file for file in json_files if "VIVOSDEV" not in file]
    # json_files = get_sublist(json_files, 0)

    print(json_files)
    print(f'there are {len(json_files)} speakers in test set')

    output_folder = "xtts_output_1410" 
    for i, speaker in enumerate(json_files):
        print(f'processing speaker {i}: {speaker}')
        with open(speaker, 'r') as f:
            speaker_samples = json.load(f)

        # Use the first sample as reference speaker
        prompt_data = speaker_samples[0]
        reference_audio = prompt_data['path']

        # Process the rest of the samples
        for sample in tqdm(speaker_samples[1:], desc=f"Processing {speaker}", leave=False):
            original_audio_path = sample['path']
            original_audio_name = os.path.basename(original_audio_path)
            output_dir = os.path.join("/lustre/scratch/client/vinai/users/thivt1/code/VoiceCraft/", output_folder, os.path.splitext(os.path.basename(speaker))[0], os.path.splitext(original_audio_name)[0])
            output_path = os.path.join(output_dir, "xtts_ft_360k_5k_500h_13k.wav")
            print(f'output_path: {output_path}')
            if os.path.exists(output_path):
                continue
            os.makedirs(output_dir, exist_ok=True)
            text = sample['transcript']
            run_tts(model, args.lang, text, reference_audio, output_path, seed=seed)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XTTS Batch Inference")
    parser.add_argument("--xtts_checkpoint", type=str, required=True, help="Path to the XTTS checkpoint")
    parser.add_argument("--xtts_config", type=str, required=True, help="Path to the XTTS config file")
    parser.add_argument("--xtts_vocab", type=str, required=True, help="Path to the XTTS vocab file")
    parser.add_argument("--lang", type=str, default="vi", help="Language code for the TTS (default: 'vi')")
    parser.add_argument("--test_set_path", type=str, required=True, help="Path to the test_set_5k folder containing JSON files")

    args = parser.parse_args()
    main(args)
