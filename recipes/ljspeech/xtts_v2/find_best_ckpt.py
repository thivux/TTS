import argparse
import os
import numpy as np
import torch
import glob
from tqdm import tqdm
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torchaudio


def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(xtts_checkpoint, xtts_config, xtts_vocab):
    clear_gpu_cache()
    config = XttsConfig()
    config.load_json(xtts_config)
    xtts_model = Xtts.init_from_config(config)
    print(f"Loading XTTS model from checkpoint: {xtts_checkpoint}...")
    xtts_model.load_checkpoint(
        config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False)
    if torch.cuda.is_available():
        xtts_model.cuda()
    print("Model loaded!")
    return xtts_model


def run_tts(model, lang, tts_text, speaker_audio_file, output_path):
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
    )

    out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
    torchaudio.save(output_path, out["wav"], 24000)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    set_seed(42)

    # Get all checkpoint .pth files in the specified folder
    checkpoint_files = sorted(glob.glob(f"{args.checkpoints_folder}/*.pth"))
    # checkpoint_files = [args.checkpoints_folder + '/' + x for x in ['checkpoint_60000.pth', 'checkpoint_120000.pth', 'checkpoint_130000.pth']]
    # checkpoint_files = [args.checkpoints_folder + '/' + x for x in ['checkpoint_85000.pth']]
    print(f"Found {len(checkpoint_files)} checkpoint files: {checkpoint_files}")

    # Loop through each checkpoint file
    for checkpoint in checkpoint_files:
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint))[
            0]  # Get the name without extension
        print(f"Processing checkpoint: {checkpoint_name}")

        # Create output folder for this checkpoint
        output_folder = "results_find_best_xtts_short"
        os.makedirs(output_folder, exist_ok=True)

        # Set output path for this checkpoint
        output_path = os.path.join(output_folder, f"{checkpoint_name}.wav")
        print(f'Output path: {output_path}')

        if os.path.exists(output_path):
            print(f"Output already exists for {output_path}, skipping...")
            continue

        # Load the model with the current checkpoint
        model = load_model(checkpoint, args.xtts_config, args.xtts_vocab)

        # Run TTS for the single sample
        run_tts(model, args.lang, args.text, args.speaker_audio, output_path)
        # Remove model from GPU and clear cache
        del model  # Delete the model to free up memory
        clear_gpu_cache()  # Clear GPU cache

        print(f"Finished processing checkpoint: {checkpoint_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XTTS Single Sample Inference with Multiple Checkpoints")
    parser.add_argument("--checkpoints_folder", type=str, required=True,
                        help="Path to the folder containing XTTS checkpoints (.pth files)")
    parser.add_argument("--xtts_config", type=str,
                        required=True, help="Path to the XTTS config file")
    parser.add_argument("--xtts_vocab", type=str,
                        required=True, help="Path to the XTTS vocab file")
    parser.add_argument("--lang", type=str, default="vi",
                        help="Language code for the TTS (default: 'vi')")
    parser.add_argument("--speaker_audio", type=str, required=True,
                        help="Path to the reference speaker audio file")
    parser.add_argument("--text", type=str, required=True,
                        help="Text to synthesize")

    args = parser.parse_args()
    main(args)
