import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

# Logging parameters
RUN_NAME = "SACHNOI_FINETUNE_500h_360k_5k"
PROJECT_NAME = "XTTS_trainer"
DASHBOARD_LOGGER = "tensorboard"
LOGGER_URI = None

# Set here the path that the checkpoints will be saved. Default: ./run/training/
OUT_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "run", "training")

# Training Parameters
# for multi-gpu training please make it False
OPTIMIZER_WD_ONLY_ON_WEIGHTS = True
START_WITH_EVAL = True  # if True it will star with evaluation
BATCH_SIZE = 4  # set here the batch size
GRAD_ACUMM_STEPS = 84  # set here the grad accumulation steps
# Note: we recommend that BATCH_SIZE * GRAD_ACUMM_STEPS need to be at least 252 for more efficient training. You can increase/decrease BATCH_SIZE but then set GRAD_ACUMM_STEPS accordingly.

# Define here the dataset that you want to use for the fine-tuning on.
# config_dataset = BaseDatasetConfig(
#     formatter="ljspeech",
#     dataset_name="ljspeech",
#     path="/raid/datasets/LJSpeech-1.1_24khz/",
#     meta_file_train="/raid/datasets/LJSpeech-1.1_24khz/metadata.csv",
#     language="en",
# )

# VIN27_PATH = os.path.join(CURRENT_PATH, "VIN27")
# VIN27_PATH = "/lustre/scratch/client/vinai/users/linhnt140/zero-shot-tts/preprocess_audio/vin27_16k"
# vin27_dataset = BaseDatasetConfig(
#     path=VIN27_PATH,
#     meta_file_train="train.json",
#     meta_file_val="val.json",
#     language='vi',
#     dataset_name="VIN27",
#     formatter="vin27",
# )

SACH_NOI_PATH = "/lustre/scratch/client/vinai/users/thivt1/code/oneshot"
sach_noi_config = BaseDatasetConfig(
    path=SACH_NOI_PATH,
    # meta_file_train="/lustre/scratch/client/vinai/users/thivt1/code/oneshot/step20_train_xtts.csv",
    # meta_file_val="/lustre/scratch/client/vinai/users/thivt1/code/oneshot/step20_val_xtts.csv",
    meta_file_train='/lustre/scratch/client/vinai/users/thivt1/code/TTS/recipes/ljspeech/xtts_v2/SACHNOI/train_500h.csv',
    meta_file_val='/lustre/scratch/client/vinai/users/thivt1/code/TTS/recipes/ljspeech/xtts_v2/SACHNOI/validation_500h.csv',
    language='vi',
    dataset_name="SACH_NOI",
    formatter="coqui",
)

# Add here the configs of the datasets
DATASETS_CONFIG_LIST = [sach_noi_config]


# Define the path where XTTS v2.0.1 files will be downloaded
CHECKPOINTS_OUT_PATH = os.path.join(
    OUT_PATH, "XTTS_v2.0_original_model_files/")
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)


# DVAE files
DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

# Set the path to the downloaded files
DVAE_CHECKPOINT = os.path.join(
    CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
# DVAE_CHECKPOINT = '/lustre/scratch/client/vinai/users/thivt1/code/XTTSv2-Finetuning-for-New-Languages/checkpoints/finetuned/dvae.pth'
print(f'dvae checkpoint: {DVAE_CHECKPOINT}')
MEL_NORM_FILE = os.path.join(
    CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))
print(f'mel norm file: {MEL_NORM_FILE}')

# download DVAE files if needed
if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print(" > Downloading DVAE files!")
    ModelManager._download_model_files(
        [MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)


# Download XTTS v2.0 checkpoint if needed
TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

# XTTS transfer learning parameters: You we need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
# TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))  # vocab.json file
TOKENIZER_FILE = "./vocab.json"
print(f'tokenizer file: {TOKENIZER_FILE}')
# XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))  # model.pth file
# XTTS_CHECKPOINT = "run/training/SACHNOI_FINETUNE_1,5K-October-12-2024_04+34PM-0000000/checkpoint_15000.pth"
XTTS_CHECKPOINT = 'run/training/SACHNOI_FINETUNE_1,5K_FROM_CKPT_SHORT_AUDIO_360K-October-12-2024_06+33PM-0000000/5k.pth'
print(f'xtts checkpoint: {XTTS_CHECKPOINT}')

# download XTTS v2.0 files if needed
if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
    print(" > Downloading XTTS v2.0 files!")
    ModelManager._download_model_files(
        [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
    )


# Training sentences generations
SPEAKER_REFERENCE = [
    "/lustre/scratch/client/vinai/users/thivt1/code/oneshot/big_processed_data/Đàm_Thanh_Phượng/Cuộc_Đại_Lạm_Phát_/11._DLP_C05B_Su_Phuc_Hoi_Cua_Chu_Nghia_Tu_Ban/chunk-2485_48-2498_7_trimmed_norm_float32.wav",
    "/lustre/scratch/client/vinai/users/thivt1/code/oneshot/big_processed_data/Nguyễn_Văn_Khỏa/Thần_Thoại_Hy_Lạp/than-thoai-9/chunk-2541_74-2554_64_trimmed_norm_float32.wav"
]
LANGUAGE = "vi"


def main():
    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,  # 3 secs
        debug_loading_failures=False,
        max_wav_length=441000,  # 20 seconds
        max_text_length=450,  # in characters
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        # checkpoint path of the model that you want to fine-tune
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    # define audio config
    audio_config = XttsAudioConfig(
        sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    # training parameters config
    config = GPTTrainerConfig(
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="""
            GPT XTTS training
            """,
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=8,
        eval_split_max_size=256,
        print_step=500,
        plot_step=500,
        log_model_step=500,
        save_step=3360,
        save_n_checkpoints=20,
        save_checkpoints=True,
        # target_loss="loss",
        print_eval=False,
        # Optimizer values like tortoise, pytorch implementation with modifications to not apply WD to non-weight parameters.
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96],
                          "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,  # learning rate
        lr_scheduler="MultiStepLR",
        # it was adjusted accordly for the new step scheme
        lr_scheduler_params={"milestones": [
            50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[
            {
                "text": "minh lê đang làm gì thế",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "minh lê đang ăn cơm và đánh bóng bàn",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
        ],
    )

    # init the model from config
    model = GPTTrainer.init_from_config(config)

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    print(f'len(train_samples): {len(train_samples)}')
    print(f'len(eval_samples): {len(eval_samples)}')

    vin27_train = 0
    vin27_eval = 0
    sachnoi_train = 0
    sachnoi_eval = 0

    for sample in train_samples:
        audio_file = sample['audio_file']
        if "vin27" in audio_file:
            vin27_train += 1
        elif "oneshot" in audio_file:
            sachnoi_train += 1
        else:
            print(f'wtf is going on with {audio_file}')

    for sample in eval_samples:
        audio_file = sample['audio_file']
        if "vin27_16k" in audio_file:
            vin27_eval += 1
        elif "oneshot" in audio_file:
            sachnoi_eval += 1
        else:
            print(f'wtf is going on with {audio_file}')

    print(f'# vin27 samples in trainset: {vin27_train}')
    print(f'# vin27 samples in eval: {vin27_eval}')

    print(f'# sachnoi samples in trainset: {sachnoi_train}')
    print(f'# sachnoi samples in eval: {sachnoi_eval}')

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,  # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
