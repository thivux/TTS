import os

import torch
from trainer import Trainer, TrainerArgs

from TTS.bin.compute_embeddings import compute_embeddings
from TTS.bin.resample import resample_files
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
from TTS.utils.downloaders import download_vctk

torch.set_num_threads(24)

# pylint: disable=W0105
"""
    This recipe replicates the first experiment proposed in the YourTTS paper (https://arxiv.org/abs/2112.02418).
    YourTTS model is based on the VITS model however it uses external speaker embeddings extracted from a pre-trained speaker encoder and has small architecture changes.
    In addition, YourTTS can be trained in multilingual data, however, this recipe replicates the single language training using the VCTK dataset.
    If you are interested in multilingual training, we have commented on parameters on the VitsArgs class instance that should be enabled for multilingual training.
    In addition, you will need to add the extra datasets following the VCTK as an example.
"""
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

# Name of the run for the Trainer
RUN_NAME = "YourTTS-MIX-VIN27-SACH-NOI-PHONE-TEST-MULTIGPU"

# Path where you want to save the models outputs (configs, checkpoints and tensorboard logs)
# "/raid/coqui/Checkpoints/original-YourTTS/"
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# If you want to do transfer learning and speedup your training you can set here the path to the original YourTTS model
RESTORE_PATH = None 

# This paramter is useful to debug, it skips the training epochs and just do the evaluation  and produce the test sentences
SKIP_TRAIN_EPOCH = False

# Set here the batch size to be used in training and evaluation
BATCH_SIZE = 40 

# Training Sampling rate and the target sampling rate for resampling the downloaded dataset (Note: If you change this you might need to redownload the dataset !!)
# Note: If you add new datasets, please make sure that the dataset sampling rate and this parameter are matching, otherwise resample your audios
SAMPLE_RATE = 16000

# Max audio length in seconds to be used in training (every audio bigger than it wilssl be ignored)
MAX_AUDIO_LEN_IN_SECONDS = 25

# Download VCTK dataset
VCTK_DOWNLOAD_PATH = os.path.join(CURRENT_PATH, "VCTK")
# Define the number of threads used during the audio resampling
NUM_RESAMPLE_THREADS = 10
# Check if VCTK dataset is not already downloaded, if not download it
if not os.path.exists(VCTK_DOWNLOAD_PATH):
    print(">>> Downloading VCTK dataset:")
    download_vctk(VCTK_DOWNLOAD_PATH)
    resample_files(VCTK_DOWNLOAD_PATH, SAMPLE_RATE,
                   file_ext="flac", n_jobs=NUM_RESAMPLE_THREADS)

# init configs
vctk_config = BaseDatasetConfig(
    formatter="vctk",
    dataset_name="vctk",
    meta_file_train="",
    meta_file_val="",
    path=VCTK_DOWNLOAD_PATH,
    language="en-us",
    ignored_speakers=[
        "p261",
        "p225",
        "p294",
        "p347",
        "p238",
        "p234",
        "p248",
        "p335",
        "p245",
        "p326",
        "p302",
    ],  # Ignore the test speakers to full replicate the paper experiment
)

# SACH_NOI_PATH = os.path.join(CURRENT_PATH, "SACH_NOI")
# sach_noi_config = BaseDatasetConfig(
#     path=SACH_NOI_PATH,
#     meta_file_train="metadata_50speakers.txt",
#     language='vi',
#     dataset_name="SACH_NOI",
#     formatter="sach_noi",
#     ignored_speakers=[
#         "Trí_An",
#         "Hoàn_Lê",
#         "Quỳnh_Hái",
#         "Phạm_Công_Luận",
#         "Dan_Sullivan",
#     ]
# )

# VIN27_PATH = os.path.join(CURRENT_PATH, "VIN27")
# vin27_config = BaseDatasetConfig(
#     path=VIN27_PATH,
#     meta_file_train="updated_metadata.csv",
#     language='vi',
#     dataset_name="VIN27",
#     formatter="vin27",
#     ignored_speakers=[
#         "quang-nam_0327404", "quang-ngai_3596416", "khanh-hoa_0906516", # center
#         "ho-chi-minh_3540345", "hau-giang_3516917", "ho-chi-minh_0931102", # south
#         "hai-phong_3552914", "hai-phong_3564273", "hai-phong_3650991" # north
#     ]
# )


# Add here all datasets configs, in our case we just want to train with the VCTK dataset then we need to add just VCTK. Note: If you want to add new datasets, just add them here and it will automatically compute the speaker embeddings (d-vectors) for this new dataset :)
# DATASETS_CONFIG_LIST = [vin27_config, sach_noi_config, vctk_config]
DATASETS_CONFIG_LIST = [sach_noi_config, vctk_config]

# Extract speaker embeddings
# SPEAKER_ENCODER_CHECKPOINT_PATH = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar"
SPEAKER_ENCODER_CHECKPOINT_PATH = os.path.join(
    CURRENT_PATH, "tts_models--multilingual--multi-dataset--your_tts/model_se.pth.tar")
# SPEAKER_ENCODER_CONFIG_PATH = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json"
SPEAKER_ENCODER_CONFIG_PATH = os.path.join(
    CURRENT_PATH, "tts_models--multilingual--multi-dataset--your_tts/config_se.json")

D_VECTOR_FILES = []  # List of speaker embeddings/d-vectors to be used during the training

# Iterates all the dataset configs checking if the speakers embeddings are already computated, if not compute it
for dataset_conf in DATASETS_CONFIG_LIST:
    # Check if the embeddings weren't already computed, if not compute it
    embeddings_file = os.path.join(dataset_conf.path, "speakers.pth")
    if not os.path.isfile(embeddings_file):
        print(
            f">>> Computing the speaker embeddings for the {dataset_conf.dataset_name} dataset")
        compute_embeddings(
            SPEAKER_ENCODER_CHECKPOINT_PATH,
            SPEAKER_ENCODER_CONFIG_PATH,
            embeddings_file,
            old_speakers_file=None,
            config_dataset_path=None,
            formatter_name=dataset_conf.formatter,
            dataset_name=dataset_conf.dataset_name,
            dataset_path=dataset_conf.path,
            meta_file_train=dataset_conf.meta_file_train,
            meta_file_val=dataset_conf.meta_file_val,
            disable_cuda=False,
            no_eval=False,
        )
    D_VECTOR_FILES.append(embeddings_file)


# Audio config used in training.
audio_config = VitsAudioConfig(
    sample_rate=SAMPLE_RATE,
    hop_length=256,
    win_length=1024,
    fft_size=1024,
    mel_fmin=0.0,
    mel_fmax=None,
    num_mels=80,
)

# Init VITSArgs setting the arguments that are needed for the YourTTS model
model_args = VitsArgs(
    d_vector_file=D_VECTOR_FILES,
    use_d_vector_file=True,
    d_vector_dim=512,
    num_layers_text_encoder=10,
    speaker_encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
    speaker_encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
    resblock_type_decoder="2",  # In the paper, we accidentally trained the YourTTS using ResNet blocks type 2, if you like you can use the ResNet blocks type 1 like the VITS model
    # Useful parameters to enable the Speaker Consistency Loss (SCL) described in the paper
    use_speaker_encoder_as_loss=True,
    # Useful parameters to enable multilingual training
    use_language_embedding=True,
    embedded_language_dim=4,
)

# General training config, here you can change the batch size and others useful parameters
config = VitsConfig(
    output_path=OUT_PATH,
    model_args=model_args,
    run_name=RUN_NAME,
    project_name="YourTTS",
    run_description="""
            - Original YourTTS trained using VCTK dataset
        """,
    dashboard_logger="tensorboard",
    logger_uri=None,
    audio=audio_config,
    batch_size=BATCH_SIZE,
    batch_group_size=48,
    eval_batch_size=BATCH_SIZE,
    num_loader_workers=8,
    eval_split_max_size=256,
    print_step=50,
    plot_step=100,
    log_model_step=1000,
    save_step=5000,
    save_n_checkpoints=2,
    save_checkpoints=True,
    target_loss="loss_1",  # NOTE: best ckpt is based on this metric
    print_eval=False,
    use_phonemes=True,
    phonemizer="multi_phonemizer",
    phoneme_language=None,
    compute_input_seq_cache=True,
    add_blank=True,
    text_cleaner="multilingual_cleaners",
    # lr_gen=5e-5,
    # lr_disc=5e-5,
    phoneme_cache_path="./PHONE",
    precompute_num_workers=12,
    start_by_longest=True,
    datasets=DATASETS_CONFIG_LIST,
    cudnn_benchmark=False,
    max_audio_len=SAMPLE_RATE * MAX_AUDIO_LEN_IN_SECONDS,
    mixed_precision=False,
    test_sentences=[
        # SACH_NOI
        [
            "và thường là hơn cả sự mong đợi. qua thực hành, theo dõi và kiểm chứng từ bản thân cũng như từ nhiều bạn hữu đã tiếp cận và áp dụng phương pháp này.",
            "Joe_Vitale", 
            None,
            'vi'
        ],
        [
            "nhà bác ấy có nước mưa múc vào đền mẫu, ngọt lịm như đường phèn. ở tình nhỏ, người ta dễ quen nhau lắm. gia đình thúy và gia đình thằng vũ, thằng côn, thằng luyến đều là chỗ thân tình.",
            "Khắc_Thiệu", 
            None,
            'vi'
        ],
        [
            "những nỗi đau ấy tuyệt đối không thể nào xóa bỏ. tuy nhiên, họ vẫn đối diện với nó, và trong cơn vùng vẫy, họ không chỉ cảm nhận mà còn học được nhiều điều.",
            "Hân_Phạm", 
            None,
            'vi'
        ],
        [
            "bạn sẽ tránh được tổn thương, lo lắng mà tập trung vào phương án giải quyết. trẻ lớn không có nghĩa là thời gian phát triển của chúng đã được cố định. vì vậy, trẻ vẫn có thể học hỏi được.",
            "Thu_Hà", 
            None,
            'vi'
        ],
        # # VIN27
        # [

        #     "Thứ ma thuật đen của họ khiến không ít kẻ thách thức phải khiếp sợ",
        #     'hai-phong_3586631', # north
        #     None,
        #     'vi'
        # ],
        # [
        #     "có một cách này hay lắm không biết anh có muốn nghe không",
        #     'vinh-phuc_3544849', # north
        #     None,
        #     'vi'
        # ],
        # [
        #     "mỗi ngày phục vụ hàng trăm suất cơm và chịu lỗ cả trăm triệu đồng",
        #     'ho-chi-minh_3595510', # south
        #     None,
        #     'vi'
        # ],
        # [
        #     "Khi tiếng còi mãn cuộc vang lên, những cầu thủ vừa thêm một lần lọt vào trận đấu cuối cùng, bủa ra mọi phía để ăn mừng cùng khán giả. Họ ôm nhau nhảy múa, trượt dài trên mặt cỏ.",
        #     'khanh-hoa_3547359', # center
        #     None,
        #     'vi'
        # ],
        # VCTK
        [
            "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
            "VCTK_p277",
            None,
            "en-us",
        ],
        [
            "Be a voice, not an echo.",
            "VCTK_p239",
            None,
            "en-us",
        ],
        [
            "I'm sorry Dave. I'm afraid I can't do that.",
            "VCTK_p258",
            None,
            "en-us",
        ],
        [
            "This cake is great. It's so delicious and moist.",
            "VCTK_p244",
            None,
            "en-us",
        ],
        [
            "Prior to November 22, 1963.",
            "VCTK_p305",
            None,
            "en-us",
        ]
    ],

    # Enable the weighted sampler
    use_weighted_sampler=True,
    # Ensures that all speakers are seen in the training batch equally no matter how many samples each speaker has
    weighted_sampler_attrs={"speaker_name": 1.0},
    weighted_sampler_multipliers={},
    # It defines the Speaker Consistency Loss (SCL) α to 9 like the paper
    speaker_encoder_loss_alpha=9.0,
)

# Load all the datasets samples and split traning and evaluation sets
train_samples, eval_samples = load_tts_samples(
    config.datasets,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# # ========== NEW CODE ===================
# from TTS.tts.utils.languages import LanguageManager
# from TTS.tts.utils.speakers import SpeakerManager
# from TTS.tts.utils.text.tokenizer import TTSTokenizer
# from TTS.utils.audio import AudioProcessor

# # force the convertion of the custom characters to a config attribute
# config.from_dict(config.to_dict())

# # init audio processor
# ap = AudioProcessor(**config.audio.to_dict())

# # init speaker manager for multi-speaker training
# # it maps speaker-id to speaker-name in the model and data-loader
# speaker_manager = SpeakerManager()
# speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
# config.model_args.num_speakers = speaker_manager.num_speakers

# language_manager = LanguageManager(config=config)
# config.model_args.num_languages = language_manager.num_languages

# # INITIALIZE THE TOKENIZER
# # Tokenizer is used to convert text to sequences of token IDs.
# # config is updated with the default characters if not defined in the config.
# tokenizer, config = TTSTokenizer.init_from_config(config)

# # init model
# model = Vits(config, ap, tokenizer, speaker_manager, language_manager)

# # init the trainer and 🚀
# trainer = Trainer(
#     TrainerArgs(), config, output_path=OUT_PATH, model=model, train_samples=train_samples, eval_samples=eval_samples
# )

# ========== NEW CODE ===================


# ============== OLD CODE ===================
# Init the model
model = Vits.init_from_config(config)

# Init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(restore_path=RESTORE_PATH, skip_train_epoch=SKIP_TRAIN_EPOCH),
    config,
    output_path=OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

# ============== OLD CODE ===================

# number of samples in train & valid, for vi & en
vin27_train = 0
vin27_eval = 0
sachnoi_train = 0
sachnoi_eval = 0
vctk_train = 0
vctk_eval = 0

for sample in train_samples:
    audio_file = sample['audio_file']
    if "vin27_16k" in audio_file:
        vin27_train += 1
    elif "SACH_NOI" in audio_file:
        sachnoi_train += 1
    elif "VCTK" in audio_file: 
        vctk_train += 1

for sample in eval_samples:
    audio_file = sample['audio_file']
    if "vin27_16k" in audio_file:
        vin27_eval += 1
    elif "SACH_NOI" in audio_file:
        sachnoi_eval += 1
    elif "VCTK" in audio_file: 
        vctk_eval += 1

print(f'# vin27 samples in trainset: {vin27_train}')
print(f'# vin27 samples in eval: {vin27_eval}')

print(f'# sachnoi samples in trainset: {sachnoi_train}')
print(f'# sachnoi samples in eval: {sachnoi_eval}')

print(f'# vctk samples in trainset: {vctk_train}')
print(f'# vctk samples in eval: {vctk_eval}')

trainer.fit()
