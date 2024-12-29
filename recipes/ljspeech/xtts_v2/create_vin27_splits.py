import json
from tqdm import tqdm


if __name__ == '__main__':
    # dev set
    with open("/lustre/scratch/client/vinai/users/thivt1/code/oneshot/linh_transfer/vin27_dev.jsonl", 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
        vin27_validation = set([item['path'].replace(
            "vin27", "vin27_16k") for item in data])
    print(f'first 5 samples in vin27 validation: {list(vin27_validation)[:5]}\n')

    # test set
    with open("/lustre/scratch/client/vinai/users/thivt1/code/oneshot/linh_transfer/vin27_test.jsonl", 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
        vin27_test = set([item['path'].replace("vin27", "vin27_16k")
                         for item in data])
    print(f'first 5 samples in vin27 test: {list(vin27_test)[:5]}\n')

    # full data
    with open('/lustre/scratch/client/vinai/users/thivt1/code/TTS/recipes/ljspeech/xtts_v2/VIN27/full_metadata.csv', 'r', encoding='utf-8') as file:
        full_data = json.load(file)

    paths = [sample['path'] for sample in full_data]
    print(f'first 5 samples in vin27 full data: {paths[:5]}\n')

    vin27_val_list = []
    vin27_test_list = []
    vin27_train_list = []
    for sample in tqdm(full_data): 
        path = sample['path']
        # if path == '/lustre/scratch/client/vinai/users/linhnt140/zero-shot-tts/preprocess_audio/vin27_16k/thua-thien---hue/3500217/113.wav':
        #     breakpoint()
        if path in vin27_validation:
            vin27_val_list.append(sample)
        elif path in vin27_test:
            vin27_test_list.append(sample)
        else: 
            vin27_train_list.append(sample)

    print(f'vin27 train: {len(vin27_train_list)}')
    print(f'vin27 val: {len(vin27_val_list)}')
    print(f'vin27 test: {len(vin27_test_list)}')
    
    # save train, val & test set 
    with open('./VIN27/train.json', 'w', encoding='utf-8') as file:
        json.dump(vin27_train_list, file, ensure_ascii=False, indent=4)

    with open('./VIN27/val.json', 'w', encoding='utf-8') as file:
        json.dump(vin27_val_list, file, ensure_ascii=False, indent=4)

    with open('./VIN27/test.json', 'w', encoding='utf-8') as file:
        json.dump(vin27_test_list, file, ensure_ascii=False, indent=4)
