'''
find character limit for sach noi dataset
'''

import json

if __name__ == "__main__": 
    json_train_path = './SACH_NOI/sach_noi_train.json'
    with open(json_train_path, 'r', encoding='utf-8') as file:
        train_data = json.load(file)
    
    max_char_len = 0
    for sample in train_data: 
        text = sample['transcript']
        # find char len 
        char_len = len(text)
        if char_len > max_char_len: 
            max_char_len = char_len

    print(f'max char len: {max_char_len}')
        