"""
Set the number of distractors to generate per question
"""

import argparse
import os
import sys
import json
import torch
from transformers import T5Tokenizer

MAXLEN = 100

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--model_path', type=str, help='Load path of trained model')
parser.add_argument('--num_distractors', type=int, default=1, help='Number of distractors')
parser.add_argument('--test_data_path', type=str, help='Load path of test data')
parser.add_argument('--save_path', type=str, help='Path to save generated text')

# Set device
def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Choose device
    device = get_default_device()

    with open(args.test_data_path + "middle.json") as f:
        middle_data = json.load(f)
    with open(args.test_data_path + "high.json") as f:
        high_data = json.load(f)
    test_data = middle_data + high_data    

    tokenizer = T5Tokenizer.from_pretrained("t5-base", truncation_side='left')

    count = 0
    model = torch.load(args.model_path, map_location=device)
    model.eval().to(device)

    def asNum(x):
        if x=="A":
            return 0
        if x=="B":
            return 1
        if x=="C":
            return 2
        if x=="D":
            return 3

    all_generated_text = []

    for item in test_data:
        print(count, len(test_data))
        count+=1
        # if count == 10:
        #     break
        context = item["article"]
        questions = item["questions"]
        answers = item["answers"]
        options = item["options"]
        for qu_num in range(len(questions)):
            current = {}
            lab = asNum(answers[qu_num])
            question = questions[qu_num]
            opts = options[qu_num]
            corr_opt = opts[lab]
            current['context'] = context
            current['question'] = question
            current['options'] = opts
            current['label'] = lab
            current['generated_distractors'] = []
            combo = context + " [SEP] " + question + " [SEP] " + corr_opt
            input_encodings_dict = tokenizer(combo, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
            for _ in range(args.num_distractors):
                all_generated_ids = model.generate(
                    input_ids=input_encodings_dict['input_ids'].to(device),
                    attention_mask=input_encodings_dict['attention_mask'].to(device),
                    do_sample=True,
                    top_k=50, 
                    top_p=0.95,          
                    max_length=MAXLEN,
                    repetition_penalty=2.5,
                    length_penalty=1.0,
                    early_stopping=True,
                    use_cache=True
                )
                for generated_ids in all_generated_ids:
                    genDist = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    current['generated_distractors'].append(genDist)
            all_generated_text.append(current)
    with open(args.save_path + 'generated.json', 'w') as f:
        json.dump(all_generated_text, f)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)