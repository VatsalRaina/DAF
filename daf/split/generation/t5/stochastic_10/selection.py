"""
Construct the training dataset with either the worst distractor or the best distractor (i.e no diversity)
according to some distractor ranking process.
"""

import argparse
import json
import torch
from transformers import ElectraTokenizer, ElectraForMultipleChoice
from keras.preprocessing.sequence import pad_sequences
import numpy as np

MAXLEN = 512

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--generated_path', type=str, help='Load path of all generated distractors')
parser.add_argument('--selection', type=str, default='random', help='Method of selecting distractor')
parser.add_argument('--mrc_model_path', type=str, default=None, help='If selection is mrc, then pass path of mrc model')
parser.add_argument('--save_dir', type=str, help='Path to save generated text')

def asLetter(x):
    letters = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    if x not in letters:
        return 'Invalid input'
    return letters[x]

# Set device
def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')


class Selector:
    def __init__(self, all_data_path, mrc_model_path=None):

        with open(all_data_path) as f:
            self.all_data = json.load(f)

        if mrc_model_path is not None:
            device = get_default_device()
            electra_large = "google/electra-large-discriminator"
            tokenizer = ElectraTokenizer.from_pretrained(electra_large, do_lower_case=True)
            model = torch.load(mrc_model_path, map_location=device)
            model.eval().to(device)
            self.model = model
            self.tokenizer = tokenizer
            self.device = device

    def _random_select(self):
        processed_data = []
        for ex in self.all_data:
            new_opts = []
            for opt_num, opt in enumerate(ex['options']):
                if opt_num == ex['label']:
                    new_opts.append(opt)
                else:
                    new_opts.append(ex['generated_distractors'][0])
            current = {"article": ex['context'], "questions":[ex['question']], "answers":[asLetter(ex['label'])], "options": [new_opts]}
            processed_data.append(current)
        return processed_data

    def _mrc_rank(self, context, question, distractors):
        four_inp_ids = []
        four_tok_type_ids = []
        for opt in distractors:
            combo = context + " [SEP] " + question + " " + opt
            inp_ids = self.tokenizer.encode(combo)
            if len(inp_ids)>512:
                inp_ids = inp_ids[-512:]
            tok_type_ids = [0 if i<= inp_ids.index(102) else 1 for i in range(len(inp_ids))]
            four_inp_ids.append(inp_ids)
            four_tok_type_ids.append(tok_type_ids)
        four_inp_ids = pad_sequences(four_inp_ids, maxlen=MAXLEN, dtype="long", value=0, truncating="post", padding="post")
        four_tok_type_ids = pad_sequences(four_tok_type_ids, maxlen=MAXLEN, dtype="long", value=0, truncating="post", padding="post")
        input_ids = [four_inp_ids]
        token_type_ids = [four_tok_type_ids]
        # Create attention masks
        attention_masks = []
        for sen in input_ids:
            sen_attention_masks = []
            for opt in sen:
                att_mask = [int(token_id > 0) for token_id in opt]
                sen_attention_masks.append(att_mask)
            attention_masks.append(sen_attention_masks)
        # Convert to torch tensors
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.long().to(self.device)
        token_type_ids = torch.tensor(token_type_ids)
        token_type_ids = token_type_ids.long().to(self.device)
        attention_masks = torch.tensor(attention_masks)
        attention_masks = attention_masks.long().to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
        logits = outputs[0][0].detach().cpu().numpy()
        ordering = np.argsort(logits)
        ordered_distractors = []
        for pos in ordering:
            ordered_distractors.append(distractors[pos])
        return ordered_distractors

    def _save_for_discriminator(self, data):
        with open('generated_for_discriminator.json', 'w') as f:
            json.dump(data, f)

    def _mrc_select(self):
        processed_data = []
        processed_data_for_discriminator = []
        for count, ex in enumerate(self.all_data):
            print(count, len(self.all_data))
            context, question, generated_distractors = ex['context'], ex['question'], ex['generated_distractors']
            ranked_generated_distractors = self._mrc_rank(context, question, generated_distractors)
            new_opts = []
            for opt_num, opt in enumerate(ex['options']):
                if opt_num == ex['label']:
                    new_opts.append(opt)
                else:
                    new_opts.append(ranked_generated_distractors[0])
            current = {"article": ex['context'], "questions":[ex['question']], "answers":[asLetter(ex['label'])], "options": [new_opts]}
            processed_data.append(current)
            current_for_discriminator = ex
            current_for_discriminator['generated_distractors'] = ranked_generated_distractors
            processed_data_for_discriminator.append(current_for_discriminator)
        self._save_for_discriminator(processed_data_for_discriminator)
        return processed_data


    def select(self, ranking):
        if ranking == 'random':
            return self._random_select()
        else:
            return self._mrc_select()


def main(args):

    selector = Selector(args.generated_path, args.mrc_model_path)
    final_data = selector.select(args.selection)

    with open(args.save_dir + 'generated.json', 'w') as f:
        json.dump(final_data, f)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)