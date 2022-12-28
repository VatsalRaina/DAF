"""
Construct the training dataset with either the worst distractor or the best distractor (i.e no diversity)
according to some distractor ranking process.
"""

import argparse
import json

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--generated_path', type=str, help='Load path of all generated distractors')
parser.add_argument('--selection', type=str, default='random', help='Method of selecting distractor')
parser.add_argument('--save_dir', type=str, help='Path to save generated text')

def asLetter(x):
    if x==0:
        return "A"
    if x==1:
        return "B"
    if x==2:
        return "C"
    if x==3:
        return "D"

class Selector:
    def __init__(self, all_data_path):

        with open(all_data_path) as f:
            self.all_data = json.load(f)

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

    def select(self, ranking):
        if ranking == 'random':
            return self._random_select()
        else:
            return None


def main(args):

    selector = Selector(args.generated_path)
    final_data = selector.select(args.selection)

    with open(args.save_dir + 'generated.json', 'w') as f:
        json.dump(final_data, f)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)