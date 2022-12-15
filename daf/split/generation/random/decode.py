"""
The generation technique here is that the ground-truth distractors are just randomly replaced with distractors selected from different questions.
All distractors are completely randomly sampled.
"""

import argparse
import json
import random


parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
parser.add_argument('--test_data_path', type=str, help='Load path of evaluation data')
parser.add_argument('--save_dir', type=str, help='Directory to save generated splits')


def main(args):

    seed = args.seed
    random.seed(seed)

    with open(args.test_data_path) as f:
        test_data = json.load(f)

    def asNum(x):
        if x=="A":
            return 0
        if x=="B":
            return 1
        if x=="C":
            return 2
        if x=="D":
            return 3

    generated_text = []
    for item in test_data:
        context = item["article"]
        questions = item["questions"]
        answers = item["answers"]
        options = item["options"]
        # We will replace the options
        new_options = []
        for count, opts in enumerate(options):
            lab = asNum(answers[count])
            new_opts = []
            for opt_num, opt in enumerate(opts):
                if opt_num == lab:
                    new_opts.append(opt)
                else:
                    # randomly sample a distractor
                    sample_item = test_data[random.randint(0,len(test_data))]
                    sample_options = sample_item[random.randint(0,len(sample_item))]
                    sample_distractor = sample_options[random.randint(0,4)]
                    new_opts.append(sample_distractor)
            new_options.append(new_opts)
        curr = {"article": context, "questions": questions, "answers": answers, "options": new_options}
        generated_text.append(curr)

    with open(args.save_dir + 'generated.json', 'w') as f:
        json.dump(generated_text, f)



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)





