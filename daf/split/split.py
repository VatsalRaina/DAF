"""
Split the RACE++ training set into train-comprehension and train-generation.
20% for comprehension and 80% for generation (according to contexts).
No overlap of contexts.
"""

import argparse
import json
import random

COMP_SPLIT = 0.2

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
parser.add_argument('--train_data_path', type=str, help='Load path of training data')
parser.add_argument('--save_dir', type=str, help='Directory to save generated splits')


def main(args):

    seed = args.seed
    random.seed(seed)

    with open(args.train_data_path + "middle.json") as f:
        middle_data = json.load(f)
    with open(args.train_data_path + "high.json") as f:
        high_data = json.load(f)
    with open(args.train_data_path + "college.json") as f:
        college_data = json.load(f)
    train_data = middle_data + high_data + college_data

    random.shuffle(train_data)

    cutoff = int(len(train_data) * COMP_SPLIT)
    train_comprehension = train_data[:cutoff]
    train_generation = train_data[cutoff:]

    with open(args.save_dir + 'train-comprehension.json', 'w') as f:
        json.dump(train_comprehension)

    with open(args.save_dir + 'train-generation.json', 'w') as f:
        json.dump(train_generation)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)