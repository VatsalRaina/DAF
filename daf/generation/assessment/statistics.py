"""
Calculate number of unique distractors generated per question.
"""

import argparse
import os
import sys
import json

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--data_path', type=str, help='Path of generated and ground-truth distractors.')

def main(args):

    with open(args.data_path, 'rb') as f:
        all_data = json.load(f)

    num_unique_distractors = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}

    for ex in all_data:
        generated_distractors = ex['generated_distractors']
        num_unique = len(set(generated_distractors))
        num_unique_distractors[num_unique] += 1

    print("Unique distractor distribution:")
    print(num_unique_distractors)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)