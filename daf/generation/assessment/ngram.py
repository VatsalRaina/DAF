"""
Calculate ngram scores between ground-truth and generated distractors.
"""

import argparse
import string
import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--data_path', type=str, help='Path of generated and ground-truth distractors.')
parser.add_argument('--save_path', type=str, help='directory to save generated boxplot.')

def main(args):

    with open(args.data_path, 'rb') as f:
        all_data = json.load(f)


    # A = best distractor
    scoresA = {'bleu-1': [], 'bleu-4': []}

    for ex in all_data:
        generated_distractors = ex['generated_distractors']
        unique_generated_distractors = set(generated_distractors)
        options = ex['options']
        lab = ex['label']
        ground_truth_distractors = options.pop(lab)
        # Clean out punctuation, lower case and convert to the form of a list
        clean_generated_distractors = []
        for s in unique_generated_distractors:
            clean_generated_distractors.append( s.lower().translate(str.maketrans('', '', string.punctuation)).split() )
        clean_ground_truth_distractors = []
        for s in ground_truth_distractors:
            clean_ground_truth_distractors.append( s.lower().translate(str.maketrans('', '', string.punctuation)).split() )
        bleu1_scores = []
        bleu4_scores = []
        for gen_distractor in unique_generated_distractors:
            bleu1 = sentence_bleu(clean_ground_truth_distractors, gen_distractor, weights=(1, 0, 0, 0))
            bleu4 = sentence_bleu(clean_ground_truth_distractors, gen_distractor, weights=(0, 0, 0, 1))
            bleu1_scores.append(bleu1)
            bleu4_scores.append(bleu4)

        scoresA['bleu-1'].append(min(bleu1_scores))
        scoresA['bleu-4'].append(min(bleu4_scores))

    best_bleu1_scores = np.asarray(scoresA['bleu-1'])
    best_bleu4_scores = np.asarray(scoresA['bleu-4'])

    print("Mean bleu-1 score:", np.mean(best_bleu1_scores))
    print("Mean bleu-4 score:", np.mean(best_bleu4_scores))

    # Plot boxplots
    data = []
    for b1, b4 in zip(best_bleu1_scores, best_bleu4_scores):
        data.append({'Metric': 'BLEU-1', 'Score': b1})
        data.append({'Metric': 'BLEU-4', 'Score': b4})
    df = pd.DataFrame(data)
    sns.violinplot(data=df, x='Score', y='Metric')
    plt.savefig(args.save_path + 'ngram.png')




if __name__ == '__main__':
    args = parser.parse_args()
    main(args)