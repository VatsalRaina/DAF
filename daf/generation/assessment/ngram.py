"""
Calculate ngram scores between ground-truth and generated distractors.
"""

import argparse
import string
import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
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
    scoresA = {'bleu-1': [], 'rouge-1': []}
    rouge = Rouge()
    for ex in all_data:
        generated_distractors = ex['generated_distractors']
        unique_generated_distractors = set(generated_distractors)
        options = ex['options']
        lab = ex['label']
        ground_truth_distractors = options.pop(lab)
        # Clean out punctuation, lower case and convert to the form of a list
        clean_generated_distractors = []
        for s in unique_generated_distractors:
            cleaned = s.lower().translate(str.maketrans('', '', string.punctuation)).split()
            if len(cleaned) != 0:
                clean_generated_distractors.append(cleaned)
        clean_ground_truth_distractors = []
        for s in ground_truth_distractors:
            cleaned = s.lower().translate(str.maketrans('', '', string.punctuation)).split()
            if len(cleaned) != 0:
                clean_ground_truth_distractors.append(cleaned)
        bleu1_scores = []
        rouge1_scores = []
        for gen_distractor in unique_generated_distractors:
            bleu1 = sentence_bleu(clean_ground_truth_distractors, gen_distractor, weights=(1, 0, 0, 0))
            rouge1 = max([rouge.get_scores(' '.join(gen_distractor), ' '.join(gt))[0]['rouge-1']['r'] for gt in clean_ground_truth_distractors])
            bleu1_scores.append(bleu1)
            rouge1_scores.append(rouge1)

        scoresA['bleu-1'].append(max(bleu1_scores))
        scoresA['rouge-1'].append(max(rouge1_scores))

    best_bleu1_scores = np.asarray(scoresA['bleu-1'])
    best_rouge1_scores = np.asarray(scoresA['rouge-1'])

    print("Mean bleu-1 score:", np.mean(best_bleu1_scores))
    print("Mean rouge-1 score:", np.mean(best_rouge1_scores))

    # Plot boxplots
    data = []
    for b1, r1 in zip(best_bleu1_scores, best_rouge1_scores):
        data.append({'Metric': 'BLEU-1', 'Score': b1})
        data.append({'Metric': 'ROUGE-1', 'Score': r1})
    df = pd.DataFrame(data)
    sns.violinplot(data=df, x='Score', y='Metric')
    plt.savefig(args.save_path + 'ngram.png')




if __name__ == '__main__':
    args = parser.parse_args()
    main(args)