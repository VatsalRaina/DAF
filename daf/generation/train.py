""""
Input: Context + Question + Answer 
Output: Distractor
"""

import argparse
import os
import sys
import json

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import random
import time
import datetime
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_linear_schedule_with_warmup

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=8, help='Specify the training batch size')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Specify the initial learning rate')
parser.add_argument('--adam_epsilon', type=float, default=1e-6, help='Specify the AdamW loss epsilon')
parser.add_argument('--lr_decay', type=float, default=0.85, help='Specify the learning rate decay rate')
parser.add_argument('--dropout', type=float, default=0.1, help='Specify the dropout rate')
parser.add_argument('--n_epochs', type=int, default=2, help='Specify the number of epochs to train for')
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
parser.add_argument('--train_data_path', type=str, help='Load path of training data')
parser.add_argument('--save_path', type=str, help='Load path to which trained model will be saved')

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

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

    seed_val = args.seed
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # Choose device
    device = get_default_device()
    
    # Train only using the RACE data
    with open(args.train_data_path + "middle.json") as f:
        middle_data = json.load(f)
    with open(args.train_data_path + "high.json") as f:
        high_data = json.load(f)
    with open(args.train_data_path + "college.json") as f:
        college_data = json.load(f)
    train_data = middle_data + high_data + college_data

    # Truncate from the front
    tokenizer = T5Tokenizer.from_pretrained("t5-base", truncation_side='left')

    def asNum(x):
        if x=="A":
            return 0
        if x=="B":
            return 1
        if x=="C":
            return 2
        if x=="D":
            return 3

    input_ids = []
    output_ids = []
    input_att_msks = []
    count = 0

    for item in train_data:
        print(count, len(train_data))
        context = item["article"]
        questions = item["questions"]
        answers = item["answers"]
        options = item["options"]
        for qu_num in range(len(questions)):
            lab = asNum(answers[qu_num])
            question = questions[qu_num]
            opts = options[qu_num]
            corr_opt = opts[lab]
            combo = context + " [SEP] " + question + " [SEP] " + corr_opt
            input_encodings_dict = tokenizer(combo, truncation=True, max_length=512, padding="max_length")
            for opt_num, opt in enumerate(opts):
                if opt_num == lab:
                    continue
                else:
                    input_ids.append(input_encodings_dict['input_ids'])
                    input_att_msks.append(input_encodings_dict['attention_mask'])
                    output_encodings_dict = tokenizer(opt, truncation=True, max_length=512, padding="max_length")
                    output_ids.append([x if x!=0 else -100 for x in output_encodings_dict['input_ids']])

    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.long().to(device)
    input_att_msks = torch.tensor(input_att_msks)
    input_att_msks = input_att_msks.long().to(device)
    output_ids = torch.tensor(output_ids)
    output_ids = output_ids.long().to(device)
    
    train_data = TensorDataset(input_ids, input_att_msks, output_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model.to(device)

    optimizer = AdamW(model.parameters(),
                    lr = args.learning_rate,
                    eps = args.adam_epsilon
                    # weight_decay = 0.01
                    )

    total_steps = len(train_dataloader) * args.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0.1*total_steps,
                                                num_training_steps = total_steps)

    for epoch in range(args.n_epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, args.n_epochs))
        print('Training...')
        t0 = time.time()
        total_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            b_input_ids = batch[0].to(device)
            b_att_msks = batch[1].to(device)
            b_output_ids = batch[2].to(device)
            model.zero_grad()
            outputs = model(input_ids=b_input_ids, attention_mask=b_att_msks, labels=b_output_ids)
            loss = outputs[0]
            print(loss.item())
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_loss / len(train_dataloader)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

    file_path = args.save_path+'t5_gen_seed'+str(args.seed)+'.pt'
    torch.save(model, file_path)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)