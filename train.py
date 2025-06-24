from utils.args import *
import torch
import torch
import random
import numpy as np
import os
from transformers import AutoConfig, AutoTokenizer
from utils.prepro import *
from utils.model import *
from torch.utils.data import DataLoader
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler
from sklearn.metrics import classification_report,  accuracy_score, precision_score, recall_score, f1_score

def evaluate(args, model, features, tag='dev'):
    if args.task_type == "SentEval":
        dataloader = DataLoader(features, batch_size=args.test_batch_size, collate_fn=collate_sent, drop_last=False)
    # elif args.data_name in ['semeval', 'dep_ewt', 'dpr', 'srl']:
    elif args.task_type == "EdgeProb":
        dataloader = DataLoader(features, batch_size=args.test_batch_size, collate_fn=collate_two_token, drop_last=False)
    
    keys, preds, logits = [], [], []
    for i_b, batch in enumerate(tqdm(dataloader)):
        model.eval()

        if args.task_type == "SentEval":
            inputs = {'input_ids': batch[0].to(args.device),
                    'attention_mask': batch[1].to(args.device),
                    'sent_lens': batch[3],
                    'graphs': [b.to(args.device) for b in batch[4]],
                    }
        # elif args.data_name in ['semeval', 'dep_ewt', 'dpr', 'srl']:
        elif args.task_type == "EdgeProb":
            inputs = {'input_ids': batch[0].to(args.device),
                        'attention_mask': batch[1].to(args.device),
                        'sent_lens': batch[3],
                        'graphs': [b.to(args.device) for b in batch[4]],
                        'subj_pos': batch[5],
                        'obj_pos': batch[6],
                        'labels_count': batch[7],
                        }


        if args.data_name == 'spr2':
            keys += batch[2].tolist()
            with torch.no_grad():
                logit = model(**inputs)[0]
                pred = (torch.sigmoid(logit) > 0.5).float()
            preds += pred.tolist()
            logits.append(logit)
        else:
            keys += batch[2].tolist()
            with torch.no_grad():
                logit = model(**inputs)[0]
                pred = torch.argmax(logit, dim=-1)
            preds += pred.tolist()
            logits.append(logit)
        
    if args.task_type == 'SentEval' or args.data_name in ['semeval', 'dep_ewt', 'dpr', 'srl', 'const', 'coref', 'pos', 'ner']:
        # 确定存在的标签
        existing_labels = set(np.unique(keys)).union(set(np.unique(preds)))
        labels_to_use = [label for label in args.LABEL if args.LABEL_TO_ID[label] in existing_labels]
        report = classification_report(keys, preds, target_names=labels_to_use, output_dict=True)
    elif args.data_name == 'spr2':
        report={}
        report['accuracy'] = accuracy_score(keys, preds)
        report['macro avg'] = {
            'precision': precision_score(keys, preds, average='macro'),
            'recall': recall_score(keys, preds, average='macro'),
            'f1-score': f1_score(keys, preds, average='macro'),
        }
        report['weighted avg'] = {
            'precision': precision_score(keys, preds, average='weighted'),
            'recall': recall_score(keys, preds, average='weighted'),
            'f1-score': f1_score(keys, preds, average='weighted'),
        }
    
    if tag == 'dev':
        with open(args.log_file['dev_file'], 'a') as file:
            json.dump(report, file)
            file.write('\n')
    elif tag == 'test':
        with open(args.log_file['test_file'], 'a') as file:
            json.dump(report, file)
            file.write('\n')


def train(args, model, train_features, benchmarks):
    if args.task_type == "SentEval":
        train_dataloader = DataLoader(train_features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_sent, drop_last=True)
    # elif args.data_name in ['semeval', 'dep_ewt', 'dpr', 'srl', 'const']:
    elif args.task_type == "EdgeProb":
        train_dataloader = DataLoader(train_features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_two_token, drop_last=True)



    total_steps = int(len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    for param in model.sequence_encoder.parameters():
        param.requires_grad = False

    scaler = GradScaler()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    print('Total steps: {}'.format(total_steps))
    print('Warmup steps: {}'.format(warmup_steps))
    num_steps = 0
    for epoch in range(int(args.num_train_epochs)):
        model.zero_grad()
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            if args.task_type == "SentEval":
                inputs = {'input_ids': batch[0].to(args.device),
                        'attention_mask': batch[1].to(args.device),
                        'labels': batch[2].to(args.device),
                        'sent_lens': batch[3],
                        'graphs': [b.to(args.device) for b in batch[4]],
                        }
            # elif args.data_name in ['semeval', 'dep_ewt', 'dpr', 'srl', 'const']:
            elif args.task_type == "EdgeProb":
                inputs = {'input_ids': batch[0].to(args.device),
                        'attention_mask': batch[1].to(args.device),
                        'labels': batch[2].to(args.device),
                        'sent_lens': batch[3],
                        'graphs': [b.to(args.device) for b in batch[4]],
                        'subj_pos': batch[5],
                        'obj_pos': batch[6],
                        'labels_count': batch[7],
                        }
            outputs = model(**inputs)
            loss = outputs[0] / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if step % args.gradient_accumulation_steps == 0:
                num_steps += 1
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                model.zero_grad()

                write_line = '[{}/{}, {}/{}] loss: {:.8}'.format(epoch, args.num_train_epochs, step, len(train_dataloader), loss.item())
                with open(args.log_file['loss_file'], 'a') as file:
                    file.write(write_line + '\n')
            

            # show progress (loss)
            if (num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                
                # print('[{}/{}, {}/{}] loss: {:.8}'.format(epoch, args.num_train_epochs, step, len(train_dataloader), loss.item()))
                
                for tag, features in benchmarks:
                    evaluate(args, model, features, tag=tag)


def main():
    args = get_parse()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    # if args.seed > 0:
    #     set_seed(args)

    if args.do_middle:
        log_dir = args.log_dir + '/' + args.model_name.split('/')[-1] + '/' + args.task_type + '/' + args.data_name + '/' + args.middle_format + '/' + str(args.prob_layers) + '/' + str(args.runs)
    else:
        log_dir = args.log_dir + '/' + args.model_name.split('/')[-1] + '/' + args.task_type + '/' + args.data_name + '/' + str(args.do_middle) + '/' + str(args.prob_layers) + '/' + str(args.runs)

    os.makedirs(log_dir, exist_ok=True)
    args.log_file = {
        'loss_file': os.path.join(log_dir, 'loss_log.txt'),
        'dev_file': os.path.join(log_dir, 'dev_log.txt'),
        'test_file': os.path.join(log_dir, 'test_log.txt'),

    }
    for file in args.log_file.values():
        with open(file, 'w'):
            pass

    # 1. obtain dataset
    Data_dir = args.input_dir + "/" + args.task_type + "/" + args.data_name
    train_file = os.path.join(Data_dir, "train.json")
    dev_file = os.path.join(Data_dir, "dev.json")
    test_file = os.path.join(Data_dir, "test.json")
    label_file = os.path.join(Data_dir, "labels.txt")


    # 2. datasets processing

    with open(label_file, 'r') as file:
        lines = file.readlines()
    LABEL = [line.strip() for line in lines]
    LABEL_TO_ID = {label: index for index, label in enumerate(LABEL)}
    args.LABEL = LABEL
    args.LABEL_TO_ID = LABEL_TO_ID
    args.num_labels = len(LABEL)

    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=len(LABEL),
    )

    config.gradient_checkpointing = True
    config.output_hidden_states = True

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
    )
    if args.task_type == 'SentEval':
        processor = SentProcessor(args, tokenizer)
    elif args.task_type == 'EdgeProb':
        processor = EdgeProcessor(args, tokenizer)
    
    train_features = processor.read(train_file, 'train')
    dev_features = processor.read(dev_file, 'dev')
    test_features = processor.read(test_file, 'test')


    # 3. define model
    model = ProbModel(args, config)
    
    if args.device.type != 'cpu':
        model.to(args.device)

    benchmarks = [
            ("dev", dev_features),
            ("test", test_features),
        ]
    

    # 4. train model
    train(args, model, train_features, benchmarks)



    
    
    
    
    # print('a')


if __name__ == "__main__":
    main()