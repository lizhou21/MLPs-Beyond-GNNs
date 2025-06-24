import os, copy
import math, torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from utils.util import read_json, write_pickle, read_json_random


class Processor:
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer

    def get_align_edges(self, token_to_bert, head):
        dep_edges = []
        sub_edges = []
        for h, idx in head:
            # 正常的是当前词idx->当前词的head
            if h != 0:
                start_word = token_to_bert[h - 1]
            else:
                start_word = [h]
            end_word = token_to_bert[idx]
            dep_edges.append((start_word[0], end_word[0]))
            if len(token_to_bert[idx]) != 1:
                start_word = token_to_bert[idx][0]
                for j in range(1, len(token_to_bert[idx])):
                    sub_edges.append((start_word, token_to_bert[idx][j]))
        return dep_edges, sub_edges

    def tokenize(self, tokens, edges):
        # prefix
        if "Llama" in self.tokenizer.name_or_path:
            sents = ["<|start_header_id|>", "user", "<|end_header_id|>", "\n\n"]
        elif "Mistral" in self.tokenizer.name_or_path:
            sents = ['▁[', 'INST', ']']
        else:
            sents = ["[CLS]"]

        # sentence content and align
        L_bert = len(sents)
        token_to_bert = dict()
        for i_t, token in enumerate(tokens):
            tokens_wordpiece = self.tokenizer.tokenize(token)
            token_to_bert[i_t] = list(range(L_bert, L_bert + len(tokens_wordpiece)))
            sents.extend(tokens_wordpiece)
            L_bert = len(sents)

        if "Mistral" in self.tokenizer.name_or_path:
            sents = sents + ['▁[', '/', 'INST', ']']
        elif "Llama" in self.tokenizer.name_or_path:
            sents = sents + ["<|eot_id|>"]
        else:
            sents = sents + ["[SEP]"]

        sents = sents[: self.args.max_seq_length]
        input_ids = self.tokenizer.convert_tokens_to_ids(sents)

        # llama tokenize \n\n as 271, however, conver_to_ids function tokenize it as None
        if "Llama" in self.tokenizer.name_or_path:
            input_ids[3] = 271

        dep_edges, sub_edges = self.get_align_edges(token_to_bert, edges)
        new_edges = dep_edges + sub_edges
        sent_len = len(sents)
        return (
            input_ids,
            new_edges,
            sent_len,
            token_to_bert,
            sents,
            self.tokenizer.pad_token_id,
        )


class SentProcessor(Processor):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        self.LABEL_TO_ID = args.LABEL_TO_ID

    def read(self, file_in, type, model):
        print(f"processing: {self.args.task_type} {self.args.data_name} {type}")

        features = []
        cache_file = f"/data2/public/{self.args.input_dir}/{self.args.task_type}/{self.args.data_name}/{self.args.model_name.split('/')[-1]}/{type}"
        if not os.path.exists(cache_file):
            os.makedirs(cache_file, exist_ok=True)
            data = read_json(file_in)
            for i, d in enumerate(tqdm(data)):
                input_ids, edges, sent_len, token_to_bert, sents, pad_token_id = (
                    self.tokenize(d["tokens"], d["edges"])
                )
                feature = {
                    "input_ids": input_ids,
                    "labels": self.LABEL_TO_ID[d["label"]],
                    "sent_len": sent_len,
                    "edges": edges,
                    "pad_token_id": pad_token_id,
                }
                features.append(feature)
            # encode batch inputs to save computational overhead
            write_and_encode_corpus(cache_file, features, model, self.tokenizer.pad_token_id)


class EdgeProcessor(Processor):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        self.LABEL_TO_ID = args.LABEL_TO_ID

    def read(self, file_in, type, model):
        features = []
        count = 0
        c = 0

        cache_file = f"/data2/public/{self.args.input_dir}/{self.args.task_type}/{self.args.data_name}/{self.args.model_name.split('/')[-1]}/{type}"
        if not os.path.exists(cache_file):
            os.makedirs(cache_file, exist_ok=True)
        # if self.args.random_graph:
        #     data = read_json_random(file_in)
        # else:
        #     data = read_json(file_in)
        data = read_json_random(file_in)
        for i, d in enumerate(tqdm(data)):
            if self.args.data_name == "semeval":
                d["tokens"] = d["token"]

            if len(d["edges"]) == 0:
                c = c + 1
                continue

            if max([t for edge in d["edges"] for t in edge]) >= len(d["tokens"]):
                count = count + 1
                continue

            input_ids, edges, sent_len, token_to_bert, sents, pad_token_id = (
                self.tokenize(d["tokens"], d["edges"])
            )

            rel = []
            for target in d["targets"]:
                t = {}
                # if self.args.data_name == 'semeval':
                if self.args.data_name == "spr2":
                    t["label"] = [0] * len(self.LABEL_TO_ID)
                    for l in target["label"]:
                        t["label"][self.LABEL_TO_ID[l]] = 1
                else:
                    t["label"] = self.LABEL_TO_ID[target["label"]]
                t["s"] = [token_to_bert[target["span1"][0]][0], token_to_bert[target["span1"][1] - 1][-1]]

                if "span2" in target:
                    t["e"] = [token_to_bert[target["span2"][0]][0], token_to_bert[target["span2"][1] - 1][-1]]
                else:
                    t["e"] = []
                rel.append(t)

            feature = {
                "input_ids": input_ids,
                "labels": rel,
                "sent_len": sent_len,
                "edges": edges,
                "pad_token_id": pad_token_id,
            }
            features.append(feature)

        # encode batch inputs to save computational overhead
        write_and_encode_corpus(cache_file, features, model, self.tokenizer.pad_token_id)


def write_and_encode_corpus(cache_file, corpus, model, pad_token_id):
    count = 0
    batch_size = 64
    total_steps = math.ceil(len(corpus) / batch_size)

    for i in tqdm(range(total_steps), total=total_steps):
        start = i * batch_size
        end = min(start + batch_size, len(corpus))
        batch = pad_sequence(
            [torch.LongTensor(f["input_ids"]) for f in corpus[start:end]],
            batch_first=True,
            padding_value=pad_token_id,
        )

        pooled_outputs = model.encode_sequence(batch)
        for i, (pooled_output, input_id) in enumerate(zip(pooled_outputs, batch)):
            item = copy.deepcopy(corpus[i + start])
            item["pooled_output"] = pooled_output[: len(input_id), :].tolist()
            write_pickle(f"{cache_file}/{count}.pkl", item)
            count += 1
    return corpus
