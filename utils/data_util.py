import dgl
import glob
import random
import torch
from utils.util import read_pickle
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

    def __getitem__(self, index):
        item = read_pickle(f"{self.data_path}/{index}.pkl")
        return item

    def __len__(self):
        return len(glob.glob(f"{self.data_path}/*.pkl"))


def collate_sent(batch):
    pad_token_id = batch[0]["pad_token_id"]

    input_ids = pad_sequence(
        [torch.LongTensor(f["input_ids"]) for f in batch],
        batch_first=True,
        padding_value=pad_token_id,
    )
    attention_mask = pad_sequence(
        [torch.FloatTensor([1.0] * len(f["input_ids"])) for f in batch],
        batch_first=True,
        padding_value=0,
    )
    pooled_output = pad_sequence(
        [torch.FloatTensor(f["pooled_output"]) for f in batch],
        batch_first=True,
        padding_value=0.0,
    )
    labels = torch.LongTensor([f["labels"] for f in batch])
    sent_lens = [f["sent_len"] for f in batch]
    graph_list = []
    for example in batch:
        edge_ss = [ee[0] for ee in example["edges"]]
        edge_ee = [ee[1] for ee in example["edges"]]
        graph = dgl.graph((edge_ss, edge_ee), num_nodes=example["sent_len"] - 1)
        graph_list.append(graph)
        assert example["sent_len"] - 1 == graph.nodes().size()[0]
    # output = (input_ids, input_mask, labels, sent_lens, graph_list)
    ouput = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pooled_output": pooled_output,
        "labels": labels,
        "sent_lens": sent_lens,
        "graphs": graph_list,
    }
    return ouput


def collate_two_token(batch):
    pad_token_id = batch[0]["pad_token_id"]

    input_ids = pad_sequence(
        [torch.LongTensor(f["input_ids"]) for f in batch],
        batch_first=True,
        padding_value=pad_token_id,
    )
    attention_mask = pad_sequence(
        [torch.FloatTensor([1.0] * len(f["input_ids"])) for f in batch],
        batch_first=True,
        padding_value=0,
    )
    pooled_output = pad_sequence(
        [torch.FloatTensor(f["pooled_output"]) for f in batch],
        batch_first=True,
        padding_value=0.0,
    )
    labels = torch.LongTensor([l["label"] for f in batch for l in f["labels"]])
    subj_pos = [[l["s"] for l in f["labels"]] for f in batch]
    obj_pos = [[l["e"] for l in f["labels"]] for f in batch]
    labels_count = [len(f["labels"]) for f in batch]
    sent_lens = [f["sent_len"] for f in batch]

    graph_list = []
    for example in batch:
        edge_ss = [ee[0] for ee in example["edges"]]
        edge_ee = [ee[1] for ee in example["edges"]]
        graph = dgl.graph((edge_ss, edge_ee), num_nodes=example["sent_len"] - 1)
        graph_list.append(graph)
        assert example["sent_len"] - 1 == graph.nodes().size()[0]
    # output = input_ids, input_mask, labels, sent_lens, graph_list, subj_pos, obj_pos, labels_count
    output = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pooled_output": pooled_output,
        "labels": labels,
        "sent_lens": sent_lens,
        "graphs": graph_list,
        "subj_pos": subj_pos,
        "obj_pos": obj_pos,
        "labels_count": labels_count,
    }
    return output


def collate_semeval(batch):
    pad_token_id = batch[0]["pad_token_id"]
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [
        f["input_ids"] + [pad_token_id] * (max_len - len(f["input_ids"])) for f in batch
    ]
    input_mask = [
        [1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"]))
        for f in batch
    ]
    labels = [
        f["labels"][0]["label"] for f in batch
    ]  # semeval中每个sentence只有一个label， 所以f["labels"][0]

    subj_pos = [f["labels"][0]["s"] for f in batch]
    obj_pos = [f["labels"][0]["e"] for f in batch]
    sent_lens = [f["sent_len"] for f in batch]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    subj_pos = torch.tensor(subj_pos, dtype=torch.long)
    obj_pos = torch.tensor(obj_pos, dtype=torch.long)

    graph_list = []
    if True:
        for i, example in enumerate(batch):
            edge_ss = [ee[0] for ee in example["edges"]]
            edge_ee = [ee[1] for ee in example["edges"]]
            graph = dgl.graph((edge_ss, edge_ee), num_nodes=example["sent_len"] - 1)
            graph_list.append(graph)
            assert example["sent_len"] - 1 == graph.nodes().size()[0]

    output = (input_ids, input_mask, labels, sent_lens, graph_list, subj_pos, obj_pos)
    return output


def collate_dep(batch):
    pad_token_id = batch[0]["pad_token_id"]
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [
        f["input_ids"] + [pad_token_id] * (max_len - len(f["input_ids"])) for f in batch
    ]
    input_mask = [
        [1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"]))
        for f in batch
    ]
    labels = [
        l["label"] for f in batch for l in f["labels"]
    ]  # 每个sentence有多个label，

    subj_pos = [l["s"] for f in batch for l in f["labels"]]
    obj_pos = [l["e"] for f in batch for l in f["labels"]]
    labels_count = [len(f["labels"]) for f in batch]
    sent_lens = [f["sent_len"] for f in batch]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    subj_pos = torch.tensor(subj_pos, dtype=torch.long)
    obj_pos = torch.tensor(obj_pos, dtype=torch.long)
    # labels_count = torch.tensor(labels_count, dtype=torch.long)

    graph_list = []
    if True:
        for i, example in enumerate(batch):
            edge_ss = [ee[0] for ee in example["edges"]]
            edge_ee = [ee[1] for ee in example["edges"]]
            graph = dgl.graph((edge_ss, edge_ee), num_nodes=example["sent_len"] - 1)
            graph_list.append(graph)
            assert example["sent_len"] - 1 == graph.nodes().size()[0]

    output = (
        input_ids,
        input_mask,
        labels,
        sent_lens,
        graph_list,
        subj_pos,
        obj_pos,
        labels_count,
    )
    return output
