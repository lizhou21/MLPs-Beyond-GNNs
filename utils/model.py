import dgl
import torch
import torch.nn as nn
from llm2vec import LLM2Vec
from transformers import AutoModel, T5EncoderModel
from torch.cuda.amp import autocast

from utils.GNNs import *

# from utils.efficient_kan import KAN

class ProbModel(nn.Module): 
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        if args.model_name == args.base_model_name_or_path:
            if "t5" in args.model_name:
                config.output_hidden_states = True
                self.sequence_encoder = T5EncoderModel.from_pretrained(args.model_name, config=config)
            else:
                self.sequence_encoder = AutoModel.from_pretrained(args.model_name, config=config)
        else:
            self.sequence_encoder = LLM2Vec.from_pretrained(args.base_model_name_or_path, 
                                                            peft_model_name_or_path=args.model_name,
                                                            config=config,
                                                            torch_dtype=torch.bfloat16)
        hidden_size = config.hidden_size

        # define middle component
        if self.args.do_middle:
            if self.args.middle_format == 'GNN':
                self.middle_encoder = GCN_Encoder(hidden_size, self.args.mid_layers, torch.relu, 0.5)
            elif self.args.middle_format == 'Message':
                self.middle_encoder = DAGNN_Encoder(hidden_size, self.args.mid_layers)
            elif self.args.middle_format == 'MLP':
                self.middle_encoder = MLP_Encoder(hidden_size, self.args.mid_layers, torch.relu, 0.5)
            elif self.args.middle_format == 'KAN':
                self.middle_encoder = KANEncoder(hidden_size, self.args.mid_layers, torch.relu, 0.5)

        if self.args.task_type == "SentEval":
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(p=self.args.dropout_prob),
                nn.Linear(hidden_size, self.args.num_labels)
                )
            self.loss_fnt = nn.CrossEntropyLoss()

        elif self.args.data_name in ['semeval', 'dep_ewt', 'dpr', 'srl', 'coref']:
            self.classifier = nn.Sequential(
                nn.Linear(2 * hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(p=self.args.dropout_prob),
                nn.Linear(hidden_size, self.args.num_labels)
                )
            self.loss_fnt = nn.CrossEntropyLoss()
        
        elif self.args.data_name in ['const', 'pos', 'ner']:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(p=self.args.dropout_prob),
                nn.Linear(hidden_size, self.args.num_labels)
                )
            self.loss_fnt = nn.CrossEntropyLoss()
        
        elif self.args.data_name == 'spr2':
            self.classifier = nn.Sequential(
                nn.Linear(2 * hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(p=self.args.dropout_prob),
                nn.Linear(hidden_size, self.args.num_labels)
                )
            self.loss_fnt = nn.BCEWithLogitsLoss()
        

    def get_entity(self, output, subj_pos, obj_pos):
        ss_emb = []
        os_emb = []
        for i in range(len(subj_pos)):
            ss_emb.append(torch.mean(output[i][subj_pos[i][0]:subj_pos[i][1]][:],dim=0))
            os_emb.append(torch.mean(output[i][obj_pos[i][0]:obj_pos[i][1]][:],dim=0))
        return torch.stack(ss_emb, dim=0), torch.stack(os_emb, dim=0)

    def get_graph(self, outputs, sent_len, graphs):
        node_f = []
        for i in range(len(sent_len)):
            node_f.append(outputs[i][0: sent_len[i]-1])
        graph_big =  dgl.batch(graphs).to(outputs.device)
        node_f = torch.cat(node_f,dim=0)
        return node_f, graph_big

    def get_rep(self, output, subj_pos, obj_pos, sent_len):
        global_len = 0
     
        ss_emb = []
        os_emb = []
        for i, token_l in enumerate(sent_len):
            local_subj_start = int(subj_pos[i]) + global_len
            local_obj_start = int(obj_pos[i]) + global_len
            ss_emb.append(output[local_subj_start])
            os_emb.append(output[local_obj_start])
            global_len = global_len + token_l -1 
        
        return torch.stack(ss_emb, dim=0), torch.stack(os_emb, dim=0), 

    def get_sent(self, output, sent_len):
        global_len = 0
     
        sent_emb = []
        for i, token_l in enumerate(sent_len):
            idx = global_len
            sent_emb.append(output[idx])
            global_len = global_len + token_l -1 
        
        return torch.stack(sent_emb, dim=0)
   
    def get_Graph_rep(self, output, subj_pos, obj_pos, sent_len, label_count=None):
        global_len = 0
        ss_emb = []
        os_emb = []
        for i, token_l in enumerate(sent_len):
            seq_rep = output[global_len: global_len+token_l-1]
            for j, pos in enumerate(subj_pos[i]):
                # subj
                if self.args.token_format == 'max':
                    subj_rep = torch.max(seq_rep[subj_pos[i][j][0]: subj_pos[i][j][1]+1], dim=0)[0]
                    ss_emb.append(subj_rep)
                    if len(obj_pos[i][j]) != 0:
                        obj_rep = torch.max(seq_rep[obj_pos[i][j][0]: obj_pos[i][j][1]+1], dim=0)[0]
                        os_emb.append(obj_rep)
                elif self.args.token_format == 'mean':
                    subj_rep = torch.mean(seq_rep[subj_pos[i][j][0]: subj_pos[i][j][1]+1], dim=0).unsqueeze(0)
                    ss_emb.append(subj_rep)
                    if len(obj_pos[i][j]) != 0:
                        obj_rep = torch.mean(seq_rep[obj_pos[i][j][0]: obj_pos[i][j][1]+1], dim=0).unsqueeze(0)
                        os_emb.append(obj_rep)
            global_len = global_len + token_l - 1
        if len(os_emb)!=0:
            ss_emb = torch.stack(ss_emb, dim=0)
            os_emb = torch.stack(os_emb, dim=0)
            return torch.cat((ss_emb, os_emb), dim=-1)
        else:
            ss_emb = torch.stack(ss_emb, dim=0)
            return ss_emb
    
    def get_MLP_rep(self, output, subj_pos, obj_pos, sent_len, label_count=None):
        ss_emb = []
        os_emb = []
        for i, sl in enumerate(sent_len):
            seq_rep = output[i]
            for j, pos in enumerate(subj_pos[i]):
                # subj
                if self.args.token_format == 'max':
                    subj_rep = torch.max(seq_rep[subj_pos[i][j][0]: subj_pos[i][j][1]+1], dim=0)[0]
                    ss_emb.append(subj_rep)
                    if len(obj_pos[i][j]) != 0:
                        obj_rep = torch.max(seq_rep[obj_pos[i][j][0]: obj_pos[i][j][1]+1], dim=0)[0]
                        os_emb.append(obj_rep)
                elif self.args.token_format == 'mean':
                    subj_rep = torch.mean(seq_rep[subj_pos[i][j][0]: subj_pos[i][j][1]+1], dim=0).unsqueeze(0)
                    ss_emb.append(subj_rep)
                    if len(obj_pos[i][j]) != 0:
                        obj_rep = torch.mean(seq_rep[obj_pos[i][j][0]: obj_pos[i][j][1]+1], dim=0).unsqueeze(0)
                        os_emb.append(obj_rep)
        if len(os_emb)!=0:
            ss_emb = torch.stack(ss_emb, dim=0)
            os_emb = torch.stack(os_emb, dim=0)
            return torch.cat((ss_emb, os_emb), dim=-1)
        else:
            ss_emb = torch.stack(ss_emb, dim=0)
            return ss_emb

    def tensor_repeat(self, A, count):
        result_list = []
        for i, value in enumerate(count):
            repeated_A = A[i].unsqueeze(0).repeat(value, 1, 1)
            result_list.append(repeated_A)
        A_result = torch.cat(result_list, dim=0)
        return A_result

    def encode_sequence(self, input_ids):
        with torch.no_grad():
            if ("llama" in self.args.model_name.lower()) or ("mistral" in self.args.model_name.lower()):
                input_ids = input_ids.to(self.sequence_encoder.model.device)
                pooled_output = self.sequence_encoder.model(input_ids).hidden_states[-1]
            else:
                input_ids = input_ids.to(self.sequence_encoder.device)
                # pooled_output = self.sequence_encoder(input_ids).hidden_states[self.args.prob_layers]
                pooled_output = self.sequence_encoder(input_ids).hidden_states[-1]
        torch.cuda.empty_cache()
        return pooled_output

    @autocast()
    def forward(self, input_ids=None, attention_mask=None, pooled_output=None, labels=None, subj_pos=None, obj_pos=None, sent_lens=None, graphs=None, labels_count=None):
        device = input_ids.device
        if self.args.do_middle:
            # get node_feature and graph
            #  DGL graph
            if self.args.middle_format in ['MLP', 'KAN']:
                output_nodes = self.middle_encoder(pooled_output)
                # print('MLP output nodes:{}'.format(output_nodes.shape))
                if self.args.task_type == "SentEval":
                    sent_emb = output_nodes[:,0,:] # representation after mid encoder
                # elif self.args.data_name in ['semeval', 'dep_ewt', 'dpr', 'srl']:
                elif self.args.task_type == "EdgeProb":
                    sent_emb =  self.get_MLP_rep(output_nodes, subj_pos, obj_pos, sent_lens, labels_count)
                    # print('MLP sent emb:{}'.format(sent_emb.shape))


            else:
                node_feature, batch_graph = self.get_graph(pooled_output, sent_lens, graphs)
                batch_graph = batch_graph.add_self_loop()
                output_nodes = self.middle_encoder(batch_graph, node_feature)

                if self.args.task_type == "SentEval":
                    sent_emb = self.get_sent(output_nodes, sent_lens) # representation after mid encoder
                # elif self.args.data_name in ['semeval', 'dep_ewt', 'dpr', 'srl']:
                elif self.args.task_type == "EdgeProb":
                    sent_emb =  self.get_Graph_rep(output_nodes, subj_pos, obj_pos, sent_lens, labels_count)
                    # print('GNN sent emb:{}'.format(sent_emb.shape))
        else:
            if self.args.task_type == "SentEval":
                sent_emb = pooled_output[:,0,:] # without mid encoder
            # elif self.args.data_name in ['semeval', 'dep_ewt', 'dpr', 'srl']:
            elif self.args.task_type == "EdgeProb":
                sent_emb =  self.get_MLP_rep(pooled_output, subj_pos, obj_pos, sent_lens, labels_count)
                # print('WO sent emb:{}'.format(sent_emb.shape))

        logits = self.classifier(sent_emb.to(device))
        outputs = (logits, )
        if labels is not None:
            if self.args.data_name == 'spr2':
                labels = labels.float()
            loss = self.loss_fnt(logits.float().squeeze(), labels.squeeze())
            outputs = (loss,) + outputs
        return outputs

    