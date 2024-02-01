import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from transformers import BertConfig, BertModel, BertPreTrainedModel

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def is_symbol(c):
    if c == "\\" or c == "{" or c == "}" or c == "$"or c == "-" \
        or c == "+" or c == "_" or c == "^"or c == "," or c == "=" \
        or c == "[" or c == "]" or c == "(" or c == ")" or c == "<" or c == "*" \
        or c == ">" or c == "|"or c == "/" or c == ":" or c.isdigit() or c.isupper() \
        or c == "." or c == "'" or c == "~" or c == ";"or c == "&":
        return True
    return False

def NER_Prepro(context, tokenizer):
    context_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    prev_is_symbol = False

    for c in context:
        if is_symbol(c):
            context_tokens.append(c)
            char_to_word_offset.append(len(context_tokens) - 1)
            prev_is_symbol = True
            continue
        
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace or prev_is_symbol:
                context_tokens.append(c)
            else:
                context_tokens[-1] += c
            prev_is_whitespace = False
            prev_is_symbol = False
        char_to_word_offset.append(len(context_tokens) - 1)

    assert len(char_to_word_offset) == len(context)
    
    word_to_char_offset = []

    char_span = [0,0]
    temp = 0
    for count, word_offset in enumerate(char_to_word_offset):
        if temp == word_offset:
            continue
        elif temp != word_offset:
            char_span[1] = count - 1
            word_to_char_offset.append(char_span)                    
            temp = word_offset
            char_span = [count,count]
    word_to_char_offset.append([char_span[0], len(char_to_word_offset)])
    
    assert len(word_to_char_offset) == len(context_tokens)

    all_context_tokens = []
    all_tok_to_char_list = []
    

    for (i, token) in enumerate(context_tokens):
        sub_tokens = tokenizer.tokenize(token)
        tok_to_char = tokenizer.encode_plus(token, return_offsets_mapping=True)['offset_mapping'][1:-1]
        if i == 0:
            all_tok_to_char_list+=tok_to_char
        else:
            plus = word_to_char_offset[i][0]
            for i, char_tuple in enumerate(tok_to_char):
                tok_to_char[i] = (char_tuple[0]+plus, char_tuple[1]+plus)         
            
            all_tok_to_char_list+=tok_to_char
            
        for (j, sub_token) in enumerate(sub_tokens):
            all_context_tokens.append(sub_token)
    
    assert len(all_context_tokens) == len(all_tok_to_char_list)
    
    return all_context_tokens, all_tok_to_char_list

def NER_Data(all_context_tokens, all_tok_to_char_list, tokenizer, max_length):
    label_query_id = {'SYMBOL': tokenizer.convert_tokens_to_ids('symbol'),
                      'PRIMARY': tokenizer.convert_tokens_to_ids('description'),
                      'ORDERED': tokenizer.convert_tokens_to_ids('ordered')}
    
    bos_id = tokenizer.convert_tokens_to_ids('[CLS]')
    eos_id = tokenizer.convert_tokens_to_ids('[SEP]')
    pad_id = tokenizer.pad_token_id

    result = []
    
    sliding_length = 100
    maxlen = 500
    
    all_tokens_len = len(all_context_tokens)
    boundary_start = maxlen - sliding_length
    boundary_list = []
    
    if all_tokens_len < maxlen:
        boundary_list.append([0, all_tokens_len])
    else:
        boundary_list.append([0, maxlen])
        while all_tokens_len - (boundary_start + 1) > 0:
            if boundary_start + maxlen < all_tokens_len:
                boundary_list.append([boundary_start, boundary_start + maxlen])
                boundary_start += (maxlen - sliding_length)
            else:
                boundary_list.append([boundary_start, all_tokens_len])
                break
    
    for boundary in boundary_list:
        tokens = all_context_tokens[boundary[0]:boundary[1]]                            
        for query_ids in [[label_query_id['SYMBOL']], [label_query_id['PRIMARY']]]:
            tokens_len = len(tokens)
            query_len = len(query_ids)
            pad_len = max_length - (query_len + tokens_len + 3)
    
            input_ids = [bos_id] + query_ids + [eos_id] + \
                        tokenizer.convert_tokens_to_ids(tokens) \
                        + [eos_id] + ([pad_id] * pad_len)
            token_type_ids = [0] * (query_len + 2) + [1] * (len(input_ids) - (query_len + 2))
            
            assert len(input_ids) == len(token_type_ids)
    
            label_mask = [
                (0 if token_type_ids[token_idx] == 0 else 1)
                for token_idx in range(len(input_ids))
            ]
            start_label_mask = label_mask.copy()
            end_label_mask = label_mask.copy()
            
            result.append([
                torch.LongTensor(input_ids),
                torch.LongTensor(token_type_ids),
                torch.LongTensor(start_label_mask),
                torch.LongTensor(end_label_mask),
                query_ids,
                boundary[0],
                all_tok_to_char_list
            ])
    return result

class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate, act_func="gelu", intermediate_hidden_size=None):
        super(MultiNonLinearClassifier, self).__init__()
        self.num_label = num_label
        self.intermediate_hidden_size = hidden_size if intermediate_hidden_size is None else intermediate_hidden_size
        self.classifier1 = nn.Linear(hidden_size, self.intermediate_hidden_size)
        self.classifier2 = nn.Linear(self.intermediate_hidden_size, self.num_label)
        self.dropout = nn.Dropout(dropout_rate)
        self.act_func = act_func

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        if self.act_func == "gelu":
            features_output1 = F.gelu(features_output1)
        elif self.act_func == "relu":
            features_output1 = F.relu(features_output1)
        elif self.act_func == "tanh":
            features_output1 = F.tanh(features_output1)
        else:
            raise ValueError
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2
    
class BertQueryNerConfig(BertConfig):
    def __init__(self, **kwargs):
        super(BertQueryNerConfig, self).__init__(**kwargs)
        self.mrc_dropout = kwargs.get("mrc_dropout", 0.1)
        self.classifier_intermediate_hidden_size = kwargs.get("classifier_intermediate_hidden_size", 1024)
        self.classifier_act_func = kwargs.get("classifier_act_func", "gelu")

class BertQueryNER(BertPreTrainedModel):
    def __init__(self, config):
        super(BertQueryNER, self).__init__(config)
        self.bert = BertModel(config)

        self.start_outputs = nn.Linear(config.hidden_size, 1)
        self.end_outputs = nn.Linear(config.hidden_size, 1)
        self.span_embedding = MultiNonLinearClassifier(config.hidden_size * 2, 1, config.mrc_dropout, intermediate_hidden_size=config.classifier_intermediate_hidden_size)
        self.hidden_size = config.hidden_size
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        bert_outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        sequence_heatmap = bert_outputs[0]  # [batch, seq_len, hidden]
        batch_size, seq_len, hid_size = sequence_heatmap.size()

        start_logits = self.start_outputs(sequence_heatmap).squeeze(-1)  # [batch, seq_len, 1]
        end_logits = self.end_outputs(sequence_heatmap).squeeze(-1)  # [batch, seq_len, 1]

        start_extend = sequence_heatmap.unsqueeze(2).expand(-1, -1, seq_len, -1)
        end_extend = sequence_heatmap.unsqueeze(1).expand(-1, seq_len, -1, -1)
        span_matrix = torch.cat([start_extend, end_extend], 3)
        span_logits = self.span_embedding(span_matrix).squeeze(-1)

        return start_logits, end_logits, span_logits

class BertLabeling_infer(pl.LightningModule):
    def __init__(
        self,
        classifier_intermediate_hidden_size=1024,
    ):
        super().__init__()
        bert_config = BertQueryNerConfig.from_pretrained("allenai/scibert_scivocab_uncased",
                                                         hidden_dropout_prob=0.1,
                                                         attention_probs_dropout_prob=0.1,
                                                         mrc_dropout=0.1,
                                                         classifier_act_func = "gelu",
                                                         classifier_intermediate_hidden_size=classifier_intermediate_hidden_size)
        self.model = BertQueryNER.from_pretrained("allenai/scibert_scivocab_uncased", config=bert_config)
    
def extract_nested_spans(start_preds, end_preds, match_preds, start_label_mask, end_label_mask, pseudo_tag="TAG"):
    start_label_mask = start_label_mask.bool()
    end_label_mask = end_label_mask.bool()
    bsz, seq_len = start_label_mask.size()
    start_preds = start_preds.bool()
    end_preds = end_preds.bool()

    match_preds = (match_preds & start_preds.unsqueeze(-1).expand(-1, -1, seq_len) & end_preds.unsqueeze(1).expand(-1, seq_len, -1))
    match_label_mask = (start_label_mask.unsqueeze(-1).expand(-1, -1, seq_len) & end_label_mask.unsqueeze(1).expand(-1, seq_len, -1))
    match_label_mask = torch.triu(match_label_mask, 0)  # start should be less or equal to end
    match_preds = match_label_mask & match_preds
    match_pos_pairs = np.transpose(np.nonzero(match_preds.numpy())).tolist()
    return [(pos[1], pos[2], pseudo_tag) for pos in match_pos_pairs]

def NER(context, tokenizer, model, device, max_length=512):
    all_context_tokens, all_tok_to_char_list = NER_Prepro(context, tokenizer)
    data = NER_Data(all_context_tokens, all_tok_to_char_list, tokenizer, max_length)
    entity = {}
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in data:
            tokens, token_type_ids, start_label_mask, end_label_mask, label, start_at_doc, all_tok_to_char_list = batch
            tokens = torch.reshape(tokens, (1, max_length))
            token_type_ids = torch.reshape(token_type_ids, (1, max_length))
            start_label_mask = torch.reshape(start_label_mask, (1, max_length))
            end_label_mask = torch.reshape(end_label_mask, (1, max_length))
            
            attention_mask = (tokens != 0).long()
            start_logits, end_logits, span_logits = model.model(tokens.to(device), attention_mask=attention_mask.to(device), token_type_ids=token_type_ids.to(device))
            start_preds, end_preds, span_preds = start_logits > 0, end_logits > 0, span_logits > 0

            if(not start_preds.is_cpu):
                start_preds = start_preds.cpu()
                end_preds = end_preds.cpu()
                span_preds = span_preds.cpu()
            
            label = label[0]
            if label == 5888: # 'symbol - token_id
                label = 'SYMBOL'
            elif label == 3756: # 'description' token_id
                label = 'PRIMARY'         

            entities_info = extract_nested_spans(start_preds, end_preds, span_preds, start_label_mask, end_label_mask, pseudo_tag="TAG")
            
            if len(entities_info) != 0:
                for entity_info in entities_info:
                    start, end = entity_info[0], entity_info[1]
                    
                    start_char_idx = all_tok_to_char_list[start - 3 + start_at_doc][0]
                    end_char_idx = all_tok_to_char_list[end - 3  + start_at_doc][1]
                    
                    entity_idx = len(entity) + 1
                    
                    flag = False
                    for entity_name in entity:
                        if entity[entity_name]['start']==start_char_idx and entity[entity_name]['end']==end_char_idx and entity[entity_name]['label']==label:
                            flag = True
                            break
                    if flag:
                        continue
                                    
                    entity[f'T{entity_idx}'] = {"eid": f'T{entity_idx}', "label": label, "start": start_char_idx, "end": end_char_idx, "text": context[start_char_idx:end_char_idx]}
    return entity