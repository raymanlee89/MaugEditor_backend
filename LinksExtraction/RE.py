import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertConfig, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput

def RE_Prepro(context, entities, tokenizer, maxlen):
    examples = []
    ent_bound_map = {'SYMBOL': ['<S>', '</S>'], 'PRIMARY': ['<P>', '</P>']}
    example_for_each_label = {}
    for eid in entities:
        entity = entities[eid]
        example_for_each_label[entity['eid']]= {'orig_answer_text' : entity['text'],
                                                'start' : entity['start'],
                                                'end': entity['end'],
                                                'label': entity['label'],
                                                }
    for i, e1 in enumerate(example_for_each_label):
        for j, e2 in enumerate(example_for_each_label):
            if i < j:
                try:
                    span_1 = [example_for_each_label[e1]['start'], example_for_each_label[e1]['end']]
                    span_2 = [example_for_each_label[e2]['start'], example_for_each_label[e2]['end']]
                except:
                    continue
                
                # exclude nested span 1, 2 and 'ORDERED' labeled data.
                span_1_element = [i for i in range(span_1[0], span_1[1])]
                span_2_element = [i for i in range(span_2[0], span_2[1])]
                flag = True
                
                span_1_label = example_for_each_label[e1]['label']
                span_2_label = example_for_each_label[e2]['label']

                for element in span_1_element: 
                    if element in span_2_element:
                        flag = False
                                            
                if flag == False or span_1_label == 'ORDERED' or span_2_label == 'ORDERED':
                    continue
                
                if span_1[0] >= span_2[0]:   
                    span_1, span_2 = span_2, span_1
                    span_1_label, span_2_label = span_2_label, span_1_label
                
                span_1_left_tokens = tokenizer.tokenize(context[:span_1[0]])
                span_1_tokens = tokenizer.tokenize(context[span_1[0]:span_1[1]])
                mid_tokens = tokenizer.tokenize(context[span_1[1]:span_2[0]])
                span_2_tokens = tokenizer.tokenize(context[span_2[0]:span_2[1]])
                span_2_right_tokens = tokenizer.tokenize(context[span_2[1]:])
                
                all_tokens = span_1_left_tokens + \
                            [ent_bound_map[span_1_label][0]] + span_1_tokens + [ent_bound_map[span_1_label][1]] \
                            + mid_tokens + \
                            [ent_bound_map[span_2_label][0]] + span_2_tokens + [ent_bound_map[span_2_label][1]] \
                            + span_2_right_tokens
                span_1 = [len(span_1_left_tokens) + 1, len(span_1_left_tokens) + 1 + len(span_1_tokens)]
                span_2 = [span_1[1] + 2 + len(mid_tokens), span_1[1] + 2 + len(mid_tokens + span_2_tokens)]
                
                if len(all_tokens) <= maxlen - 2:
                    examples.append({
                                    'e1_id': e1,
                                    'e2_id': e2,
                                    'tokens' : all_tokens,
                                    'span_1': span_1,
                                    'span_2': span_2,
                                    'e_1_label': span_1_label,
                                    'e_2_label': span_2_label
                                    })
                else:
                    start = span_1[0]
                    end = span_2[1]
                    
                    if end-start > maxlen-2:
                        continue
                    
                    while len(all_tokens[start:end]) <= maxlen - 4:
                        if start > 0:
                            start -= 1
                        if end < len(all_tokens):
                            end += 1
                            
                    all_tokens = all_tokens[start:end]
                    span_1 = [span_1[0]-start, span_1[1]-start]
                    span_2 = [span_2[0]-start, span_2[1]-start]                 
                    examples.append({
                                    'e1_id': e1,
                                    'e2_id': e2,
                                    'tokens' : all_tokens,
                                    'span_1': span_1,
                                    'span_2': span_2,
                                    'e_1_label': span_1_label,
                                    'e_2_label': span_2_label
                                    })
    return examples

def RE_Data(examples, tokenizer, seq_len):
    cls = tokenizer.convert_tokens_to_ids('[CLS]')
    sep = tokenizer.convert_tokens_to_ids('[SEP]')
    pad = tokenizer.pad_token_id
    
    dataset = []
    for example in examples:
        ids, span_1, span_2 = tokenizer.convert_tokens_to_ids(example['tokens']),\
                example['span_1'], example['span_2']
    
        assert len(ids) <= seq_len - 2
        
        ids = [cls] + ids + [sep]
    
        pad_length = seq_len - len(ids)
    
        attention_mask = (len(ids) * [1]) + (pad_length * [0])
    
        ids = ids + (pad_length * [pad])
        
        span_1, span_2 = [span_1[0] + 1, span_1[1] + 1], [span_2[0] + 1, span_2[1] + 1]
        
        dataset.append({
            "input_ids": ids,
            'attention_mask': attention_mask,
            "span_1": span_1,
            "span_2": span_2,
            'e1_id': int(example['e1_id'][1:]),
            'e2_id': int(example['e2_id'][1:])
        })
    return [{key: torch.tensor(value) for key, value in data.items()} for data in dataset]

class RE_classifier(nn.Module):
    def __init__(self, resize_token_embd_len):
        super().__init__()
        model_config = BertConfig.from_pretrained('allenai/scibert_scivocab_uncased', num_labels=5)
        self.model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased', config=model_config)
        self.out_proj = nn.Linear(768 * 3, 5)
        
        self.model.resize_token_embeddings(resize_token_embd_len)
        self.dropout = nn.Dropout(0.1)
        self.criterion = nn.CrossEntropyLoss()

    def get_span_representation(self, output, span):
        repre = torch.stack([
            torch.sum(output[i, l[0]:l[1], :], dim=0) / (l[1]-l[0]) for i, l in enumerate(span)
        ])
        return repre
    def forward(self, input_ids, attention_mask,  span_1, span_2, labels=None, inputs_embeds=None):

        outputs = self.model(input_ids, attention_mask, inputs_embeds=inputs_embeds)
        output = outputs.last_hidden_state

        span_1 = self.get_span_representation(output, span_1) #  bsz, hidden
        span_2 = self.get_span_representation(output, span_2) #  bsz, hidden
        cls = output[:,0,:].squeeze(1)

        x = self.dropout(torch.cat([span_1,span_2,cls],dim=1))

        if labels is None:
            return self.out_proj(x)

        logits = self.out_proj(x)

        loss = self.criterion(logits, labels)

        return SequenceClassifierOutput(logits= logits, loss = loss)
    
def check_valid_rel(label, arg0, arg1, entities):
    arg0_label = entities[arg0]['label']
    arg1_label = entities[arg1]['label']

    if label == 'Count' or label == 'Direct':
        if arg0_label == arg1_label:
            return False
        else:
            if arg0_label == 'SYMBOL':
                return 2
            else:
                return True
    elif label == 'Corefer-Symbol':
        if arg0_label == arg1_label and arg0_label == 'SYMBOL':
            return True
        else:
            return False
    elif label == 'Corefer-Description':
        if arg0_label == arg1_label and arg0_label == 'PRIMARY':
            return True
        else:
            return False
    elif label == 'Negative_Sample':
        return False
    
def RE(context, entities, tokenizer, model, device, max_length=512):
    label_map = ['Count', 'Direct', 'Corefer-Symbol', 'Corefer-Description', 'Negative_Sample']
    
    examples = RE_Prepro(context, entities, tokenizer, max_length)
    data = RE_Data(examples, tokenizer, max_length)
    
    model.to(device)
    model.eval()
    result = []
    with torch.no_grad():
        for batch in data:
            batch_data = {key: value.to(device) for key, value in batch.items()}
            input_ids = torch.reshape(batch_data['input_ids'], (1, max_length))
            attention_mask = torch.reshape(batch_data['attention_mask'], (1, max_length))
            span_1 = torch.reshape(batch_data['span_1'], (1, 2))
            span_2 = torch.reshape(batch_data['span_2'], (1, 2))
            logits = model.forward(input_ids=input_ids, attention_mask=attention_mask, span_1=span_1, span_2=span_2)
            
            predict = F.softmax(logits, dim=1).argmax(dim=1)
            predicted_e1_id = batch_data['e1_id'].item()
            predicted_e2_id = batch_data['e2_id'].item()
            predicted_label = predict.item()
    
            e1 = 'T'+str(predicted_e1_id)
            e2 = 'T'+str(predicted_e2_id)
            label = label_map[predicted_label]
            
            check_valid = check_valid_rel(label, e1, e2, entities)
            if check_valid == 2: # change
                if e1 != e2:
                    e1, e2 = e2, e1
                result.append({"label": label, "arg0": e1, "arg1": e2})
            elif check_valid == 1: # True
                result.append({"label": label, "arg0": e1, "arg1": e2})
            else:
                if len(result) >= 1:
                    flag = True
                    for rel in result:
                        if (rel['arg0'] == e1 and rel['arg1'] == e2) or (rel['arg1'] == e1 and rel['arg0'] == e2):
                            flag = False
                            break
                    if flag == False:
                        continue
                        
                    check_valid = check_valid_rel(label, e1, e2, entities)
                    if check_valid == 2: # change
                        if e1 != e2:
                            e1, e2 = e2, e1
                        result.append({"label": label, "arg0": e1, "arg1": e2})
                    elif check_valid == 1: # True
                        result.append({"label": label, "arg0": e1, "arg1": e2})                                
                else:
                    check_valid = check_valid_rel(label, e1, e2, entities)
                    if check_valid == 2: # change
                        if e1 != e2:
                            e1, e2 = e2, e1
                        result.append({"label": label, "arg0": e1, "arg1": e2})
                    elif check_valid == 1: # True
                        result.append({"label": label, "arg0": e1, "arg1": e2})
    
    relations = {}
    for rid, relation in enumerate(result):
        relation['rid'] = 'R'+str(rid+1)
        relations['R'+str(rid+1)] = relation
    
    return relations