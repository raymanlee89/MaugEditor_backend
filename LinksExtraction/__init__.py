import torch
from transformers import BertTokenizerFast, AutoTokenizer
from LinksExtraction.NER import BertLabeling_infer, NER
from LinksExtraction.RE import RE_classifier, RE
from LinksExtraction.post_processing import post_processing

device = 'cuda'

NER_checkpoint_path = "LinksExtraction/symlink_checkpoints/ner/epoch=304.ckpt"
RE_checkpoint_path = "LinksExtraction/symlink_checkpoints/re/20400_95.44_332_scibert_uncased"

NER_tokenizer = BertTokenizerFast.from_pretrained("allenai/scibert_scivocab_uncased")
NER_tokenizer.add_special_tokens({'additional_special_tokens': ['[unused10]']})

NER_model = BertLabeling_infer(classifier_intermediate_hidden_size=2048)
NER_model.load_state_dict(torch.load(NER_checkpoint_path, map_location=device)["state_dict"], strict=False)

RE_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
RE_tokenizer.add_special_tokens({'additional_special_tokens': ['[unused10]']})
RE_tokenizer.add_special_tokens({'additional_special_tokens': ['<S>', '</S>']})
RE_tokenizer.add_special_tokens({'additional_special_tokens': ['<P>', '</P>']})

RE_model = RE_classifier(resize_token_embd_len=len(RE_tokenizer))
RE_model.load_state_dict(torch.load(RE_checkpoint_path, map_location=device), strict=False)