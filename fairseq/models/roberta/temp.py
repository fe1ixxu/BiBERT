from fairseq.models.roberta import RobertaModel
from fairseq import checkpoint_utils
from transformers import BertTokenizer
import torch
import argparse
from fairseq.data import Dictionary
from fairseq.tasks.masked_lm import MaskedLMTask 

# parser = argparse.ArgumentParser()
# args = parser.parse_args()
# args.source_dictionary = Dictionary.load("/export/c12/haoranxu/fairseq/pretrain/models/lang-model/vocab.txt", "/export/c12/haoranxu/fairseq/pretrain/models/lang-model/vocab.txt")
# args.max_positions = 512
# args.encoder_layers_to_keep = None
# args.encoder_layerdrop = 0.0
# args.seed = 0

# task = MaskedLMTask(args, args.source_dictionary)

# model = RobertaModel.build_model(args, args)
# torch.save(model, "/export/c12/haoranxu/fairseq/pretrain/models/lang-model/model.random2.pt")

model = torch.load("/export/c12/haoranxu/fairseq/pretrain/models/lang-model/lang-model552.pt")
# state = checkpoint_utils.load_checkpoint_to_cpu("/home/felix_hxu1/models/checkpoints/checkpoint350.pt")
# state = torch.load("state.pt")
# model.load_state_dict(state["model"], strict=True)
# model = torch.load("model.pt").to("cpu")
model.to(torch.device("cuda"))
model.eval()

tokens = "[CLS] The Great Wall is located at [MASK] city in China. [SEP]"
t = BertTokenizer.from_pretrained("/export/c12/haoranxu/fairseq/pretrain/models/lang-model/wp_models/")
tokens = torch.tensor(t.convert_tokens_to_ids(t.tokenize(tokens))).long().to(torch.device("cuda"))
tokens = tokens.unsqueeze(0)
masked_index = (tokens == 4).nonzero()

features = model(tokens)[0]
logits = features[masked_index[:,0], masked_index[:,1], :]
probs, topinds = logits.topk(dim=1,k=5)
for prob, topind in zip(probs, topinds):
    print(t.convert_ids_to_tokens(topind))
    print(prob.softmax(dim=0))
