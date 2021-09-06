## Prerequisites
* [transformers](https://github.com/huggingface/transformers) >= 4.4.2
* Install our fairseq repo
```
cd fairseq
pip install --editable ./
```
## BiBERT
Download our pre-trained bilingual English-German BiBERT:
```
from transformers import BertTokenizer, AutoModel
tokenizer = BertTokenizer.from_pretrained("Coran/bibert-ende")
model = AutoModel.from_pretrained("Coran/bibert-ende")
```
An example of getting the contextual embeddings of input sentence:
```
import torch
SAMPLE_TEXT = "Hello world!"
## Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained("Coran/bibert-ende")
model = AutoModel.from_pretrained("Coran/bibert-ende")
## feed input sentence to the model
tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(SAMPLE_TEXT))
input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
## Obtain the contextual embeddings of BiBERT
layer = -1 # Last layer
output = model(input_ids, output_hidden_states=True)[2][layer]
```
## Preproduce the number in our paper
### Preprocessing
Download and prepare IWSLT'14 dataset (If a warning `file config.json not found` shows up, please ignore it.):
```
cd download_prepare
bash download_and_prepare_data.sh
```

After download and preprocessing, three preprocessed data bin will be shown in `download_prepare` folder:
* `data`: de->en preprocessed data for ordinary one-way translation
* `data_mixed`: dual-directional translation data
* `data_mixed_ft`: fine-tuning on one-way translation data

### Training
Train a model for one-way translation (with stochastic layer selection):
```
bash train.sh
```

Train a model for dual-directional translation and further fine-tuning:
```
bash train-dual.sh
```
### Evaluation
Translation for one-way model:
```
bash generate.sh
```
Translation for dual-directional model:
```
bash generate-dual.sh
```

The BLEU score will be printed out in the final output after running `generate.sh`.

