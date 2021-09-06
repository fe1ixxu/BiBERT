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
Download and prepare IWSLT'14 dataset:
```
cd download_prepare
bash download_and_prepare_data.sh
```

Train the model:
```
bash train.sh
```

Translation:
```
bash generate.sh
```

The BLEU score will be printed out in the final output after running `generate.sh`.

