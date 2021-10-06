This is the repository for our EMNLP 2021 paper "[BERT, mBERT, or BiBERT? A Study on Contextualized Embeddings for Neural Machine Translation](https://arxiv.org/abs/2109.04588)".
```
@misc{xu2021bert,
      title={BERT, mBERT, or BiBERT? A Study on Contextualized Embeddings for Neural Machine Translation}, 
      author={Haoran Xu and Benjamin Van Durme and Kenton Murray},
      year={2021},
      eprint={2109.04588},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
## Prerequisites
```
conda create -n bibert python=3.7
conda activate bibert
```
* [transformers](https://github.com/huggingface/transformers) >= 4.4.2
  ```
  pip install transformers
  ```
* Install our fairseq repo
  ```
  cd BiBERT
  pip install --editable ./
  ```
* [hydra](https://github.com/facebookresearch/hydra) = 1.0.3
  ```
  pip install hydra-core==1.0.3
  ```
* ```
  pip install sentencepiece
  ```

* ```
  pip install protobuf
  ```

## BiBERT
Download our pre-trained bilingual English-German BiBERT:
```
from transformers import BertTokenizer, AutoModel
tokenizer = BertTokenizer.from_pretrained("jhu-clsp/bibert-ende")
model = AutoModel.from_pretrained("jhu-clsp/bibert-ende")
```
An example of obtaining the contextual embeddings of the input sentence:
```
import torch
text = "Hello world!"

## Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained("jhu-clsp/bibert-ende")
model = AutoModel.from_pretrained("jhu-clsp/bibert-ende")

## Feed input sentence to the model
tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

## Obtain the contextual embeddings of BiBERT
layer = -1 # Last layer
output = model(input_ids, output_hidden_states=True)[2][layer]
```
## Reproduction
### Preprocessing
Download and prepare IWSLT'14 dataset (if you meet warnings like `file config.json not found`, please feel safe to ignore it):
```
cd download_prepare
bash download_and_prepare_data.sh
```

After download and preprocessing, three preprocessed data bin will be shown in `download_prepare` folder:
* `data`: de->en preprocessed data for ordinary one-way translation
* `data_mixed`: dual-directional translation data
* `data_mixed_ft`: after dual-directional training, fine-tuning on one-way translation data

### Training
Train a model for one-way translation. Note that passing field `--use_drop_embedding` to consider number of layers in stochastic layer selection, where the default is 8. Training with less GPUs should increase `--update-freq`, e.g., `update-freq=8` for 2 GPUs and `update-freq=4` for 4 GPUs.
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

## WMT'14 Data Training 
Download our preprocessed WMT'14 dataset [wmt-data.zip](https://drive.google.com/file/d/1wbcnqwamiI5IfZrkNlQZmhvIsuSoCqwn/view?usp=sharing)
```
cd download_prepare
unzip wmt-data.zip
```
The resource of training data comes from [Standford WMT'14 dataset](https://nlp.stanford.edu/projects/nmt/). The data in `wmt-data` has been preprocessed the same way as IWSLT'14.

Similar to IWSLT'14 training and evaluation discribed above, we train and evaluate the model by running `train-wmt.sh/train-wmt-dual.sh` and `generate-wmt.sh/generate-wmt-dual.sh`.
