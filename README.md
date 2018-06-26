# Showcase: Japanese Predicate-Argument Structure (PAS) analyzer

Showcase is a Pytorch implementation of the Japanese Predicate-Argument Structure (PAS) analyser presented in the paper of Matsubayashi & Inui (2018) with some improvements. 
Given a input sentence, Showcase identifies verbal and nominal predicates in the sentence and detects their nominative (が), accusative (を), and dative (に) case arguments. 
The output case labels are based on the label definition of the NAIST Text Corpus where case markers in different voices are generalized into the case markers of an active voice.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Usage

### Pass input as an argument

`echo '今日は雨が降る' | showcase`

### Pass input from STDIN

`cat example.txt | showcase`

### Input file format
 
- One raw sentence per line.
- A blank line can be used to segment a document. (Showcase just resets an argument index to zero.)


## Requirements

- Python 3.5 (or higher)
    - We do not support Python 2
- [CaboCha](https://taku910.github.io/cabocha/) with JUMAN dict
- PyTorch 0.4.0

## Instllation

### Step 1. Install Showcase

`pip install showcase-parser`

### Step 2: Download Resources

Resources include following files:

- 10 Model files for predicate detector (`pred_model_0{0..9}.h5`)
- 10 Model files for argument detector (`arg_model_0{0..9}.h5`)
- Word embedding Matrix (`word_embedding.npz`)
- POS embedding Matrix (`pos_embedding.npz`)
- Word index file (`word.index`)
- Part-of-Speech tag index file (`pos.index`)

Resources are all available at [Google Drive](https://drive.google.com/drive/folders/1AK_oWgx1jd5cF2QAGv--r63ky0dgd52C?usp=sharing).

- train/*.h5: models trained with the training set described in the paper.
- train-test/*.h5:  models trained with the training and test sets.

### Step 3: Create and edit config.json

Run `showcase setup` to create `config.json` file in `$HOME/.config/showcase`.

Then edit `config.json` and specify valid paths for:

- Resources downloaded in Step 2
- CaboCha and its JUMAN dictionary

Original `config.json` can be found at `showcase/data/config.json` of this repo.

You may specify path to `config.json` as follows:

`showcase -c /path/to/config/config.json`

Note that the apporopriate thresholds (hyperparameters) for arguments differ for each model. 
The thresholds for the provided models are described in the sample config file in each Google Drive directory.  

## (Re-)training
TBA

### Step1: Train word2vec
TBA

### Step2: Train model
TBA

### Step3: Convert word2vec

- run `get_vocab_from_word2vec.py` and `convert_word2vec_to_npy.py`

## Citation

```
@InProceedings{matsubayashi:2018:coling,
  author    = {Matsubayashi, Yuichiroh and Inui, Kentaro},
  title     = {Distance-Free Modeling of Multi-Predicate Interactions in End-to-End Japanese Predicate Argument Structure Analysis},
  booktitle = {Proceedings of the 27th International Conference on Computational Linguistics (COLING)},
  year      = {2018},
}
```

## Contributor
- [Yuichiroh Matsubayashi](http://www.cl.ecei.tohoku.ac.jp/~y-matsu/)
- [Shun Kiyono](http://www.cl.ecei.tohoku.ac.jp/~kiyono/)
