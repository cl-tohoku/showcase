# Installation

## Test

```eval_rst
.. note::
    ここに注釈を書くことができる
```

```eval_rst
.. warning::
    ここに警告文を書くことができる
```
<i class="fa fa-exclamation-triangle" aria-hidden="true"></i>Font Awesomeが使える！

## Recommended Environments

We recommend following environments:

- [Ubuntu](https://www.ubuntu.com/) 14.04/16.04 LTS 64bit
- CUDA 8.0
- CuDNN 5.1

## Dependencies

- Python 3.5 (or higher)
- [CaboCha](https://taku910.github.io/cabocha/)
    - Make sure to install [mecab-jumandic](http://sourceforge.net/projects/mecab/files/)

## Install Showcase

### Step 1. Install Showcase
`pip install git+https://github.com/cl-tohoku/showcase`

or

`pip install git+ssh://git@github.com/cl-tohoku/showcase`

### Step 2. Configure Environment Variable
Showcase downloads resources (e.g. model files, word indices, etc...) to the directory indicated by the environment variable `SHOWCASE_ROOT` .
The default is: `${HOME}/.showcase`

Open `.bashrc` / `.zshrc` and:
`export SHOWCASE_ROOT=<ANY DIRECTORY>`

```eval_rst
.. note::
    (乾研の人は `export SHOWCASE_ROOT=/home/kiyono/deploy/showcase/data` すると当面は動きます．
    これ以降のステップは無視してOKです．)
```

### Step 3: Download Model Files
Execute `showcase setup`
良いようにモデルファイルをダウンロードして， `SHOWCASE_ROOT` に配置すること．

### Step 4: Create config.yaml
Create `config.yaml` in `SHOWCASE_ROOT`.

Write necessary variables as follows:
```config.yaml
cabocha_path: <path_to_cabocha>
juman_dic_path: <path_to_juman_dictionary>
juman_dep_model_path: <path_to_juman_dep_model>
juman_chunk_model_path: <path_to_juman_chunk_model>

# モデルの次元数
embed_dim: 256  # DO NOT CHANGE THIS
rnn_dim: 256  # DO NOT CHANGE THIS
pooling_dim: 1024  # DO NOT CHANGE THIS
rnn_depth: 10  # DO NOT CHANGE THIS
```

Example is available in `data/config.yaml` .
