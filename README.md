#  Improving Neural Machine Translation with the Abstract Meaning Representation by Combining Graph and Sequence Transformers

## Environment set up

Install Python3.6+ 

Run https://github.com/jlab-nlp/amr-nmt/blob/main/install_environment.sh

## Training

Download and unzip [data](https://drive.google.com/file/d/1JJhRSahvestQGCvZyhW2BJ6dKeEHbj4w/view?usp=sharing) directory into the project directory. The **data** directory includes processed amr data for the experimented language in the paper. 

Example on en to mg: https://github.com/jlab-nlp/amr-nmt/blob/main/train_mha_concat.sh This uses the main model of the paper. There are other variations. You can find in https://github.com/jlab-nlp/amr-nmt/blob/main/onmt/models/model.py.

**Caveats:** note that for tokenization, you need to train the correponding sentencepiece tokenizer, and change the name to **"sentencepiece.bpe.model"** and put it into the project directory. We already have several trained sentencepiece tokenizers for different languages. You can see it in the project main directory named **sentencepiece.bpe.model.***. When you use them just to rembember to change it to **"sentencepiece.bpe.model"** in the project directory. Besides, for different size of the tokenizer, you need to debug to change the code in the https://github.com/jlab-nlp/amr-nmt/blob/main/reduce_embeding_size.py line 21-68 into corresponding and line 80 - 87 to the corresponding size and language training file.

## Prediction

Example on en to mg: https://github.com/jlab-nlp/amr-nmt/blob/main/predict_mha_concat.sh. This uses the main model of the paper. There are other variations.

## Evaluation

Here are the prediction outputs for the experimented languages: https://github.com/jlab-nlp/amr-nmt/blob/main/pred_outs.zip. Go to https://github.com/jlab-nlp/amr-nmt/tree/main/multeval-0.5.1, unzip the prediction outputs into it and here is the example on en to mg https://github.com/jlab-nlp/amr-nmt/blob/main/multeval-0.5.1/eval_mg_tokenized.sh

## Citation

```tex
@inproceedings{li-flanigan-2022-improving,
    title = "Improving Neural Machine Translation with the {A}bstract {M}eaning {R}epresentation by Combining Graph and Sequence Transformers",
    author = "Li, Changmao  and
      Flanigan, Jeffrey",
    editor = "Wu, Lingfei  and
      Liu, Bang  and
      Mihalcea, Rada  and
      Pei, Jian  and
      Zhang, Yue  and
      Li, Yunyao",
    booktitle = "Proceedings of the 2nd Workshop on Deep Learning on Graphs for Natural Language Processing (DLG4NLP 2022)",
    month = jul,
    year = "2022",
    address = "Seattle, Washington",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.dlg4nlp-1.2",
    doi = "10.18653/v1/2022.dlg4nlp-1.2",
    pages = "12--21",
    abstract = "Previous studies have shown that the Abstract Meaning Representation (AMR) can improve Neural Machine Translation (NMT). However, there has been little work investigating incorporating AMR graphs into Transformer models. In this work, we propose a novel encoder-decoder architecture which augments the Transformer model with a Heterogeneous Graph Transformer (Yao et al., 2020) which encodes source sentence AMR graphs. Experimental results demonstrate the proposed model outperforms the Transformer model and previous non-Transformer based models on two different language pairs in both the high resource setting and low resource setting. Our source code, training corpus and released models are available at \url{https://github.com/jlab-nlp/amr-nmt}.",
}
```

For any questions put it into the github issue or contact me at changmao.li@ucsc.edu.

