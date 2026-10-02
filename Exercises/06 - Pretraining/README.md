# 06 - Pretraining

This tutorial is about pretraining. You build a small **GPT** from scratch and **pretrain** it on Shakespeare with nothing but next-character prediction. Then you sample text from it, and estimate how much more compute GPT-3 needed. After that you work with three real **pre-trained models** from Hugging Face, one from each family of Transformers: **BERT** (encoder-only), **GPT-2** (decoder-only) and **T5** (encoder–decoder).

## Notebooks
- `0. Mini GPT.ipynb`: the main exercise notebook. You build a decoder-only Transformer, pretrain it on Tiny Shakespeare and generate text with it. Work through it in order and fill in every `# TODO`. The model trains in a few minutes on a CPU.
- `0. Mini GPT - Solution.ipynb`: the worked solutions. Try to finish a part before looking at them.
- `1. Pretrained Models.ipynb`: the follow-up notebook. You load BERT, GPT-2 and T5 with the Hugging Face `transformers` library and explore what their pretraining objectives taught them.
- `1. Pretrained Models - Solution.ipynb`: the worked solutions for the follow-up notebook.

## What you will practice
In `0. Mini GPT.ipynb`:
1. **The data**: a character vocabulary, and batches in which the target is the input shifted by one
2. **What loss should we expect?**: uniform and unigram baselines, and what a loss means as a perplexity
3. **Causal self-attention**: all heads in one module, checked against `F.scaled_dot_product_attention`
4. **The GPT model**: pre-norm blocks, GELU, learned positions, a final layer norm and weight tying, with tests for causality
5. **Pretraining**: overfitting a few sequences as a sanity check, then the training run
6. **Generating text**: sampling with temperature and top-k, and what the model has (not) learned
7. **From mini GPT to GPT-3**: scale, the $6ND$ estimate of training compute, and the Chinchilla rule of thumb
8. **MCQs**: on self-attention, pre-trained language models and GPT

In `1. Pretrained Models.ipynb`:
1. **BERT**: masked language modelling by hand and with a pipeline, why context on both sides matters, social biases learned from the data, and contextual word vectors
2. **GPT-2**: its architecture compared with your mini GPT, BPE tokens, next-token distributions, perplexity, and greedy decoding vs. nucleus sampling
3. **T5**: span corruption with sentinel tokens, and translation, summarisation and classification as text-to-text tasks
4. **MCQs**: on BART and T5

Most implementation tasks are followed by a **✅ Check** cell that verifies your code automatically, so you can confirm each part before moving on.

## Before you start
The notebooks use the project environment (`uv sync`). There are no new packages this week.

`0. Mini GPT.ipynb` downloads Tiny Shakespeare (~1 MB) the first time you run it. With the default configuration the model trains in a few minutes on a laptop CPU. If you have a GPU (for example a free one on Colab or Kaggle, see `COMPUTE_RESOURCES.md`), you can switch to the larger configuration in the setup cell for much better samples.

`1. Pretrained Models.ipynb` downloads three models, about 1.2 GB in total, so make sure you have a working internet connection and start the loading cells early:
- **`google-bert/bert-base-uncased`** (~440 MB),
- **`openai-community/gpt2`** (~550 MB),
- **`google-t5/t5-small`** (~240 MB).

They are cached after the first download. Nothing is trained in this notebook, so a CPU is enough.

## Further reading
- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) and [nanoGPT](https://github.com/karpathy/nanoGPT) by [Andrej Karpathy](https://karpathy.ai/), which the mini GPT is based on
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) and [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/)
- [How do Transformers work?](https://huggingface.co/learn/llm-course/en/chapter1/4) (Hugging Face LLM course)
- [BERT](https://arxiv.org/abs/1810.04805) (Devlin et al., 2019), [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (Radford et al., 2019), [T5](https://arxiv.org/abs/1910.10683) (Raffel et al., 2020) and [Chinchilla](https://arxiv.org/abs/2203.15556) (Hoffmann et al., 2022)

Please give us feedback for this tutorial!

![LS Feedback 6](../QRs/qrf6.png)
