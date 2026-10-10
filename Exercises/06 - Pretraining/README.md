# 06 - Pretraining

This tutorial is about pretraining. You build a small **GPT** from scratch and **pretrain** it on Shakespeare with nothing but next-character prediction. Then you sample text from it, and estimate how much more compute GPT-3 needed. After that you work with three real **pre-trained models** from Hugging Face, one from each family of Transformers: **BERT** (encoder-only), **GPT-2** (decoder-only) and **T5** (encoder–decoder).

## Notebooks
- `0. Mini GPT.ipynb`: the main exercise notebook. You build a decoder-only Transformer, pretrain it on Tiny Shakespeare and generate text with it. Work through it in order and fill in every `# TODO`. The model trains in a few minutes on a CPU.
- `0. Mini GPT - Solution.ipynb`: the worked solutions. Try to finish a part before looking at them.
- `1. Pretrained Models.ipynb`: the follow-up notebook. You load BERT, GPT-2 and T5 with the Hugging Face `transformers` library and explore what their pretraining objectives taught them.
- `1. Pretrained Models - Solution.ipynb`: the worked solutions for the follow-up notebook.
- `2. Positional Encodings.ipynb`: a second follow-up to the mini GPT. You implement RoPE and ALiBi, train the mini GPT with five different positional encodings, and compare them empirically: on Shakespeare, on sequences longer than in training, and on a task that needs exact relative positions. Runs on a CPU.
- `2. Positional Encodings - Solution.ipynb`: the worked solutions for the positional encodings notebook.

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

In `2. Positional Encodings.ipynb`:
1. **RoPE**: rotating queries and keys, why the scores depend only on relative positions, and long-term decay
2. **ALiBi**: distance penalties per head, and the recency windows they create
3. **One model, five encodings**: none (NoPE), learned, sinusoidal, RoPE and ALiBi in the same mini GPT, and how a causal decoder without positional encoding still learns about order
4. **Language modelling**: the five methods on Tiny Shakespeare at the training length
5. **Length extrapolation**: the loss per position on sequences four times longer than in training, and RoPE position interpolation
6. **Exact positions**: a synthetic look-back task, in and beyond the training length
7. **MCQs**: on positional encodings

Most implementation tasks are followed by a **✅ Check** cell that verifies your code automatically, so you can confirm each part before moving on.

## Before you start
The notebooks use the project environment (`uv sync`). There are no new packages this week.

`0. Mini GPT.ipynb` downloads Tiny Shakespeare (~1 MB) the first time you run it. With the default configuration the model trains in a few minutes on a laptop CPU. If you have a GPU (for example a free one on Colab or Kaggle, see `COMPUTE_RESOURCES.md`), you can switch to the larger configuration in the setup cell for much better samples.

`1. Pretrained Models.ipynb` downloads three models, about 1.2 GB in total, so make sure you have a working internet connection and start the loading cells early:
- **`google-bert/bert-base-uncased`** (~440 MB),
- **`openai-community/gpt2`** (~550 MB),
- **`google-t5/t5-small`** (~240 MB).

They are cached after the first download. Nothing is trained in this notebook, so a CPU is enough.

`2. Positional Encodings.ipynb` trains ten tiny models, which takes roughly 5–10 minutes on a laptop CPU in total. It reuses `input.txt` from `0. Mini GPT.ipynb` (and downloads it if it is missing).

## Further reading
- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) and [nanoGPT](https://github.com/karpathy/nanoGPT) by [Andrej Karpathy](https://karpathy.ai/), which the mini GPT is based on
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) and [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/)
- [How do Transformers work?](https://huggingface.co/learn/llm-course/en/chapter1/4) (Hugging Face LLM course)
- [BERT](https://arxiv.org/abs/1810.04805) (Devlin et al., 2019), [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (Radford et al., 2019), [T5](https://arxiv.org/abs/1910.10683) (Raffel et al., 2020) and [Chinchilla](https://arxiv.org/abs/2203.15556) (Hoffmann et al., 2022)
- [RoPE](https://arxiv.org/abs/2104.09864) (Su et al., 2021), [ALiBi](https://arxiv.org/abs/2108.12409) (Press et al., 2022), [NoPE](https://arxiv.org/abs/2305.19466) (Kazemnejad et al., 2023) and [Position Interpolation](https://arxiv.org/abs/2306.15595) (Chen et al., 2023)

Please give us feedback for this tutorial!

![LS Feedback 6](../QRs/qrf6.png)
