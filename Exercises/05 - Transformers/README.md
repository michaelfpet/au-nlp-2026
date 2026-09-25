# 05 - Transformers

This tutorial is about attention and the Transformer. You implement **scaled dot-product attention** by hand, build a complete **encoder–decoder Transformer** in PyTorch and train it on a small translation task, visualise what its **cross-attention** has learned, and then work with a full-size pre-trained **English-to-French translation model**: you evaluate it with **BLEU** and fine-tune it on a new domain.

## Notebooks
- `0. Attention.ipynb` — the warm-up notebook: attention, the scaling, masks and positional encodings, one small piece at a time. No training, no downloads, and every cell runs in a second.
- `0. Attention - Solution.ipynb` — the worked solutions. Try to finish a part before looking at them.
- `1. Transformer.ipynb` — the main exercise notebook. You put the pieces together into an encoder–decoder Transformer and train it to translate dates (`saturday, 12 september 2026` → `2026-09-12`). Work through it in order and fill in every `# TODO`. No downloads; the training takes one to two minutes on a CPU.
- `1. Transformer - Solution.ipynb` — the worked solutions for the main notebook.
- `2. Pretrained Translation.ipynb` — the follow-up notebook. The same architecture at full size: you load `Helsinki-NLP/opus-mt-en-fr` with the Hugging Face `transformers` library, translate sentences from novels, compute BLEU, look at the cross-attention of a real model, and fine-tune it with the `Seq2SeqTrainer`.
- `2. Pretrained Translation - Solution.ipynb` — the worked solutions for the follow-up notebook.

## What you will practice
In `0. Attention.ipynb`:
1. **Attention by hand** — queries, keys and values, the four steps of scaled dot-product attention, and self-attention on a four-word sentence
2. **Why divide by $\sqrt{d_k}$?** — the variance of a dot product, and what a saturated softmax does to the gradients
3. **Masking** — the causal mask and the padding mask, and why they are applied before the softmax
4. **Word order** — self-attention is blind to it, and sinusoidal positional encodings fix that

In `1. Transformer.ipynb`:
1. **The task** — generating date pairs, a character vocabulary with `<pad>`, `<s>` and `</s>`, and padded batches
2. **Multi-head attention** — splitting into heads and merging them again, checked against `nn.MultiheadAttention`
3. **Encoder and decoder layers** — self-attention, cross-attention, feed-forward, residual connections and layer norm
4. **The full model** — embeddings, learned positions and the masks, with tests for causality and padding
5. **Training and decoding** — teacher forcing, overfitting a single batch as a sanity check, the training run, and greedy decoding
6. **What did the model learn?** — cross-attention heatmaps, and inputs from outside the training distribution
7. **Optional** — the same model with `nn.Transformer`
8. **MCQs** — on encoder–decoder models, scaled dot-product attention, attentive decoders, the Transformer block and self-attention

In `2. Pretrained Translation.ipynb`: **a pre-trained translation model** — its size and where the parameters sit, subword tokenization of sources and targets, `generate`, BLEU and what it does and does not measure, cross-attention as a word alignment, and fine-tuning on 1,000 sentence pairs.

Most implementation tasks are followed by a **✅ Check** cell that verifies your code automatically, so you can confirm each part before moving on.

## Before you start
The notebooks need the updated project environment (`uv sync`), which now includes `sentencepiece`, `sacrebleu` and `accelerate` for the follow-up notebook.

`0. Attention.ipynb` and `1. Transformer.ipynb` need no downloads. A GPU is not required: the main notebook trains its model in one to two minutes on a CPU (and the optional Part 7 trains a second one).

`2. Pretrained Translation.ipynb` downloads data, so make sure you have a working internet connection and start these cells early:
- the **`Helsinki-NLP/opus-mt-en-fr`** model and tokenizer (~600 MB, cached after the first run),
- the English–French part of the **`opus_books`** dataset from the Hugging Face Hub (~25 MB).

On a CPU, translating the 200 test sentences takes about half a minute and the fine-tuning about 3–5 minutes, so start these cells as soon as they are ready and read ahead while they run.

## Further reading
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [How do Transformers work?](https://huggingface.co/learn/llm-course/en/chapter1/4) (Hugging Face LLM course)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)

Please give us feedback for this tutorial!

![LS Feedback 5](../QRs/qrf5.png)
