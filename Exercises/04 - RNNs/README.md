# 04 - RNNs

This tutorial is about classifying text with recurrent networks. You load the **AG News** dataset, tokenize it with a pre-trained tokenizer, build a **vanilla RNN** classifier in PyTorch, write the training and evaluation loop, and then swap the recurrent layer for an **LSTM** and a **GRU** to see whether gating helps.

## Notebooks
- `0. RNNs.ipynb` — the exercise notebook. Work through it in order and fill in every `# TODO`.
- `0. RNNs - Solution.ipynb` — the worked solutions. Try to finish a part before looking at them.

## What you will practice
1. **Exploring the data** — the size of AG News, the average article length and the label distribution, and why macro-F1 rather than accuracy
2. **From text to tensors** — turning articles into fixed-length sequences of token ids with the BERT tokenizer, checking how much truncation costs, and building `DataLoader`s
3. **The RNN classifier** — an `nn.Embedding` layer, an `nn.RNN` layer and a linear layer on the final hidden state, plus what padding does to a recurrent model
4. **Training and evaluation** — per-class and macro-F1, the evaluation loop, and the training loop with Adam and gradient clipping
5. **LSTM and GRU** — implementing both, comparing parameter counts, and plotting all three models against each other
6. **MCQs** — on the forget gate, GRU vs. LSTM, vanishing gradients in BPTT, truncated BPTT, long dependencies and the final hidden state

Most implementation tasks are followed by a **✅ Check** cell that verifies your code automatically, so you can confirm each part before moving on.

## Before you start
The notebook needs the updated project environment (`uv sync`), which now includes `transformers` for the tokenizer.

Two cells download data, so make sure you have a working internet connection and start them early:
- the **AG News** dataset from the Hugging Face Hub (~30 MB, 120k training and 7.6k test articles),
- the **`bert-base-uncased` tokenizer** (cached after the first run).

A GPU is not required, but the notebook trains three models for 3 epochs each on 120k articles. On a CPU that is roughly 3–6 minutes per model, so start each training cell as soon as it is ready and read ahead while it runs.

Please give us feedback for this tutorial!

![LS Feedback 4](../QRs/qrf4.png)
