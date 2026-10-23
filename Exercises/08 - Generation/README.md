# 08 - Generation

This tutorial is about how a language model, which only ever predicts the next token, writes whole texts. You get to know Hugging Face's **`generate`** API: its settings, what it returns, and how to batch prompts. You then compare the classic **decoding strategies** on GPT-2: **greedy search**, **beam search**, and **sampling** with **temperature**, **top-k** and **top-p**. You measure how likely and how repetitive their texts are, and you reproduce the figure that shows why the most likely text is not the best text. Finally, you implement all of these strategies **from scratch**, add a **KV cache** and **min-p** sampling, and plug your own rules into `generate`.

## Notebooks
- `0. Generation Intro.ipynb`: the warm-up notebook. You learn what goes into and comes out of `generate`, control length and stopping with `GenerationConfig`, shape the output with repetition penalties, n-gram blocking and banned words, read token probabilities from the scores, and generate for a batch of prompts. Runs on a CPU.
- `0. Generation Intro - Solutions.ipynb`: the worked solutions. Try to finish a part before looking at them.
- `1. Decoding Strategies.ipynb`: the main exercise notebook. You compare greedy search, beam search, n-gram penalties and sampling (pure, temperature, top-k, top-p) on GPT-2, with the log-probability and distinct-n of every text, and reproduce the human-vs-beam-search figure of Holtzman et al. Runs on a CPU.
- `1. Decoding Strategies - Solutions.ipynb`: the worked solutions for the main notebook.
- `2. Decoding from Scratch.ipynb`: the follow-up notebook. You implement greedy search (with and without a KV cache), beam search, and top-k, top-p and min-p sampling yourself, check them against `generate`, and write your own logits processors. Runs on a CPU.
- `2. Decoding from Scratch - Solutions.ipynb`: the worked solutions for the follow-up notebook.

## What you will practice
In `0. Generation Intro.ipynb`:
1. **`generate`**: the inputs and outputs, and how to keep only the new text
2. **`GenerationConfig`, length and stopping**: `max_new_tokens`, `min_new_tokens`, stop strings, and the EOS token as just another token
3. **Shaping the output**: `repetition_penalty`, `no_repeat_ngram_size`, and `bad_words_ids` with BPE tokens
4. **Looking inside**: `return_dict_in_generate`, the scores of every step, token probabilities, and processed scores vs. raw logits
5. **Batched generation**: why decoder-only models need left padding
6. **MCQs**: on `generate`

In `1. Decoding Strategies.ipynb`:
1. **Measuring a generation**: the log-probability of every generated token, and distinct-n
2. **Greedy search**: and why it repeats itself
3. **Beam search**: by hand and with `num_beams`, the beam scores, and the length penalty
4. **N-gram penalties**: and where they break
5. **Is the most likely text the best text?**: token probabilities of human text vs. beam search
6. **Sampling**: pure sampling, temperature, top-k, top-p, and the size of the nucleus
7. **All strategies side by side**: the trade-off between likely and diverse text, and which strategy to use for which task
8. **MCQs**: on decoding strategies

In `2. Decoding from Scratch.ipynb`:
1. **Greedy search**: the decoding loop, checked token by token against `generate`
2. **The KV cache**: feeding only the new token, and what it saves and costs
3. **Beam search**: why the recursive version is exhaustive search, and the real algorithm with a fixed number of beams, checked against `generate`
4. **Sampling**: temperature, top-k, top-p and min-p as filters on the logits, a reproducible sampling loop, and how many tokens each filter keeps
5. **Your own logits processors**: a repetition penalty, checked against `generate`, and a lipogram without the letter *e*
6. **MCQs**: on implementing decoding

Most implementation tasks are followed by a **✅ Check** cell that verifies your code automatically, so you can confirm each part before moving on.

## Before you start
The notebooks use the project environment (`uv sync`). There are no new packages this week.

All three notebooks use **`openai-community/gpt2`** (~550 MB), which is still cached if you did week 6. Nothing is trained, and everything runs on a CPU:
- `0. Generation Intro.ipynb` takes a few seconds per cell.
- `1. Decoding Strategies.ipynb` takes one to two minutes for the comparison in Part 7 (25 generations) and a few seconds for everything else.
- `2. Decoding from Scratch.ipynb` takes about half a minute for the timing experiment in Part 2 and a few seconds for everything else.

## Further reading
- [How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate) by Patrick von Platen, which `1. Decoding Strategies.ipynb` and its figures are based on
- [Generation strategies](https://huggingface.co/docs/transformers/generation_strategies) and the [`GenerationConfig` reference](https://huggingface.co/docs/transformers/main_classes/text_generation) in the Hugging Face documentation
- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) (Holtzman et al., 2020), [Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833) (Fan et al., 2018), [CTRL](https://arxiv.org/abs/1909.05858) (Keskar et al., 2019) and [Min-p Sampling](https://arxiv.org/abs/2407.01082) (Nguyen et al., 2025)

## Credits
`1. Decoding Strategies.ipynb` draws on the explanations and graphics of the Hugging Face blog post [How to generate text](https://huggingface.co/blog/how-to-generate) by [Patrick von Platen](https://huggingface.co/patrickvonplaten).

Please give us feedback for this tutorial!

![LS Feedback 8](../QRs/qrf8.png)
