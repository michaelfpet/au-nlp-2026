# 07 - Finetuning and Prompting

This tutorial is about adapting pre-trained models to tasks. You **fine-tune** DistilBERT to classify the sentiment of movie reviews and compare it with a classifier on *frozen* features. Then you **prompt** a large hosted LLM to solve math word problems, with and without chain-of-thought. Finally, you go through the two training stages that turn a base model into an assistant: **supervised fine-tuning** (in full and with **LoRA**) and **preference alignment** with **DPO**.

## Notebooks
- `0. Finetuning.ipynb`: the main exercise notebook. You fine-tune `distilbert-base-uncased` on IMDB with the Hugging Face `Trainer`, and compare it with logistic regression on the frozen model's `[CLS]` vectors. Work through it in order and fill in every `# TODO`. Runs on a CPU.
- `0. Finetuning - Solutions.ipynb`: the worked solutions. Try to finish a part before looking at them.
- `1. Prompting.ipynb`: you prompt a 27 B parameter model through the free Groq API to solve GSM8K problems: zero-shot and few-shot, direct answers and chain-of-thought, and self-consistency. No GPU needed, but you need a free Groq account.
- `1. Prompting - Solutions.ipynb`: the worked solutions for the prompting notebook.
- `2. Preference Alignment.ipynb`: you turn the base model Qwen2.5-0.5B into a GSM8K solver with supervised fine-tuning (full and LoRA) using TRL's `SFTTrainer`, and align it with DPO on preference pairs. **Needs a GPU** (a free T4 on Colab or Kaggle is enough).
- `2. Preference Alignment - Solutions.ipynb`: the worked solutions for the alignment notebook.

## What you will practice
In `0. Finetuning.ipynb`:
1. **The data**: IMDB with the `datasets` library, balanced random subsets (the dataset is sorted by label!), and review lengths in tokens
2. **The model**: DistilBERT with a randomly initialised classification head, batched prediction, and accuracy, F1 and per-class accuracy before training
3. **Fine-tuning**: tokenizing with `Dataset.map`, dynamic padding, `TrainingArguments`, `compute_metrics` and the `Trainer`, and an error analysis
4. **Frozen features**: logistic regression on `[CLS]` vectors, compared with fine-tuning in accuracy and trained parameters
5. **MCQs**: on fine-tuning

In `1. Prompting.ipynb`:
1. **Setup**: a Groq API key, kept out of the notebook, and a cached `chat` function
2. **The chat format**: system, user and assistant messages, stateless conversations, and temperature
3. **GSM8K**: parsing the reference solutions into reasoning steps and a final answer
4. **Getting a number out**: robust extraction of the final answer from free text
5. **Direct answers**: zero-shot and few-shot prompts built from demonstrations
6. **Chain-of-thought**: zero-shot and few-shot, accuracy against token cost, and an error analysis
7. **Self-consistency**: sampling several reasoning paths and a majority vote
8. **MCQs**: on prompting

In `2. Preference Alignment.ipynb`:
1. **The base model**: what a pretrained model does with a question, and the chat template
2. **The data**: GSM8K as prompt–completion conversations, and a loss on the completion only
3. **Measuring accuracy**: batched greedy generation and answer extraction
4. **Supervised fine-tuning**: all 494 M parameters with TRL's `SFTTrainer`
5. **LoRA**: counting its parameters, the memory it saves, and LoRA fine-tuning with PEFT
6. **DPO**: the DPO objective, Step-DPO preference pairs, implicit rewards and margins
7. **MCQs**: on SFT, LoRA and DPO

Most implementation tasks are followed by a **✅ Check** cell that verifies your code automatically, so you can confirm each part before moving on.

## Before you start
The notebooks need the updated project environment (`uv sync`), which now includes `trl`, `peft` and `groq`.

`0. Finetuning.ipynb` downloads **`distilbert/distilbert-base-uncased`** (~270 MB) and the **IMDB** dataset (~85 MB). With the default settings, fine-tuning takes roughly 5–15 minutes on a laptop CPU and under a minute on a GPU. A setup cell shows larger settings for a GPU.

`1. Prompting.ipynb` calls the Groq API. Create a free account at [console.groq.com](https://console.groq.com) and an API key **before the session**; no payment details are needed. The free tier is rate-limited (as of October 2026: 8,000 tokens per minute and 200,000 tokens per day for the model we use). The experiments use about half of the daily budget and take 10–15 minutes, mostly waiting for the per-minute limit. Replies are cached in `groq_cache.json`, so re-running a cell costs nothing. Never paste your API key into a notebook cell.

`2. Preference Alignment.ipynb` needs a **GPU**. Open it on Google Colab or Kaggle with a T4 GPU (see `COMPUTE_RESOURCES.md`) and run its install cell first. It downloads **`Qwen/Qwen2.5-0.5B`** (~1 GB), GSM8K (~5 MB) and **`xinlai/Math-Step-DPO-10K`** (~12 MB). Each of its three training runs takes roughly 5–15 minutes on a T4, so plan about an hour for the whole notebook.

## Further reading
- [Fine-tuning a pretrained model](https://huggingface.co/learn/llm-course/en/chapter3/1) (Hugging Face LLM course) and [A Visual Guide to Using BERT for the First Time](https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/) and [Prompt engineering overview](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview)
- [Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf) and the [TRL documentation](https://huggingface.co/docs/trl/index)
- [DistilBERT](https://arxiv.org/abs/1910.01108) (Sanh et al., 2019), [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (Brown et al., 2020), [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) (Wei et al., 2022), [Self-Consistency](https://arxiv.org/abs/2203.11171) (Wang et al., 2023), [LoRA](https://arxiv.org/abs/2106.09685) (Hu et al., 2021), [InstructGPT](https://arxiv.org/abs/2203.02155) (Ouyang et al., 2022), [DPO](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023) and [Step-DPO](https://arxiv.org/abs/2406.18629) (Lai et al., 2024)

Please give us feedback for this tutorial!

![LS Feedback 7](../QRs/qrf7.png)
