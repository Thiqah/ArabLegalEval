# [ArabLegalEval: A Multitask Benchmark for Assessing Arabic Legal Knowledge in Large Language Models](https://www.arxiv.org/abs/2408.07983)

*Faris Hijazi (1), Somayah AlHarbi (1), Abdulaziz AlHussein (1), Harethah Abu Shairah (2), Reem AlZahrani (2), Hebah AlShamlan (1), Omar Knio (2), George Turkiyyah (2) ((1) THIQAH, (2) KAUST)*

## Introduction

**ArabLegalEval** is a benchmark for LLMs to evaluate their ability to reason in the legal domain in the Arabic language.

## Structure

The ArLegalEval dataset is composed of 3 main parent tasks. Evaluation code for each of these tasks can be found in its directory as follows:

- `[benchmarkQA/](benchmarkQA/)`
  - contains the question-answering task
  - task data path: `./benchmarkQA/Najiz_QA_with_context_v2.benchmark.json`
- `[benchmarkArLegalBench/](benchmarkArLegalBench/)`
  - contains the translated subset of LegalBench
  - task data path: `./benchmarkArLegalBench/tasks/<task_name>/<split>/*.json`
- `[benchmarkMCQs/](benchmarkMCQs/)`
  - contains the multiple-choice questions task
  - task data path: `./benchmarkMCQs/All_Generated_MCQs_Evaluated_Shuffled.csv`

- (not yet available) the raw data which is scraped from public sources is in `data/` which is handled using `DVC`, details can be found in [data/README.md](data/README.md)

---

Tested with Python 3.12

## Citing this work

Please include all citations below, which credit all sources ArabLegalEval draws on.

```text
@inproceedings{
      anonymous2024arablegaleval,
      title={ArabLegalEval: A Multitask Benchmark for Assessing Arabic Legal Knowledge in Large Language Models},
      author={Anonymous},
      booktitle={The Second Arabic Natural Language Processing Conference},
      year={2024},
      url={https://openreview.net/forum?id=3EHYXqKKLA}
}
```
