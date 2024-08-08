# ArabLegalEval: A Multitask Benchmark for Assessing Arabic Legal Knowledge in Large Language Models

## Structure

- `data/` is handled using `DVC`, details can be found in [data/README.md](data/README.md)
- `RAG/` here you can find code for Retrieval Augmented Generation (RAG) and benchmarking. This is a modification of [https://github.com/predlico/ARAGOG](https://github.com/predlico/ARAGOG)

The ArLegalEval dataset is composed of 3 main parent tasks. Evaluation code for each of these tasks can be found in its directory as follows:
- `[benchmarkQA/](benchmarkQA/)` contains the question-answering task
- `[benchmarkArLegalBench/](benchmarkArLegalBench/)` contains the translated subset of LegalBench
- `[benchmarkMCQs/](benchmarkMCQs/)` contains the multiple-choice questions task

In this folder `data/processed/ArabLegalEval` you will find the benchmark as follows:

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
