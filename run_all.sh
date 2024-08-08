#!/bin/bash

tmux new-session -d -s benchmark

tmux split-window -h
tmux split-window -h

tmux send-keys -t 0 "conda activate ArabLegalEval ; cd benchmarkArLegalBench ; python run_benchmark.py" C-m
tmux send-keys -t 1 "conda activate ArabLegalEval ; cd benchmarkMCQs ; python run_benchmark.py" C-m
tmux send-keys -t 2 "conda activate ArabLegalEval ; cd benchmarkQA ; python run_benchmark.py" C-m

tmux select-pane -t 0

tmux attach-session -t benchmark
