import argparse
import datetime
import json
import logging
import os
import pprint
import random
import re
import time
from multiprocessing.pool import ThreadPool

# import llama_index.core.instrumentation as instrument
import pandas as pd
import tonic_validate
import yaml
from easydict import EasyDict
from tqdm.auto import tqdm

import utils


def config2llm(model_config):
    import importlib

    # Extracting the class path and parameters from the JSON
    # Splitting the class path to import the module and get the class
    if '.' not in model_config['class']:
        raise ValueError('Class path should be module_name.class_name')

    from copy import deepcopy

    params = deepcopy(model_config['params'])

    if model_config['params'] is not None:
        if 'query_wrapper_prompt' in model_config['params']:
            from llama_index.core import PromptTemplate

            params['query_wrapper_prompt'] = PromptTemplate(model_config['params']['query_wrapper_prompt'])

        def messages_to_prompt(messages):
            sep = model_config['params']['messages_to_prompt']['separator']
            footer = model_config['params']['messages_to_prompt']['footer']
            return sep.join([model_config['params']['messages_to_prompt'][x.role].format(query_str=(x)) for x in messages]) + footer

        if 'messages_to_prompt' in model_config['params']:
            params['messages_to_prompt'] = messages_to_prompt

        if 'completion_to_prompt' in model_config['params']:
            params['completion_to_prompt'] = lambda completion: model_config['params']['query_wrapper_prompt'].format(query_str=completion)

    module_name, class_name = model_config['class'].rsplit('.', 1)

    module = importlib.import_module(module_name)
    Class = getattr(module, class_name)
    return Class(**params)


# dispatcher = instrument.get_dispatcher(__name__)

# openai.log = 'warning'

loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.WARNING)


parser = argparse.ArgumentParser(description='Description of your program', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset_path', '-b', type=str, default='./Najiz_QA_with_context_v2.benchmark.json', help='Path to the benchmark file')
parser.add_argument('--outpath', '-o', type=str, default='outputs/{exp_name}/{datetime_stamp}', help='Output path')
parser.add_argument('--config_path', '-c', type=str, default='resources/defaults_final.yaml', help='Path to the config file')
parser.add_argument('--sample_size', '-l', type=int, default=None, help='Limit of benchmark questions')
# parser.add_argument('--task_parallelism', '-p', type=int, default=5, help='Concurrency')
# parser.add_argument('--experiment_parallelism', type=int, default=None, help='Concurrency')
parser.add_argument('--llm_cache', action='store_true', help='Use LLM cache')
parser.add_argument('--judge_llms', nargs='+', default=None, help='List of judge LLMs')
args = parser.parse_args()


datetime_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

llm_config = yaml.safe_load(open('../llm_config.yaml', 'r', encoding='utf8'))
llms = {model_name: config2llm(model_config) for model_name, model_config in llm_config['models'].items()}


# for each key in args, update the llm_config if the args value is not None
for key, value in vars(args).items():
    if value is not None:
        llm_config['QA'][key] = value

# Update the args object with the values from the dictionary
for key, value in llm_config['QA'].items():
    print('Overriding', key, 'with', value)
    setattr(args, key, value)

# get base filename without extension
exp_name = os.path.splitext(os.path.basename(args.config_path))[0]
args.outpath = args.outpath.format(
    exp_name=exp_name,
    datetime_stamp=datetime_stamp,
)

# Format the current date and time as YYYY-MM-DD-HH-MM-SS

print('===== config =====')
pprint.pprint(vars(args))
print('=================')

###############################################
#
###############################################


config = EasyDict(yaml.safe_load(open(args.config_path, encoding='utf8')))
config


benchmark_df = pd.DataFrame(json.loads(open(args.dataset_path, 'r', encoding='utf8').read()))

# use args.limit
if args.sample_size is not None and args.sample_size < len(benchmark_df):
    benchmark_df = benchmark_df.sample(n=args.sample_size)

tonic_benchmark = tonic_validate.classes.benchmark.Benchmark(list(benchmark_df['questions']), list(benchmark_df['ground_truths']), 'ARAGOG')

llms = utils.get_llms(llms, use_cache=args.llm_cache, benchmark=tonic_benchmark)

judge_llms = [llms[judge_llm_name] for judge_llm_name in args.judge_llms]


config['llms'] = list(sorted(list(llms.keys())))

args.config = vars(config)


def run_question(llm, question, context=None):
    if context is None:
        return llm.complete(config.TEXT_QA_NORAG_TEMPLATE.format(query_str=question))
    else:
        return llm.complete(config.TEXT_QA_TEMPLATE.format(query_str=question, context_str=context))


def benchmark_llm(i_llm_name_llm, with_context=True):
    i, (llm_name, llm) = i_llm_name_llm

    def evaluate_question(question_i_llm_question_ground_truth_context):
        question_i, (question, ground_truth, context) = question_i_llm_question_ground_truth_context
        if not with_context:
            context = None

        # print('question', question)
        # print('ground_truth', ground_truth)
        llm_answer = run_question(llm, question, context)
        # print('llm_answer', llm_answer)

        retries = 10
        for retry in range(retries):
            try:
                # randomly pick each time
                judge_llm = random.choice(judge_llms)
                judge_response = judge_llm.complete(
                    config.ANSWER_SIMILARITY_TEMPLATE
                    +
                    # f"\n\nContext\n{context}" +
                    f'\n\nQuestion: "{question}"'
                    + f'\n\nReference answer: "{ground_truth}"'
                    + f'\n\nReference new answer:\n{llm_answer}'
                )
                # print('judge_response', judge_response)
                judge_score = int(re.search(r'([\d\.]+)', str(judge_response)).group(1))
                break
            except Exception as e:
                print('retry', retry, e)
                judge_score = 0
                time.sleep(8)

        return {
            'question': question,
            'ground_truth': ground_truth,
            'question_i': question_i,
            'llm': llm_name,
            'context': context,
            'llm_answer': str(llm_answer),
            'answer_similarity': judge_score,
        }

    llm_results = list(
        tqdm(
            ThreadPool(args.task_parallelism).imap(
                evaluate_question,
                enumerate(zip(benchmark_df['questions'], benchmark_df['ground_truths'], benchmark_df['context'])),
            ),
            desc=f'benchmarking {llm_name}',
            total=len(benchmark_df),
            position=i,
            leave=False,
        )
    )
    return llm_results


results_list = list(ThreadPool(args.experiment_parallelism).imap(lambda x: benchmark_llm(x, with_context=True), list(enumerate(llms.items()))))
results_list += list(ThreadPool(args.experiment_parallelism).imap(lambda x: benchmark_llm(x, with_context=False), list(enumerate(llms.items()))))

os.makedirs(args.outpath, exist_ok=True)

# flatten
for llm_name, results in zip(llms.keys(), results_list):
    for i, result in enumerate(results):
        result['llm'] = llm_name
        result['question_i'] = i
        # result['Experiment'] = llm_name + ' | ' + ("w Context" if result['context'] else "w/o Context")

results = [item for sublist in results_list for item in sublist]

# join where left key is question_i and right key is index and make sure to duplicate missing values
results_df = pd.DataFrame(results).join(benchmark_df, on='question_i', rsuffix='_r', how='outer')

results_df['Experiment'] = results_df['llm'] + ' | ' + results_df['context'].apply(lambda x: 'with Context' if x else 'w/o Context')
df_outpath = os.path.join(args.outpath, 'results.csv')
results_df.to_csv(df_outpath, index=False)
print('results be saved here', df_outpath)
results_df
# # Append the results of this experiment to the master DataFrame
# experiments_results_df = pd.concat(experiment_results_dfs, ignore_index=True)
# # Assuming experiments_results_df is your DataFrame
# experiments_results_df['RetrievalPrecision'] = experiments_results_df['OverallScores'].apply(lambda x: x.get('retrieval_precision', None))

# os.makedirs(args.outpath, exist_ok=True)
# with open(os.path.join(args.outpath, 'args.yaml'), 'w', encoding='utf8') as f:
#     yaml.dump(vars(args), f, default_flow_style=False, allow_unicode=True)
