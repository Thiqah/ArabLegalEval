import json
import os
import hashlib
import logging
import random
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
import json
import os
import random
import yaml
from llama_index.core.llms import LLM
import glob
from pathlib import Path
from multiprocess.pool import ThreadPool


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check and create necessary directories
os.makedirs('experiment_results', exist_ok=True)


def config2llm(model_config):
    from llama_index.core import PromptTemplate
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


def load_dataset(file_path):
    if not os.path.exists(file_path):
        raise Exception(f'Dataset file does not exist: {file_path}')
    if os.path.getsize(file_path) == 0:
        raise Exception(f'Dataset file is empty: {file_path}')

    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as e:
            raise Exception(f'Error decoding JSON from file {file_path}: {str(e)}')

    processed_data = [
        {
            'question': item['Question_translation'],
            'answer': item['answer_Translation'],
            'contract': item['contract_Translation'],
        }
        for item in data
    ]

    return processed_data


def load_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()


def generate_experiment_id(model_name, technique_name, prompt):
    """
    generate a filename based on the model, technique, and prompt
    let's say we have model_name=GPT4 and technique_name=technique1 and prompt="prompt1....."
    only the prompt is hashed and we want just part of the hash
    """
    # Hash the prompt
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:6]
    return f'{model_name}_{technique_name}_{prompt_hash}'


def postprocess_prediction(prediction: str) -> str:
    return prediction.replace('الصلة', 'صلة').replace('أ', 'ا').strip().split('\n')[0].strip('.').strip()


def run_benchmarks():
    # Load configuration
    llm_config = yaml.safe_load(open('../llm_config.yaml', 'r', encoding='utf8'))
    llms = {model_name: config2llm(model_config) for model_name, model_config in llm_config['models'].items()}

    dataset_files = [f for path in llm_config['ArLegalBench']['dataset_files'] for f in glob.glob(path + '*.json')]
    task_names = [Path(f).parent.parent.stem for f in dataset_files]
    datasets = {Path(f).parent.parent.stem: load_dataset(f) for f in dataset_files}

    for task_name in datasets:
        # logger.info(f'Loaded {len(dataset)} entries for task {task_name}')
        if llm_config['ArLegalBench'].get('sample_size') is not None and llm_config['ArLegalBench'].get('sample_size') < len(datasets[task_name]):
            # logger.info(f'Sampling {llm_config["ArLegalBench"].get("sample_size")} entries for task {task_name}')
            datasets[task_name] = random.sample(datasets[task_name], llm_config['ArLegalBench'].get('sample_size'))

    # benchmarkArLegalBench/prompts/consumer_contract/Fewshots.txt
    task2techniques = {task_name: {Path(f).parent.stem: f for f in glob.glob(f'./prompts/{task_name}/*.txt')} for task_name in task_names}

    def process_run(task_name, dataset, llm_name: str, llm: LLM, technique_name: str, technique: str):
        # Shuffle the dataset
        random.shuffle(dataset)
        # for technique_name, prompt_file in tqdm(techniques.items(), desc='Techniques', leave=False):
        prompt_template = load_prompt(technique)
        experiment_id = generate_experiment_id(llm_name, technique_name, prompt_template)
        experiment_dir = os.path.join('experiment_results', experiment_id)

        # Save results and metadata in JSON
        result_file_path = os.path.join(experiment_dir, 'results.json')

        def model_predict(dataset_entry):
            prompt = prompt_template.format(
                question=dataset_entry['question'],
                clause=dataset_entry['contract'],
            )

            logger.debug(f'Prompt: {prompt}')
            response = llm.complete(prompt)
            prediction = postprocess_prediction(response.text.strip())
            logger.info(f'Prediction: {prediction}')
            return {
                'question': dataset_entry['question'],
                'contract': dataset_entry['contract'],
                'true_label': postprocess_prediction(dataset_entry['answer']),
                'predictions': prediction,
            }

        experiment_results = list(
            tqdm(
                ThreadPool(llm_config['ArLegalBench']['task_parallelism']).imap(model_predict, dataset),
                desc='Dataset Entries',
                total=len(dataset),
                leave=False,
            ),
        )

        all_true_labels = [entry['true_label'] for entry in experiment_results]
        all_predictions = [entry['predictions'] for entry in experiment_results]
        # Calculate aggregated metrics
        metadata = {
            'model': llm_name,
            'technique': technique_name,
            'prompt_template': prompt_template,
            'dataset': task_name,
            'F1 score': f1_score(all_true_labels, all_predictions, average='macro'),
            'Precision': precision_score(all_true_labels, all_predictions, average='macro'),
            'Recall': recall_score(all_true_labels, all_predictions, average='macro'),
            'Accuracy': accuracy_score(all_true_labels, all_predictions),
            'Balanced Accuracy': balanced_accuracy_score(all_true_labels, all_predictions),
            'Number of contracts': len(dataset),
        }
        os.makedirs(experiment_dir, exist_ok=True)

        with open(result_file_path, 'w', encoding='utf-8') as file:
            o = {
                'responses': experiment_results,
                'metadata': metadata,
            }
            json.dump(o, file, ensure_ascii=False, indent=4)

        logger.info(f'Experiment results saved: "{result_file_path}"')

    experiments = [
        (task_name, datasets[task_name], llm_name, llm, technique_name, technique)
        for task_name, techniques in task2techniques.items()
        for technique_name, technique in techniques.items()
        for llm_name, llm in llms.items()
    ]
    logger.info(f'Running {len(experiments)} experiments')

    return list(
        tqdm(
            ThreadPool(llm_config['ArLegalBench']['experiment_parallelism']).imap(lambda x: process_run(*x), experiments),
            desc='Models',
            total=len(experiments),
            leave=False,
        )
    )


if __name__ == '__main__':
    run_benchmarks()
