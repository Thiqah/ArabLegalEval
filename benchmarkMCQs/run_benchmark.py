import os

os.environ['DSP_CACHEBOOL'] = 'false'
import re
import dspy
import pandas as pd
import yaml
from datasets import load_dataset
from dspy.evaluate import Evaluate
import datetime


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


# Define a Signature
class MCQSignature(dspy.Signature):
    """You are given an Arabic Multiple Choice Question, answer it by choosing the correct option with the letter included."""

    question = dspy.InputField(desc='Arabic MCQ question')
    option = dspy.OutputField(desc='Option')


# Define a module
class MCQ(dspy.Module):
    def __init__(self, signature=MCQSignature, prompting='cot'):
        super().__init__()
        self.init = dspy.ChainOfThought(signature) if prompting == 'cot' else dspy.Predict(signature)

    def forward(self, question):
        return self.init(question=question)


alpa_en = ['A.', 'B.', 'C.', 'D.', 'E.']

level_ar = {'Primary': 'للمدرسة الابتدائية', 'Middle': 'للمدرسة المتوسطة', 'High': 'للمدرسة الثانوية', 'Univ': 'للجامعات', 'Prof': 'للمحترفين'}

country_ar = {
    'UAE': 'في دولة الإمارات العربية المتحدة',
    'Egypt': 'في مصر',
    'Lebanon': 'في لبنان',
    'Jordan': 'في الأردن',
    'Kuwait': 'في الكويت',
    'KSA': 'في المملكة العربية السعودية',
    'Palestine': 'في فلسطين',
    'Morocco': 'في المغرب',
}

subject_ar = {
    'Islamic Studies': 'دراسة إسلامية',
    'Driving Test': 'اختبار القيادة',
    'Natural Science': 'في العلوم الطبيعية',
    'History': 'تاريخي',
    'General Knowledge': 'معرفي عام',
    'Law': 'قانوني',
    'Physics': 'فيزياء',
    'Social Science': 'في العلوم الاجتماعية',
    'Management': 'تسييري',
    'Arabic Language': 'باللغة العربية',
    'Political Science': ' في العلوم السياسية',
    'Philosophy': ' فلسفي',
    'Accounting': 'محاسبي',
    'Computer Science': 'في علوم الحاسوب',
    'Geography': 'جغرافيا',
    'Math': 'رياضي',
    'Biology': 'في علم الأحياء',
    'Economics': 'اقتصادي موجه',
    'Arabic Language (General)': 'في اللغة العربية (عام)',
    'Arabic Language (Grammar)': 'في اللغة العربية (قواعد)',
    'Civics': 'التربية المدنية',
}


def prepare_data_ar(data: pd.DataFrame) -> tuple[list[str], list[str], list[list[str]], list[str]]:
    """
    Prepare the data for the Arabic MMLU dataset in which the instruction is in Arabic.
    Argument:
        data: pandas.DataFrame
            Arabic MMLU dataset
    Returns:
        inputs: list of strings
            Question
        outputs: list of strings
            Answer
        outputs_options: list of string lists
            Options
        subjects: list of strings
            Subject
    """
    PROMPT = 'هذا سؤال [MAIN_META_DATA]. اختر الإجابة الصحيحة!\n\nسؤال: [INPUT]\n[OPTION]\n\nإجابة: '

    alpa = alpa_en

    inputs = []
    outputs = []
    outputs_options = []
    subjects = []

    data = data[data['is_few_shot'] == 0]

    option_list = ['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5']

    for idx, row in data.iterrows():
        subject = subject_ar[row['Subject']]
        level = '' if pd.isna(row['Level']) else ' ' + level_ar[row['Level']]
        country = '' if pd.isna(row['Country']) else ' ' + country_ar[row['Country']]
        main_meta_data = f'{subject}{level}{country}'
        question = row['Question'] if pd.isna(row['Context']) else f"{row['Context']}\n\n{row['Question']}"
        options = []
        for i, opt in enumerate(option_list):
            if pd.isna(row[opt]):
                break
            options.append(f'{alpa[i]} {row[opt]}')
        inputs.append(PROMPT.replace('[MAIN_META_DATA]', main_meta_data).replace('[INPUT]', question).replace('[OPTION]', '\n'.join(options)))
        outputs.append(row['Answer Key'].lower() + '.')
        outputs_options.append(options)
        subjects.append(row['Subject'])
    return inputs, outputs, outputs_options, subjects


def prepare_data_mcq(data: pd.DataFrame, context: bool = False) -> tuple[list[str], list[str], list[list[str]]]:
    """
    Prepare the data for the generated MCQ dataset.
    Arguments:
        data: pandas.DataFrame
            Generated MCQ dataset
        context: bool
            Include context or not
    Returns:
        inputs: list of strings
            Question
        outputs: list of strings
            Answer
        outputs_options: list of string lists
            Options
    """
    if context is False:
        PROMPT = 'هذا سؤال قانوني للعامة في المملكة العربية السعودية. اختر الإجابة الصحيحة!\nسؤال: [INPUT]\n[OPTION]\n\nإجابة: \n'
    else:
        PROMPT = 'هذا سؤال قانوني للعامة في المملكة العربية السعودية. اختر الإجابة الصحيحة!\n\nسياق: [CONTEXT]\nسؤال: [INPUT]\n[OPTION]\n\nإجابة: \n'

    alpa = alpa_en

    inputs = []
    outputs = []
    outputs_options = []
    option_list = ['Option 1', 'Option 2', 'Option 3', 'Option 4']

    for idx, row in data.iterrows():
        question = row['Question']
        context = row['Context'] if context else ''
        options = []
        for i, opt in enumerate(option_list):
            if pd.isna(row[opt]):
                break
            options.append(f'{alpa[i]} {row[opt]}')
        inputs.append(PROMPT.replace('[CONTEXT]', context).replace('[INPUT]', question).replace('[OPTION]', '\n'.join(options)))
        outputs.append(alpa_en[row['Answer Key']].lower())
        outputs_options.append(options)
    return inputs, outputs, outputs_options


def dspy_example(question: list[str], output: list[str], outputs_option: list[list[str]], subject=None) -> list[dspy.Example]:
    """
    Create examples for the dspy module.
    Arguments:
        question: list of strings
            Question
        output: list of strings
            Answer
        outputs_option: list of string lists
            Options
        subject: list of strings
            Subject
    Returns:
        examples: list of dspy.Example
            Examples in dspy format
    """
    if subject is None:
        examples = [
            dspy.Example(question=question[i], output=output[i], outputs_option=outputs_option[i]).with_inputs('question')
            for i in range(len(question))
        ]
    else:
        examples = [
            dspy.Example(question=question[i], output=output[i], outputs_option=outputs_option[i], subject=subject[i]).with_inputs('question')
            for i in range(len(question))
        ]
    return examples


def create_data(examples: list[dspy.Example], n: int = 15) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """
    Create trainset of n*2 examples and return the remaining examples.
    Arguments:
        examples: list of dspy.Example
            Examples
        n: int
            Number of examples * 2 to include in the trainset
    Returns:
        trainset: list of dspy.Example
            Trainset
        examples: list of dspy.Example
            Remaining examples
    """
    trainset = examples[:n] + examples[-n:]
    examples = examples[n:-n]
    return trainset, examples


def em_metric(correct_answer: dspy.Example, predicted_answer: dspy.Example, trace=None) -> int:
    """
    Evaluate the model answer using the exact match metric for option letter.
    Arguments:
        correct_answer: dspy.Example
            Correct answer
        predicted_answer: dspy.Example
            Predicted answer
    Returns:
        int
            1 if the predicted answer is correct, 0 otherwise
    """
    if len(predicted_answer.option.lower().strip(' ')) == 1:
        match = re.search('[abcde]', predicted_answer.option.lower())
    else:
        match = re.search('[abcde]\.', predicted_answer.option.lower())
    if match:
        return 1 if match.group().strip('.') == correct_answer.output.strip('.') else 0
    else:
        return 0


def dspy_eval(dataset: list[dspy.Example], fewshot: bool, prompting: str, filename: str, num_threads: int, model_name=None) -> None:
    """
    Evaluate the model using optimized prompt.
    Arguments:
        dataset: list of dspy.Example
            Dataset
        fewshot: bool
            Few-shot or not
        prompting: str
            Prompt technique
        filename: str
            Filename
        num_threads: int
            Number of threads
    """
    if prompting is not None:
        optimized_program = MCQ(prompting=prompting)
        try:
            state = yaml.safe_load(open(f'{filename}.yaml', 'r', encoding='utf8'))
        except FileNotFoundError:
            gpt4_filename = filename.replace(model_name, 'Llama3-8B')
            state = yaml.safe_load(open(f'{gpt4_filename}.yaml', 'r', encoding='utf8'))

        optimized_program.load_state(state)
    evaluate = Evaluate(devset=dataset, metric=em_metric, display_progress=True, display_table=0, num_threads=num_threads)
    final_score, answers, scores = (
        evaluate(optimized_program, return_all_scores=True, return_outputs=True)
        if fewshot
        else evaluate(MCQ(prompting=prompting), return_all_scores=True, return_outputs=True)
    )
    return {
        'final_score': final_score,
        'answers': answers,
        'scores': scores,
    }


def print_result(dataset: list[dspy.Example], fewshot: bool, prompting: str, model_name: str, teacher: str, num_threads: int, data_name: str) -> None:
    """
    Print the evaluation result.
    Arguments:
        dataset: list of dspy.Example
            Dataset
        fewshot: bool
            Few-shot or not
        prompting: str
            Prompt technique
        model_name: str
            Model name
        teacher: str
            Teacher model name
        num_threads: int
            Number of threads
        data_name: str
            Data name
    """
    # Load the compiled program
    filename = ''
    if prompting is not None:
        filename = f'compiled_prompts/compiled_{model_name}_{prompting}'
        if teacher is not None:
            filename += f'_{teacher}'
    # evaluate the optimized program
    if data_name == 'ArabicMMLU':
        print(f'Law Subject Evaluation'.center(100, '-'))
        law_results = dspy_eval(dataset[:299], fewshot, prompting, filename, num_threads, model_name=model_name)
        print(f'Political Science Subject Evaluation'.center(100, '-'))
        political_science_results = dspy_eval(dataset[299:], fewshot, prompting, filename, num_threads, model_name=model_name)
        return {'Law': law_results, 'Political Science': political_science_results}
    else:
        print(f'Generated MCQ Evaluation'.center(100, '-'))
        results = dspy_eval(dataset, fewshot, prompting, filename, num_threads, model_name=model_name)
        return {'All': results}


def evaluate_llm(
    dataset: pd.DataFrame, model_name: str, fewshot: bool = False, prompting=None, teacher=None, num_threads: int = 10, data_name=None
) -> None:
    """
    Run model evaluation on given data.
    Arguments:
        dataset: pandas.DataFrame
            Dataset
        model_name: str
            Model name
        fewshot: bool
            Few-shot or not
        prompting: str
            Prompt technique
        teacher: str
            Teacher model name
        num_threads: int
            Number of threads
        data_name: str
            Data name
    """
    if data_name == 'ArabicMMLU':
        examples = dspy_example(*prepare_data_ar(dataset))
        # Create trainset of n*2 examples
        trainset, examples = create_data(examples)
        return print_result(examples, fewshot, prompting, model_name, teacher, num_threads, data_name)
    else:
        examples = dspy_example(*prepare_data_mcq(dataset))
        return print_result(examples, fewshot, prompting, model_name, teacher, num_threads, data_name)


def mcq_sample(df: pd.DataFrame, frac=None, random_state: int = 1, engine=None) -> pd.DataFrame:
    """
    Sample the dataset.
    Arguments:
        df: pandas.DataFrame
            Dataset
        frac: float
            Fraction of the dataset to sample
        random_state: int
            Random state
        engine: str
            LLM
    Returns:
        pandas.DataFrame
            Sampled dataset
    """
    if engine is not None:
        df = df[df['Engine'] == engine]  # engine is either gpt-4 or claude-3-opus
    if frac is not None:
        df = df.sample(frac=frac, random_state=random_state)
    return df


def evaluate(dspy_models, dataset_path=None, frac: float = None, sample_size: int = None) -> None:
    """
    Evaluate the models on given data.
    Arguments:
        dataset_path: str
            Data file
    """
    if dataset_path is None:
        dataset = load_dataset('MBZUAI/ArabicMMLU', split='test').to_pandas()
        dataset = dataset[(dataset['Subject'] == 'Law') | (dataset['Subject'] == 'Political Science')]
    else:
        dataset = pd.read_csv(dataset_path).iloc[:10000]
        dataset.drop(columns=dataset.columns[-7:], axis=1, inplace=True)

    llm_results = {}

    # iterate over the models
    for name, model in dspy_models.items():
        with dspy.context(lm=model):
            if name.lower() in ['gpt-4', 'gpt-4o']:
                sample = mcq_sample(dataset, frac=frac, engine='claude-3-opus')
            elif name.lower() == 'claude-3-opus':
                sample = mcq_sample(dataset, frac=frac, engine='gpt-4')
            else:
                sample = mcq_sample(dataset, frac=frac)

            if sample is not None and sample_size is not None:
                sample = sample[:sample_size]

            results = {}
            print(f'{name}'.center(100, '='))
            print(f'Original Prompting'.center(100, '.'))
            results['original_prompt'] = evaluate_llm(sample, name, fewshot=False, prompting=None)
            print(f'Few-shot Prompting'.center(100, '.'))
            results['few_shot_prompt'] = evaluate_llm(sample, name, fewshot=True, prompting='few')
            print(f'Few-shot with CoT Prompting'.center(100, '.'))
            results['few_shot_cot_prompt'] = evaluate_llm(sample, name, fewshot=True, prompting='cot')

            # for gpt4 and claude, do not evaluate few-shot with gpt4 as teacher
            if name.lower() not in ['gpt-4', 'gpt-4o', 'claude-3-opus']:
                print(f'Fewshot with GPT-4 as teacher'.center(100, '.'))
                results['few_shot_gpt4_teacher_prompt'] = evaluate_llm(sample, name, fewshot=True, prompting='few', teacher='gpt4')
                print(f'Few-shot with CoT Prompting with GPT-4 as teacher'.center(100, '.'))
                results['few_shot_cot_gpt4_teacher_prompt'] = evaluate_llm(sample, name, fewshot=True, prompting='cot', teacher='gpt4')
            # model.inspect_history(n=10)
            print(f'=' * 100, '\n\n')
            llm_results[name] = results

    # from multiprocessing.pool import ThreadPool
    # results = []
    # # iterate over the models
    # for llm_name, model in dspy_models.items():
    #     results.append(evaluate_llm(llm_name, model))

    # from tqdm.auto import tqdm
    # results = list(tqdm(
    #     ThreadPool().imap(
    #         lambda x: evaluate_llm_(*x),
    #         dspy_models.items(),
    #     ),
    #     desc='Models', total=len(dspy_models), leave=False
    # ))
    # llm_results = dict(results)

    return llm_results


# Run the evaluation

if __name__ == '__main__':
    # # Load configuration
    llm_config = yaml.safe_load(open('../llm_config.yaml', 'r', encoding='utf8'))
    llms = {model_name: config2llm(model_config) for model_name, model_config in llm_config['models'].items()}

    from dspy_llama_index_wrapper import DspyLlamaIndexWrapper

    dspy_models = {model_name: DspyLlamaIndexWrapper(llm) for model_name, llm in llms.items()}

    # Run the evaluation
    llm_results = evaluate(dspy_models, **llm_config['MCQs'])
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    def my_representer(dumper, data):
        if hasattr(data, 'toDict'):
            return dumper.represent_dict(data.toDict())
        return dumper.represent_data(data)

    # Register the custom representer
    yaml.SafeDumper.add_multi_representer(object, my_representer)

    with open(f'llm_results_{timestamp}.yaml', 'w', encoding='utf8') as file:
        yaml.safe_dump(llm_results, file, default_flow_style=False, sort_keys=False, allow_unicode=True, encoding='utf8')
