"""
converts the input path from NajizFAQ format to ARAGOG benchmark format
output path is the same as the input path but with .benchmark.json extension
"""
from bs4 import BeautifulSoup
import re
import json
import argparse

parser = argparse.ArgumentParser(__doc__)
parser.add_argument(
    'najizFAQ_json',
    type=str,
)
args = parser.parse_args()

if __name__ == '__main__':
    with open(args.najizFAQ_json, 'r', encoding='utf8') as f:
        najizQA = json.load(f)

    def get_innder_text_from_HTML_string(html):
        soup = BeautifulSoup(html, 'html.parser')  # removing tags
        return soup.get_text()

    def remove_double_spaces(text):
        return re.sub(' +', ' ', text.replace('\xa0', ' ')).strip()

    benchmark = [
        {
            'question': remove_double_spaces(get_innder_text_from_HTML_string(x['question'])),
            'ground_truth': remove_double_spaces(get_innder_text_from_HTML_string(x['answer'])),
        }
        for x in najizQA
    ]

    # transpose JSON from records to columns
    benchmark = {
        'questions': [x['question'] for x in benchmark],
        'ground_truths': [x['ground_truth'] for x in benchmark],
    }

    # save as same exact input path but instead of .json, make it .benchmark.json
    outpath = args.najizFAQ_json.replace('.json', '.benchmark.json')
    with open(outpath, 'w') as f:
        json.dump(benchmark, f, indent=4, ensure_ascii=False)
    print(f'Saved benchmark to {outpath}')
