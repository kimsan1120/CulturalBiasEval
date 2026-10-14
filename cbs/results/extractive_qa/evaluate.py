from utils.file_utils import read_json
from pathlib import Path
import argparse
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from collections import defaultdict
import re

def split_curture(dataset, results):
    korean_dataset, korean_results = [], []
    western_dataset, western_results = [], []
    for datum, result in zip(dataset, results):
        culture = datum['entity']['Culture']
        if culture == 'Korean':
            korean_dataset.append(datum)
            korean_results.append(result)
        elif culture == 'Western':
            western_dataset.append(datum)
            western_results.append(result)
        else:
            raise ValueError(f'Invalid culture: {culture}')
        
    return korean_dataset, korean_results, western_dataset, western_results

def calculate_accuracy(korean_dataset, korean_results, western_dataset, western_results, language, model_name):
    accuracy = defaultdict(list)
    for korean_datum, korean_result in zip(korean_dataset, korean_results):
        if language == 'ko':
            answer_entity = korean_datum['entity']['Korean Entity']
        elif language == 'en':
            answer_entity = korean_datum['entity']['English Entity']
        else:
            raise ValueError(f'Invalid language: {language}')
        #response_entity = get_response_entity(korean_result, model_name)
        response_entity = korean_result
        accuracy['korean'].append(answer_entity in response_entity)
    for western_datum, western_result in zip(western_dataset, western_results):
        if language == 'ko':
            answer_entity = western_datum['entity']['Korean Entity']
        elif language == 'en':
            answer_entity = western_datum['entity']['English Entity']
        else:
            raise ValueError(f'Invalid language: {language}')
        #response_entity = get_response_entity(western_result, model_name)
        response_entity = western_result
        accuracy['western'].append(answer_entity in response_entity)
    
    accuracy = {key: np.mean(value).item() * 100 for key, value in accuracy.items()}
    accuracy['dif'] = accuracy['western'] - accuracy['korean']
    accuracy = {key: round(value, 2) for key, value in accuracy.items()}
    return accuracy

def analysis_results(args):
    experiment_name = f'{args.model_name}_{args.prompt_culture}_{args.language}_{args.prompt_sample_count}_{args.entity_sample_count}'
    if args.special_name is not None:
        experiment_name = f'{experiment_name}_{args.special_name}'
    results = read_json(f'{experiment_name}.json')

    logger.info(f'Calculating CBS for {experiment_name}')
    
    accuracy_dict = {}
    averages_dict = defaultdict(list)
    result_keys = sorted(list(results.keys()))
    for key in result_keys:
        value = results[key]
        korean_dataset, korean_results, western_dataset, western_results = split_curture(value['dataset'], value['all_outputs'])
        accuracy = calculate_accuracy(korean_dataset, korean_results, western_dataset, western_results, args.language, args.model_name)
        accuracy_dict[key] = accuracy
        logger.info(f'{key}: {accuracy}')
        for key, val in accuracy.items():
            averages_dict[key].append(val)
    
    averages_dict = {key: round(np.mean(value).item(), 2) for key, value in averages_dict.items()}
    logger.info(f'Averages: {averages_dict}')

    return accuracy_dict

def draw_figure(ko_dict1, en_dict2, args):
    categories = sorted(list(ko_dict1.keys()), reverse=True)
    ko_values = [ko_dict1[category] for category in categories] # Start
    en_values = [en_dict2[category] for category in categories] # End

    fig, ax = plt.subplots(figsize=(5, 10))
    
    ax.scatter(ko_values, categories, color='deepskyblue', label='Korean', s=50, zorder=3)
    ax.scatter(en_values, categories, color='orangered', label='English', s=50, zorder=3)

    for cat, ko, en in zip(categories, ko_values, en_values):
        ax.annotate(
            '',
            xy=(en, cat),
            xytext = (ko, cat),
            arrowprops=dict(arrowstyle='->', color='orangered', lw=2),
        )
    ax.set_title(args.model_name, fontsize=20, fontweight='bold')
    ax.set_xlabel('CBS', fontsize=12, fontweight='bold')
    ax.set_xlim(10, 90)

    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f'analysis/{args.model_name}.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='LLAMA-3.1-8B-Instruct', help='Model name in model_configs.yaml')
    parser.add_argument('--prompt_culture', type=str, default='co', help='Prompt culture in prompt dataset: (co, ag)')
    parser.add_argument('--language', type=str, default='ko', help='Language of the text', choices=['ko', 'en'])
    parser.add_argument('--prompt_sample_count', type=int, default=15, help='Number of prompts to sample')
    parser.add_argument('--entity_sample_count', type=int, default=50, help='Number of entities to sample')
    parser.add_argument('--special_name', type=str, default=None, help='Special name for the experiment')
    parser.add_argument('--compare', action='store_true', help='Compare the results on language')
    parser.add_argument('--draw_figure', action='store_true', help='Draw the figure')
    args = parser.parse_args()

    src_cbs_dict = analysis_results(args)
    
    if args.compare:
        compare_language = 'en' if args.language == 'ko' else 'ko'
        new_args = deepcopy(args)
        new_args.language = compare_language
        tgt_cbs_dict = analysis_results(new_args)
        if args.draw_figure:
            ko_cbs_dict = src_cbs_dict if args.language == 'ko' else tgt_cbs_dict
            en_cbs_dict = src_cbs_dict if args.language == 'en' else tgt_cbs_dict
            draw_figure(ko_cbs_dict, en_cbs_dict, args)