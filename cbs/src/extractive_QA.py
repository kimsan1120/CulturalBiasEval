import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1])) # /mnt/bias/ours
from tqdm import tqdm
from loguru import logger
import argparse

import torch
import torch.nn.functional as F
import deepspeed

from utils.load_model import load_matching_model
from utils.load_dataset import ExtractiveQADataset

from utils.file_utils import append_jsonl, write_json

RESULT_DIR = Path(__file__).resolve().parents[1] / 'results' / 'extractive_qa'

def run_decoder(model, tokenizer, dataset, batch_size, args):
    terminators = [tokenizer.eos_token_id]
    all_outputs = []
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]
        batch_messages = [datum['input']['model_prompt'] for datum in batch]

        #batch_input = tokenizer(batch_messages, return_tensors='pt', padding=True, truncation=True).to(model.device)
        batch_input = [tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
                ) for message in batch_messages
            ]    
        batch_input_tokens = tokenizer(batch_input, return_tensors='pt', padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **batch_input_tokens,
                max_new_tokens=128,
                eos_token_id=terminators,
                do_sample=False,
                temperature=0.0,
                top_p=1,
                top_k=1,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
            )
        string_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        string_inputs = tokenizer.batch_decode(batch_input_tokens['input_ids'], skip_special_tokens=True)
        processed_outputs = [string_output.replace(string_input, '').strip() for string_input, string_output in zip(string_inputs, string_outputs)]
        all_outputs.extend(processed_outputs)

    return all_outputs

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='LLAMA-3.1-8B-Instruct', help='Model name in model_configs.yaml')
    parser.add_argument('--prompt_culture', type=str, default='co', help='Prompt culture in prompt dataset: (co, ag)')
    parser.add_argument('--language', type=str, default='ko', help='Language of the text', choices=['ko', 'en'])
    parser.add_argument('--prompt_sample_count', type=int, default=15, help='Number of prompts to sample')
    parser.add_argument('--entity_sample_count', type=int, default=50, help='Number of entities to sample')
    parser.add_argument('--special_name', type=str, default=None, help='Special name for the experiment')
    parser.add_argument('--entity_type', type=str, default=None, help='Entity type to run')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank passed by DeepSpeed")
    parser.add_argument("--use_deepspeed", action='store_true', help="Use DeepSpeed")
    parser.add_argument('--check_prev', action='store_true', help='Check previous results')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    # Result file setup
    experiment_name = f'{args.model_name}_{args.prompt_culture}_{args.language}_{args.prompt_sample_count}_{args.entity_sample_count}'
    if args.special_name is not None:
        experiment_name = f'{experiment_name}_{args.special_name}'
    if args.entity_type is not None:
        experiment_name = f'{experiment_name}_{args.entity_type}'
    experiment_file = RESULT_DIR / f'{experiment_name}.json'
    logger.info(f'Saving results to {experiment_file}')

    # Check prev results
    if args.check_prev and experiment_file.exists():
        logger.info(f'{experiment_file} already exists. Skipping...')
        sys.exit(0)

    # Load model & use deepspeed
    model, tokenizer, model_config = load_matching_model(args.model_name, args.use_deepspeed)
    if args.use_deepspeed:
        model, _, _, _ = deepspeed.initialize(
            model = model, 
            model_parameters = model.parameters(),
            config = model_config['ds_config']
        )
    # Load dataset
    Dataset = ExtractiveQADataset(args.prompt_culture, model_config['model_type'], args.language, args.model_name)

    # Run
    all_results = {}
    all_entity_types = Dataset.get_all_entity_types()
    for entity_type in all_entity_types:
        if args.entity_type is not None and entity_type != args.entity_type:
            continue
        dataset = Dataset.create_dataset(entity_type, args.entity_sample_count, args.prompt_sample_count)
        logger.debug(f'Running inference for {entity_type} with {len(dataset)} samples')
        all_outputs = run_decoder(model, tokenizer, dataset, args.batch_size, args)
        # Run inference
        logger.debug(f'Inference done for {entity_type} with {len(all_outputs)} samples')
        all_results[entity_type] = {
            'dataset': dataset,
            'all_outputs': all_outputs
        }
    # Save
    write_json(all_results, experiment_file)
    
        