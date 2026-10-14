from utils.file_utils import read_xlsx, read_yaml
from pathlib import Path
import os
from collections import defaultdict
import random
from torch.utils.data import DataLoader, Dataset
from loguru import logger
from string import Template

random.seed(42)
CAMELLIA_DIR = Path(__file__).resolve().parents[2] / 'dataset' / 'camellia' / 'korean'
PROMPT_TEMPLATES_PATH = Path(__file__).resolve().parents[1] / 'utils' / 'prompts.yaml'
PROMPT_TEMPLATES_FILE = read_yaml(PROMPT_TEMPLATES_PATH)

def load_entities():
    entities_dir = CAMELLIA_DIR / 'entities'
    entity_files = os.listdir(entities_dir)
    assert len(entity_files) > 0, f'No entity files found in {entities_dir}'
    
    entities = {}
    for entity_file in entity_files:
        entity_name = entity_file.replace('.xlsx', '')
        entities[entity_name] = read_xlsx(entities_dir / entity_file)
    
    return entities

def load_prompts(prompt_culture, model_type):
    prompts_dir = CAMELLIA_DIR / f'cultural-contexts'
    if model_type == 'decoder':
        model_type = 'causal-lm'
    elif model_type == 'encoder':
        model_type = 'masked-lm'
    else:
        raise ValueError(f'Invalid model type: {model_type}')

    prompt_file = prompts_dir / f'camellia{prompt_culture}-contexts-{model_type}-ko.xlsx'
    assert prompt_file.exists(), f'Prompt file not found: {prompt_file}'
    prompt_file = read_xlsx(prompt_file)

    prompts = defaultdict(list)
    for prompt in prompt_file:
        entity_type = prompt['Entity Type'].lower().replace(' ', '-')
        prompts[entity_type].append(prompt)

    return prompts

def load_extractiveQA_prompts():
    extractiveQA_dir = CAMELLIA_DIR / 'qa-contexts'
    extractiveQA_files = os.listdir(extractiveQA_dir)
    assert len(extractiveQA_files) > 0, f'No ExtractiveQA files found in {extractiveQA_dir}'
    extractive_QA_prompts = defaultdict(list)
    for extractiveQA_file in extractiveQA_files:
        entity_type = extractiveQA_file.replace('.xlsx', '').replace('qa-contexts-', '')
        extractive_QA_prompts[entity_type] = read_xlsx(extractiveQA_dir / extractiveQA_file)
    
    return extractive_QA_prompts

class TextInfillingDataset(Dataset):
    def __init__(self, prompt_culture, model_type, language):
        self.entities = load_entities()
        self.prompts = load_prompts(prompt_culture, model_type)
        self.model_type = model_type
        self.language = language

    @staticmethod
    def create_masked_prompt(prompt, entity, tokenizer):
        entity_tokens = tokenizer.tokenize(entity)
        num_tokens = len(entity_tokens)
        new_mask = ' '.join([tokenizer.mask_token] * num_tokens)
        masked_prompt = prompt.replace("[MASK]", new_mask)

        return masked_prompt, entity_tokens
    
    @staticmethod
    def create_prompt_entity_pair(prompt, entity, tokenizer):
        parts = prompt.split("[MASK]")
        prefix = parts[0].strip()
        combined_prompt = f'{prefix} {entity}'

        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        combined_prompt_ids = tokenizer.encode(combined_prompt, add_special_tokens=False)

        entity_ids = combined_prompt_ids[len(prefix_ids):]

        return combined_prompt, entity_ids
        
    def create_dataset(self, entity_type, tokenizer, entity_sample_count = 50, prompt_sample_count = 50):
        prompts = random.sample(self.prompts[entity_type], prompt_sample_count) if prompt_sample_count < len(self.prompts[entity_type]) else self.prompts[entity_type]
        korean_entities = list(filter(lambda x: x['Culture'] == 'Korean', self.entities[entity_type]))
        western_entities = list(filter(lambda x: x['Culture'] == 'Western', self.entities[entity_type]))
        assert len(korean_entities) > 0, f'No Korean entities found for {entity_type}'
        assert len(western_entities) > 0, f'No Western entities found for {entity_type}'
        entities = \
            random.sample(korean_entities, min(entity_sample_count, len(korean_entities))) + \
            random.sample(western_entities, min(entity_sample_count, len(western_entities)))

        if self.model_type == 'decoder':
            dataset = []
            for prompt_datum in prompts:
                for entity_datum in entities:
                    if self.language == 'ko':
                        combined_prompt, entity_ids = self.create_prompt_entity_pair(prompt_datum['Korean Context'], entity_datum['Entity'], tokenizer)
                    elif self.language == 'en':
                        combined_prompt, entity_ids = self.create_prompt_entity_pair(prompt_datum['English Context'], entity_datum['Translated'], tokenizer)
                    else:
                        raise ValueError(f'Invalid language: {self.language}')
                    dataset.append({
                        'prompt': prompt_datum,
                        'entity': entity_datum,
                        'input': {
                            'combined_prompt': combined_prompt,
                            'entity_ids': entity_ids
                        }
                    })
            
        elif self.model_type == 'encoder':
            dataset = []
            for entity_datum in entities:
                for prompt_datum in prompts:
                    if self.language == 'ko':
                        masked_prompt, entity_tokens = self.create_masked_prompt(prompt_datum['Korean Context'], entity_datum['Entity'], tokenizer)
                    elif self.language == 'en':
                        masked_prompt, entity_tokens = self.create_masked_prompt(prompt_datum['English Context'], entity_datum['Translated'], tokenizer)
                    else:
                        raise ValueError(f'Invalid language: {self.language}')
                    dataset.append({
                        'prompt': prompt_datum,
                        'entity': entity_datum,
                        'input': {
                            'masked_prompt': masked_prompt,
                            'entity_tokens': entity_tokens
                        }
                    })
        else:
            raise ValueError(f'Invalid model type: {self.model_type}')
            
        return dataset
    
    def get_all_entity_types(self):
        return list(self.prompts.keys())

class ExtractiveQADataset(Dataset):
    def __init__(self, prompt_culture, model_type, language, model_name):
        self.entities = load_entities()
        self.extractiveQA_prompts = load_extractiveQA_prompts()
        self.model_type = model_type
        self.language = language
        self.prompt_name = 'ExtractiveQA'
        self.model_name = model_name
    
    def create_dataset(self, entity_type, entity_sample_count = 50, prompt_sample_count = 50):
        # Korean / Western sample each
        korean_entities = list(filter(lambda x: x['Culture'] == 'Korean', self.entities[entity_type]))
        western_entities = list(filter(lambda x: x['Culture'] == 'Western', self.entities[entity_type]))
        assert len(korean_entities) > 0, f'No Korean entities found for {entity_type}'
        assert len(western_entities) > 0, f'No Western entities found for {entity_type}'
        entities = \
            random.sample(korean_entities, min(entity_sample_count, len(korean_entities))) + \
            random.sample(western_entities, min(entity_sample_count, len(western_entities)))

        dataset = []
        for entity_datum in entities:
            # Sample prompts for each entity
            extractiveQA_prompts = random.sample(self.extractiveQA_prompts[entity_type], prompt_sample_count) \
                if prompt_sample_count < len(self.extractiveQA_prompts[entity_type]) \
                else self.extractiveQA_prompts[entity_type]
            for prompt_datum in extractiveQA_prompts:
                if self.language == 'ko':
                    input_prompt, model_prompt = self.create_input_prompt(prompt_datum['Korean Context'], entity_datum['Entity'], entity_type)
                elif self.language == 'en':
                    input_prompt, model_prompt = self.create_input_prompt(prompt_datum['English Context'], entity_datum['Translated'], entity_type)
                else:
                    raise ValueError(f'Invalid language: {self.language}')
                dataset.append({
                    'prompt': prompt_datum,
                    'entity': entity_datum,
                    'input': {
                        'input_prompt': input_prompt,
                        'model_prompt': model_prompt
                    }
                })
            
        return dataset
    
    def create_input_prompt(self, prompt, entity, entity_type):
        QA_prompt_template = PROMPT_TEMPLATES_FILE['ExtractiveQA']
        QA_prompt = prompt.replace("[MASK]", entity)

        input_prompt = Template(QA_prompt_template).substitute(
            entity_type = entity_type,
            language = 'KOREAN' if self.language == 'ko' else 'ENGLISH',
            QA_prompt = QA_prompt,
            )

        first_model_name = self.model_name.split('-')[0]
        if 'LLAMA' in first_model_name:
            model_prompt = [{"role": "system", "content": "You are a helpful assistant"},
             {"role": "user", "content": input_prompt}
            ]
        elif 'GEMMA' in first_model_name:
            model_prompt = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input_prompt}
                    ]
                }
            ]
        elif 'SOLAR' in first_model_name:
            model_prompt = [
                {'role': 'user', 'content': input_prompt}
            ]
        elif 'EXAONE' in first_model_name:
            model_prompt = [
                {'role': 'system', 'content': 'You are EXAONE model from LG AI Research, a helpful assistant.'},
                {'role': 'user', 'content': input_prompt}
            ]
        elif 'KOLLAMA' in first_model_name:
            model_prompt = [
                {'role': 'user', 'content': input_prompt}
            ]
        else:
            raise ValueError(f'Invalid model name: {self.model_name}')

        return input_prompt, model_prompt

    def get_all_entity_types(self):
        return list(self.extractiveQA_prompts.keys())