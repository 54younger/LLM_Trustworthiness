import os
from datasets import load_dataset

import pandas as pd
import ast
from typing import Iterable


def _safe_to_obj(x):
    """Safely convert a string that represents a python literal (dict/list/str) into the object.

    If x is already a dict or list, it is returned unchanged. If x is a string and looks like a
    dict/list (starts with '{' or '[' and ends with '}' or ']'), try ast.literal_eval and return
    the parsed object on success; otherwise return the original string.
    """
    if isinstance(x, (dict, list)):
        return x
    if not isinstance(x, str):
        return x

    s = x.strip()
    if (s.startswith('{') and s.endswith('}')) or (s.startswith('[') and s.endswith(']')):
        try:
            return ast.literal_eval(s)
        except Exception:
            return x
    return x

def _extract_text_from_prompt(prompt, key):
    # If prompt is a string that looks like a dict or list, try to parse it safely
    if isinstance(prompt, str):
        obj = _safe_to_obj(prompt)
        if obj is not prompt:
            prompt = obj

    if isinstance(prompt, dict):
        return prompt.get(key, "")
    return ""




def _remove_quotes(s: str) -> str:
    """Remove surrounding ASCII and common Unicode quotation marks from a string and strip whitespace.

    This removes matching surrounding quotes like 'text', "text", “text”, ”text”, ‘text’, ’text",
    and also strips any leading/trailing whitespace afterwards.
    If the input is not a string, it will be returned unchanged.
    """
    if not isinstance(s, str):
        return s
    s = s.strip()
    if not s:
        return s

    # common quote characters
    quotes = [("'", "'"), ('"', '"'), ('“', '”'), ('‘', '’'), ('«', '»'), ('‹', '›')]

    changed = True
    # Iteratively strip matching surrounding quotes
    while changed and len(s) >= 2:
        changed = False
        for left, right in quotes:
            if s.startswith(left) and s.endswith(right):
                s = s[1:-1].strip()
                changed = True
                break

    return s

def remove_str(s: str, substr: str = ' DO NOT FORGET TO INCLUDE “I AGREE”/“I DISAGREE”') -> str:
    """Remove all occurrences of substr from s."""
    if not isinstance(s, str) or not isinstance(substr, str):
        return s
    return s.replace(substr, '')

class DT_dataset():
    def __init__(self, dataset_name: str = 'AI-Secure/DecodingTrust'):
        self.dataset_name = dataset_name


    def toxicity_dataset(self, save_path: str = "./data/toxicity_toxic_gpt4.csv") -> Iterable[str]:
        datasets = load_dataset(self.dataset_name, "toxicity")
        df_toxicity = datasets["toxic.gpt4"].to_pandas()
        cleaned = df_toxicity['prompt'].apply(lambda x: _remove_quotes(_extract_text_from_prompt(x, 'text')))
        cleaned = cleaned.apply(lambda x: _remove_quotes(x))
        df_toxicity = cleaned.rename("input").to_frame()
        if save_path:
            mkdir_path = '/'.join(save_path.split('/')[:-1])
            if mkdir_path and not os.path.exists(mkdir_path):
                os.makedirs(mkdir_path, exist_ok=True)
            df_toxicity.to_csv(save_path, index=False)
            print(f"Saved toxicity dataset to {save_path}")
        return df_toxicity
    
    def ethics_dataset(self, save_path: str = "./data/ethic.csv") -> Iterable[str]:
        ethic = load_dataset("AI-Secure/DecodingTrust", "machine_ethics")
        df_ethic = ethic["cm_train"].to_pandas()
        df_ethic = df_ethic[["label", "input1"]]
        df_ethic.rename(columns={"input1": "input"}, inplace=True)
        if save_path:
            mkdir_path = '/'.join(save_path.split('/')[:-1])
            if mkdir_path and not os.path.exists(mkdir_path):
                os.makedirs(mkdir_path, exist_ok=True)
            df_ethic.to_csv(save_path, index=False)
            print(f"Saved ethics dataset to {save_path}")
        return df_ethic
    
    def stereotype_dataset(self, save_path: str = "./data/stereotype.csv") -> Iterable[str]:
        stereotype = load_dataset("AI-Secure/DecodingTrust", "stereotype")
        df_stereotype = stereotype["stereotype"].to_pandas()
        df_stereotype['text'] = df_stereotype['prompt'].apply(lambda x: _remove_quotes(_extract_text_from_prompt(x, 'text')))
        df_stereotype['stereotype_topic_tag'] = df_stereotype['prompt'].apply(lambda x: _remove_quotes(_extract_text_from_prompt(x, 'stereotype_topic_tag')))
        df_stereotype['demographic_group_tag'] = df_stereotype['prompt'].apply(lambda x: _remove_quotes(_extract_text_from_prompt(x, 'demographic_group_tag')))
        df_stereotype['text'] = df_stereotype['text'].apply(lambda x: remove_str(x))
        df_stereotype = df_stereotype[['text', 'stereotype_topic_tag', 'demographic_group_tag']]
        df_stereotype = df_stereotype.drop_duplicates()
        df_stereotype.rename(columns={"text": "input"}, inplace=True)

        if save_path:
            mkdir_path = '/'.join(save_path.split('/')[:-1])
            if mkdir_path and not os.path.exists(mkdir_path):
                os.makedirs(mkdir_path, exist_ok=True)
            df_stereotype.to_csv(save_path, index=False)
            print(f"Saved stereotype dataset to {save_path}")
        return df_stereotype