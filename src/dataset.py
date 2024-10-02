import json
from abc import ABC, abstractmethod
import re
from datasets import Dataset
from transformers import PreTrainedTokenizer, Seq2SeqTrainingArguments
import torch
import os
from .template import Template
from .utils import DataTrainingArguments, IGNORE_INDEX
import pandas as pd
import csv
import re

ROLE_PROFILE_MAPPING={
    "Beethoven": "",
    "Cleopatra": "",
    "Hermione": "",
    "Martin": "",
    "Newton": "",
    "Socrates": "",
    "Spartacus": "",
    "Voldemort": "",
    "Caesar": "",
}


def preprocess_data(
    prompt_template: Template,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    data_args: DataTrainingArguments,
    training_args: Seq2SeqTrainingArguments
) -> Dataset:
    column_names = list(dataset.column_names)

    tmpl = "\n### {role_action}\n"
    _SYS = "[SYS]"
    _SYSEND = "[/SYS]"

    def preprocess_supervised_dataset(examples):
        for k in ROLE_PROFILE_MAPPING.keys():
            ROLE_PROFILE_MAPPING[k] = torch.load(os.path.join(data_args.embds_dir, k + ".pth"), map_location="cpu")
        
        pat = re.compile(r"^(.*?)( \(.*?\): )")
        # model_inputs = {"input_ids": [], "labels": [], "role_embds": []}
        model_inputs = {"input_ids": [], "labels": [], "role_embds": [], "role_ids": []}
        for i in range(len(examples["text"])):
            dialogs = examples["text"][i].split(examples["eot"][i])
            while len(dialogs)>0:
                pat_match = re.findall(pat, dialogs[-1])
                if len(pat_match)>0 and pat_match[0][0] == examples["role"][i]:
                    break
                _ = dialogs.pop()
            
            prompt = f"{_SYS} " + examples["prompt"][i].removesuffix("\n\n") + f" {_SYSEND}\n"
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            # <s> [SYS]background[/SYS]\n\n### Role (action):\n{query}\n\n### Roleb (actionb):\n{}
            input_ids = [tokenizer.bos_token_id] + prompt_ids
            labels = [IGNORE_INDEX] * (len(input_ids))

            max_len = data_args.max_source_length + data_args.max_target_length
            inp_id_ls = []
            label_ls = []

            for n, dialog in enumerate(dialogs):
                # match role and action with re
                pat_match = re.findall(pat, dialog)
                if len(pat_match) == 0 or len(dialog) == 0:
                    continue
                dial_role, dial_act = pat_match[0]

                # 先填上模板
                inp_id = tokenizer.encode(tmpl.format(role_action=dial_role+dial_act), add_special_tokens=False)
                label_id = [IGNORE_INDEX] * len(inp_id)
                # 冒号后面的是content
                len_role_action = len(pat_match[0][0])+len(pat_match[0][1])
                content_ids = tokenizer.encode(dialog[len_role_action:], add_special_tokens=False)
                # input_id = [tmpl]+[content]+[eos]
                inp_id += content_ids + [tokenizer.eos_token_id]
                
                if examples["role"][i] == dial_role:
                    # Is Label
                    # rec last idx with role
                    label_id += content_ids + [tokenizer.eos_token_id]
                else:
                    # Is Input
                    label_id += ([IGNORE_INDEX] * len(content_ids) + [tokenizer.eos_token_id])

                inp_id_ls.append(inp_id)
                label_ls.append(label_id)

            # 从末尾开始往上append对话，直到达到最大长度或已经append完
            accum_len = len(prompt_ids)
            dial_input = []
            dial_label = []
            while (len(inp_id_ls)>0 and len(label_ls)>0) and accum_len <= max_len:
                cur_len = len(inp_id_ls[-1])
                dial_input = inp_id_ls.pop() + dial_input
                dial_label = label_ls.pop() + dial_label
                accum_len +=  cur_len
            
            # 去除None
            input_ids += dial_input
            labels += dial_label

            input_ids = [n for n in input_ids if n is not None]
            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)
            model_inputs["role_embds"].append(ROLE_PROFILE_MAPPING[examples["role"][i]])

            roles = list(ROLE_PROFILE_MAPPING.keys())
            role_id = roles.index(examples["role"][i])
            model_inputs["role_ids"].append(role_id)

        return model_inputs

    def print_supervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(
            tokenizer.decode([d if d != IGNORE_INDEX else tokenizer.pad_token_id for d in example["labels"]],
                             skip_special_tokens=False)
        ))
        print("role_embds:\n{}".format(example["role_embds"]))
        print("role_ids:\n{}".format(example["role_ids"]))

    preprocess_function = preprocess_supervised_dataset

    with training_args.main_process_first(desc="dataset map pre-processing"):
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset"
        )
        print_supervised_dataset_example(dataset[0])

        return dataset


class LLaMaDataset(ABC):
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.dataset = self.__read_data_to_huggingface_dataset__(data_path)

    @abstractmethod
    def __read_data_to_huggingface_dataset__(self, data_path: str) -> Dataset:
        """
        Reading the data and preprocessing to the huggingface dataset with the column_names bellow:
        column_names = ["prefix", "prompt", "query", "response", "history"]
        :return: dataset: the huggingface Dataset
        """
        pass


class Character_LLM(LLaMaDataset):
    def __read_data_to_huggingface_dataset__(self, data_path: str) -> Dataset:

        column_names = ["prefix", "role", "prompt", "text", "eot"]
        dataset = []
        with open(data_path, 'r') as data:
            for line in data:
                one = json.loads(line)
                dataset.append({
                    "prefix": None,
                    "role": one["role"],
                    "prompt": one["prompt"],
                    "text": one["output"].replace("\n", ""),
                    "eot": one["eot"]
                })

        huggingface_data = {column_name: [] for column_name in column_names}

        for data_sample in dataset:
            for column_name in column_names:
                huggingface_data[column_name].append(data_sample[column_name])

        hf_dataset = Dataset.from_dict(huggingface_data)
        return hf_dataset


class Character_LLM_Single(LLaMaDataset):
    def __read_data_to_huggingface_dataset__(self, data_path: str) -> Dataset:
        column_names = ["prefix", "role", "prompt", "text", "eot"]
        dataset = []
        role = data_path.split("prompted_agent_dialogue_")[-1].rstrip('.jsonl')
        with open(data_path, 'r') as data:
            for line in data:
                one = json.loads(line)
                dataset.append({
                    "prefix": None,
                    "role": role,
                    "prompt": one["prompt"],
                    "text": one["output"].replace("\n", ""),
                    "eot": "<|eot|>"
                })

        huggingface_data = {column_name: [] for column_name in column_names}

        for data_sample in dataset:
            for column_name in column_names:
                huggingface_data[column_name].append(data_sample[column_name])

        hf_dataset = Dataset.from_dict(huggingface_data)
        return hf_dataset

STR_DATASET_MAP = {
    "character-llm": Character_LLM,
    "character-llm-single": Character_LLM_Single,
}