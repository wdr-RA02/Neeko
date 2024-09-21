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


def preprocess_data(
    prompt_template: Template,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    data_args: DataTrainingArguments,
    training_args: Seq2SeqTrainingArguments
) -> Dataset:
    column_names = list(dataset.column_names)

    # support question with a single answer or multiple answers
    def get_dialog(examples):
        for i in range(len(examples["prompt"])):
            if examples["prompt"][i] and examples["response"][i]:
                query, answer = examples["prompt"][i], examples["response"][i]
                query = query + "\n" + examples["query"][i] if examples["query"][i] else query
                prefix = examples["prefix"][i] if examples["prefix"][i] else ""
                dialog = prompt_template.get_dialog(query, answer, examples["history"][i], prefix)
                yield dialog

    def preprocess_supervised_dataset(examples):
        ROLE_PROFILE_MAPPING={
            "Beethoven": "",
            "Caesar": "",
            "Cleopatra": "",
            "Hermione": "",
            "Martin": "",
            "Newton": "",
            "Socrates": "",
            "Spartacus": "",
            "Voldemort": "",
        }
        for k in ROLE_PROFILE_MAPPING.keys():
            ROLE_PROFILE_MAPPING[k] = torch.load(os.path.join(data_args.embds_dir, k + ".pth"))
        # build inputs with format `<bos> X1 Y1 X2 Y2 ... <eos>` and labels with format `<ignore> Y1 <ignore> Y2 ... <eos>`
        model_inputs = {"input_ids": [], "labels": [], "role_embds": []}
        for i in range(len(examples["text"])):
            dialogs = examples["text"][i].split(examples["eot"][i])
            prompt_ids = tokenizer.encode(examples["prompt"][i], add_special_tokens=False)
            input_ids = [tokenizer.bos_token_id] + prompt_ids
            labels = [IGNORE_INDEX] * (len(prompt_ids) + 1)

            max_len = data_args.max_source_length + data_args.max_target_length
            inp_id_ls = []
            label_ls = []

            last_role_idx = 0
            for n, dialog in enumerate(dialogs):
                role_len = len(examples["role"][i])
                dialog_ids = tokenizer.encode(dialog, add_special_tokens=False)
                if examples["role"][i] == dialog[:role_len]:
                    # Is Label
                    # rec last idx with role
                    last_role_idx = n
                    len_role_action = len(examples["role"][i] + ' (speaking): ')
                    role_action = dialog[:len_role_action]
                    content = dialog[len_role_action:]
                    content_ids = tokenizer.encode(content, add_special_tokens=False)
                    role_action_ids = tokenizer.encode(role_action, add_special_tokens=False)
                    inp_id_ls.append(role_action_ids + [tokenizer.bos_token_id] + content_ids + [tokenizer.eos_token_id])
                    label_ls.append([IGNORE_INDEX] * (len(role_action_ids) + 1) + content_ids + [tokenizer.eos_token_id])
                else:
                    # Is Input
                    inp_id_ls.append(dialog_ids)
                    label_ls.append([IGNORE_INDEX] * len(dialog_ids))

            # 从末尾开始往上append对话，直到达到最大长度或已经append完
            accum_len = len(prompt_ids)
            inp_id_ls = inp_id_ls[:last_role_idx+1]
            label_ls = label_ls[:last_role_idx+1]
            dial_input = []
            dial_label = []
            while (len(inp_id_ls)>0 and len(label_ls)>0) and accum_len <= max_len:
                cur_len = len(inp_id_ls[-1])
                dial_input = inp_id_ls.pop() + dial_input
                dial_label = label_ls.pop() + dial_label
                accum_len +=  cur_len
            # if len(input_ids) > data_args.max_source_length + data_args.max_target_length:
            #     input_ids = input_ids[:data_args.max_source_length + data_args.max_target_length]
            # if len(labels) > data_args.max_source_length + data_args.max_target_length:
            #     labels = labels[:data_args.max_source_length + data_args.max_target_length]
            # 去除None
            input_ids += dial_input
            labels += dial_label

            input_ids = [n for n in input_ids if n is not None]
            # 确保input_ids以</s>结尾
            
            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)
            model_inputs["role_embds"].append(ROLE_PROFILE_MAPPING[examples["role"][i]])

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