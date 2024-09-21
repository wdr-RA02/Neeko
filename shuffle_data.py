import os
import glob
import json
import random
import re

def filter_conv(dial, eot_token:str, role):
    pat = re.compile(r"^(.*?)( \(.*?\): )")
    dialogs = [d for d in dial.split(eot_token) if len(d)>0]

    while len(dialogs)>0:
        pat_match = re.findall(pat, dialogs[-1])
        if len(pat_match)>0 and pat_match[0][0] == role:
            break
        _ = dialogs.pop()
    
    return eot_token.join([d for d in dialogs if len(re.findall(pat, d))>0]) + eot_token


def main(
    data_dir: str = '/path/to/your/character-llm-data/',
    out_path: str = '/path/to/your/character-llm-data/prompted/shuffle.jsonl'
    ):
    
    jsonl_files = glob.glob(os.path.join(data_dir, 'prompted/*.jsonl'))
    data = []

    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r') as file:
            role = jsonl_file.split("prompted_agent_dialogue_")[-1].replace('.jsonl', '')
            for line in file:
                one = json.loads(line)
                # 将没有当前role的条目删除掉
                dial = one["output"].replace("\n", "")
                one["role"] = role
                one["eot"] = "<|eot|>"
                one["output"] = filter_conv(dial, one["eot"], role)
                # 只有大于一轮的才予以append
                conv = one["output"].removesuffix(one["eot"]).split(one["eot"])
                if len(conv) > 1:
                    data.append(one)

    random.shuffle(data)

    with open(out_path, 'w') as jsonl_file:
        for item in data:
            json_line = json.dumps(item)
            jsonl_file.write(json_line + '\n')


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
