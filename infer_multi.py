from transformers import AutoModelForCausalLM, GenerationConfig, LlamaTokenizer, AutoTokenizer, AutoModel
from typing import Union, List
import json
import torch
from tqdm import tqdm
from moelora import PeftModel
import argparse
import os
import csv
from src.generation import generate_prompt, evaluate

from sentence_transformers import SentenceTransformer

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
for k in ROLE_PROFILE_MAPPING.keys():
    ROLE_PROFILE_MAPPING[k] = torch.load(os.path.join("./data/role_embds", k + ".pth")).unsqueeze(0)


def read_profile(path):
    with open(path, 'r', encoding='utf-8') as fp:
        text = fp.read().strip()
    parts = text.split('\n\n')
    assert parts[0].startswith('# '), parts[0]
    agent_profile = []
    for p in parts[1:]:
        agent_profile.append(p.strip())
    return agent_profile[0]

ROLE_PROFILE_TEXT={}
for k in ROLE_PROFILE_MAPPING.keys():
    profile = read_profile(os.path.join("./data/seed_data/profiles", "wiki_" + k + ".txt"))
    ROLE_PROFILE_TEXT[profile] = k

def parse_arguments():
    parser = argparse.ArgumentParser(description="Infer")

    parser.add_argument(
        "--infer_path", type=str, default="./data/seed_data/trans_dialogues.jsonl", help="path of all jsons."
    )
    parser.add_argument(
        "--save_path", type=str, default='./infer/multi.jsonl'
    )
    parser.add_argument(
        "--LLM", type=str, required=True,
    )
    parser.add_argument(
        "--lora_path", type=str, default="/home/tongxuluo/Neeko/ckpt/neeko/wo_caesar/20240203140411"
    )
    parser.add_argument(
        "--resume_id", type=int, default=0
    )
    args = parser.parse_args()

    return args


def main(args):
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    tokenizer = LlamaTokenizer.from_pretrained(args.LLM, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    
    print("Loading model from ", args.LLM)
    model = AutoModelForCausalLM.from_pretrained(
            args.LLM,
            # load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        ).eval() # fix zwq
    
    print("Loading PEFT from ", args.lora_path)
    model = PeftModel.from_pretrained(
            model,
            args.lora_path,
            torch_dtype=torch.float16,
            device_map="auto"
        ).eval()
    
    ckpt = "dunzhang/stella_en_400M_v5"
    s_bert = SentenceTransformer(ckpt, trust_remote_code=True, device=model.device)

    with torch.no_grad():
        sent = list(ROLE_PROFILE_TEXT.keys())
        profile_embds = s_bert.encode(sent, prompt="s2s_prompt")

    result = eval_multi(model, tokenizer, args, profile_embds=profile_embds, s_bert=s_bert)

    with open(args.save_path, "w") as f:
        f.write(result)

def find_next_embd(model, character:str, profile_embds, s_bert):
    if not hasattr(model, "global_role_embd"):
        return 
    
    with torch.no_grad():
        sent_embd = s_bert.encode(f"I want you to act like {character}")
    
    scores = s_bert.similarity(sent_embd, profile_embds).squeeze(0)
    index = scores.argmax().item()
    embd_key = list(ROLE_PROFILE_MAPPING.keys())[index]

    print("Most similar role_embd: ", embd_key)
    model.global_role_embd.clear()
    model.global_role_embd.append(ROLE_PROFILE_MAPPING[embd_key].to(model.device))

    if hasattr(model, "role_ids"):
        model.role_ids.clear()
        model.role_ids.append(torch.LongTensor([index]).to(device=model.device))


def eval_multi(model, tokenizer, args, profile_embds, s_bert):
    # select the most near character
    eval_results = []
    
    # multi = "for_multiturn_" if args.multi_turns else ""
    # infer_path = os.path.join(args.infer_path, f"generated_agent_interview_{multi}{character}.json")

    with open(args.infer_path, 'r') as file:
        test_set = []
        for line in file:
            json_obj = json.loads(line)
            test_set.append(json_obj)
    
    for i, one in enumerate(tqdm(test_set,  
                                 desc=f"Evaluating multi")):
        if i < args.resume_id - 1:
            continue

        inputs = []
        for j in range(one["max_turns"]):
            inputs.append({
                "role": one["content"][2 * j]["turn_content"][0]["role"],
                "action": one["content"][2 * j]["turn_content"][0]["action"],
                "content": one["content"][2 * j]["turn_content"][0]["content"],
            })

            next_role = one["content"][2 * j + 1]["turn_content"][0]["role"]
            find_next_embd(model, next_role, profile_embds, s_bert)

            res = evaluate(tokenizer=tokenizer, model=model, character=next_role, inputs=inputs)
            one["content"][2 * j + 1]["turn_content"][0]["content"] = res
            inputs.append({
                "role": one["content"][2 * j + 1]["turn_content"][0]["role"],
                "action": one["content"][2 * j + 1]["turn_content"][0]["action"],
                "content": one["content"][2 * j + 1]["turn_content"][0]["content"],
            })
        
        eval_results.append(inputs)
            
    result = [json.dumps(res, ensure_ascii=False) for res in eval_results]
    result = "\n".join(result)

    return result

if __name__ == "__main__":
    args = parse_arguments()
    main(args=args)
