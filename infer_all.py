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
        "--infer_path", type=str, default="./data/seed_data/questions", help="path of all jsons."
    )
    parser.add_argument(
        "--save_path", type=str, default='./infer/'
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
    parser.add_argument(
        '--multi-turns', action='store_true', help='Enable multi-turns mode'
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

    for character in ROLE_PROFILE_MAPPING:
        result = eval_single(model, tokenizer, args, character=character, profile_embds=profile_embds, s_bert=s_bert)
        suffix = "multi.jsonl" if args.multi_turns else "single.json"

        with open(os.path.join(args.save_path, f"{character}_{suffix}"), "w") as f:
            f.write(result)
        
        print("*"*50)


def eval_single(model, tokenizer, args, character:str, profile_embds, s_bert):
    # select the most near character
    eval_results = []

    if hasattr(model, "global_role_embd"):
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
    
    multi = "for_multiturn_" if args.multi_turns else ""
    infer_path = os.path.join(args.infer_path, f"generated_agent_interview_{multi}{character}.json")
    with open(infer_path, 'r') as file:
        test_set = []
        if args.multi_turns:
            for line in file:
                json_obj = json.loads(line)
                test_set.append(json_obj)
        else:
            test_set = json.load(file)
    
    suffix = "multi" if args.multi_turns else "single"
    for i, one in enumerate(tqdm(test_set,  
                                 desc=f"Evaluating {character}/{suffix}")):
        if i < args.resume_id - 1:
            continue
        if args.multi_turns:
            inputs = []
            for j in range(one["max_turns"]):
                inputs.append({
                    "role": one["content"][2 * j]["turn_content"][0]["role"],
                    "action": one["content"][2 * j]["turn_content"][0]["action"],
                    "content": one["content"][2 * j]["turn_content"][0]["content"],
                })
                res = evaluate(tokenizer=tokenizer, model=model, character=character, inputs=inputs)
                one["content"][2 * j + 1]["turn_content"][0]["content"] = res
                inputs.append({
                    "role": one["content"][2 * j + 1]["turn_content"][0]["role"],
                    "action": one["content"][2 * j + 1]["turn_content"][0]["action"],
                    "content": one["content"][2 * j + 1]["turn_content"][0]["content"],
                })
            
            eval_results.append(one)
            
        else:
            outline = {
                "topic_id": one["topic_id"],
                "question": one["question"],
            }
            inputs=[{
                "role": "Man",
                "action": "(speaking)",
                "content": one["question"]
            }]
            res = evaluate(model=model, tokenizer=tokenizer, character=character, inputs=inputs)
            reply = {
                "role": character,
                "action": "(speaking)",
                "content": res,
            }
            outline["reply"] = reply    
            eval_results.append(outline)

    if args.multi_turns:
        result = [json.dumps(res, ensure_ascii=False) for res in eval_results]
        result = "\n".join(result)
    else:
        result = json.dumps(eval_results, ensure_ascii=False, indent=4)

    return result

if __name__ == "__main__":
    args = parse_arguments()
    main(args=args)
