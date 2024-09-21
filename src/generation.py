import torch
import transformers 

from typing import List, Dict, Optional, Union
from transformers import GenerationConfig

_SYS = "[SYS]"
_SYS_END = "[/SYS]"
_SEP = "\n###"

SYS_PROMPT_TEMPLATE = """I want you to act like {character}. I want you to respond and answer like {character}, using the tone, manner and vocabulary {character} would use. You must know all of the knowledge of {character}. Reply must be brief and concise.

The status of you is as follows:
Location: Coffee Shop - Afternoon
Status: {character} is casually chatting with a man from the 21st century. {character} fully trusts the man who engage in conversation and shares everything {character} knows without reservation.

The interactions are as follows:"""

HISTORY_TEMPLATE = f"{_SEP} " + "{role} {action}:\n{content}{eos}"

def generate_prompt(character: str, inputs, eos_token:str):
    gpt_prompt = f"{_SYS} {SYS_PROMPT_TEMPLATE.format(character=character)} {_SYS_END}\n "
    # append history
    history = ""
    for dialog in inputs:
        history += HISTORY_TEMPLATE.format(role=dialog["role"], action=dialog["action"], content=dialog["content"], eos=eos_token)
    
    user = " " + HISTORY_TEMPLATE.format(role=character, action="(speaking)", content="", eos="")
    gpt_prompt += (history + user)

    return gpt_prompt

def evaluate(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    character: str,
    inputs: List[Dict[str,str]] = None,
    do_sample: bool = True,
    temperature: float=0.2,
    top_p: float=0.9,
    top_k: float=50,
    num_beams: float=1,
    max_new_tokens: int=512,
    **kwargs,
):
    prompt = generate_prompt(character, inputs, tokenizer.eos_token)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    input_len = input_ids.shape[1]

    generation_config = GenerationConfig(
        do_sample=do_sample,
        num_beams=num_beams,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
        **kwargs,
    )
    if do_sample:
        generation_config.top_k = top_k
        generation_config.top_p = top_p
        generation_config.temperature = temperature
    
    output = generate(model, input_ids, attention_mask, generation_config)
    s = output.sequences[0]
    output_seq = tokenizer.decode(s[input_len:].cpu(), skip_special_tokens=True)

    return output_seq

def generate(
    model: transformers.PreTrainedModel, 
    input_ids: torch.Tensor, 
    attention_mask: Optional[torch.Tensor] = None, 
    generation_config: Optional[transformers.GenerationConfig] = None, 
    return_dict_in_generate: bool = True,
    output_scores: bool = True,
    **kwargs
) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
    input_len = input_ids.shape[1]
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    # map to corr device
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=return_dict_in_generate,
            output_scores=output_scores,
        )
    
    if not return_dict_in_generate:
        output = output[input_len:].cpu()

    return output
