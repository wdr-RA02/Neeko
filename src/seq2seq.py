import os
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

from transformers.trainer import PredictionOutput
from transformers.tokenization_utils import PreTrainedTokenizer
import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from .peft_trainer import PeftTrainer
from .utils import get_logger, IGNORE_INDEX

import torch

logger = get_logger(__name__)


@dataclass
class ComputeMetrics:
    r"""
    Wraps the tokenizer into metric functions, used in Seq2SeqPeftTrainer.
    """

    tokenizer: PreTrainedTokenizer

    def __call__(self, eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
        r"""
        Uses the model predictions to compute metrics.
        """
        preds, labels = eval_preds
        score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}

        for pred, label in zip(preds, labels):
            # pred_pad_len, label_pad_len = np.sum(pred == IGNORE_INDEX), np.sum(label == IGNORE_INDEX)
            # pred = pred[len(label) - label_pad_len : len(pred) - pred_pad_len] # remove prompts
            # label = label[:len(label) - label_pad_len]

            # 从label中获取!=-100的下标值
            pred = pred[np.nonzero(label != IGNORE_INDEX)]
            pred_parts = np.split(pred, np.nonzero(pred==self.tokenizer.eos_token_id)[0])

            label = label[np.nonzero(label != IGNORE_INDEX)]
            label_parts = np.split(label, np.where(label==2)[0])
            
            hypo = [self.tokenizer.decode(s, skip_special_tokens=True) for s in pred_parts]
            ref = [self.tokenizer.decode(s, skip_special_tokens=True) for s in label_parts]

            hypothesis = list(jieba.cut(" ".join(hypo)))
            reference = list(jieba.cut(" ".join(ref)))
            if len(" ".join(hypothesis).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        return {k: float(np.mean(v)) for k, v in score_dict.items()}


class Seq2SeqPeftTrainer(PeftTrainer):
    r"""
    Inherits PeftTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def save_predictions(
            self,
            predict_results: PredictionOutput
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")
        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for pred, label in zip(predict_results.predictions, predict_results.label_ids):
                pred = pred[np.nonzero(label != IGNORE_INDEX)]
                pred_parts = np.split(pred, np.nonzero(pred==self.tokenizer.eos_token_id)[0])

                label = label[np.nonzero(label != IGNORE_INDEX)]
                label_parts = np.split(label, np.where(label==2)[0])
                
                pred_sents = [self.tokenizer.decode(s, skip_special_tokens=True) for s in pred_parts]
                label_sents = [self.tokenizer.decode(s, skip_special_tokens=True) for s in label_parts]

                res.extend(json.dumps({"label": l, "predict": p}, ensure_ascii=False)\
                            for l, p in zip(pred_sents, label_sents))

            writer.write("\n".join(res))


    def compute_loss(self, model, inputs, return_outputs=False):
        if self.finetuning_args.finetuning_type != "moelora":
            return super().compute_loss(model, inputs, return_outputs)
        
        # add aux_loss in model list
        unwrapped_model = self.accelerator.unwrap_model(model)
        output = super().compute_loss(model, inputs, True)
        try:
            # num_moe = unwrapped_model.active_peft_config.num_moe
            aux_losses = unwrapped_model.base_model.aux_losses
        except AttributeError:
            print("aux_losses does not exist")
        except Exception as e:
            print("other exception ", e)
        else:
            all_loss = None
            if len(aux_losses) > 0:
                all_loss = torch.stack(aux_losses[-1], dim=-1)
                all_loss = torch.mean(all_loss, dim=-1)

        total_loss = output[0]
        if isinstance(all_loss, torch.Tensor) and isinstance(output[0], torch.Tensor):
            total_loss = output[0] + all_loss
        
        if isinstance(output[1], dict):
            output[1]["loss"] = total_loss
        else:
            output[1][0] = total_loss
        
        return (total_loss, output[1]) if return_outputs else total_loss