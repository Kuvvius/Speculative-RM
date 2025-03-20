import torch
import torch.nn as nn
from base_data_model import BaseDataModel, BaseDataset
from data_preprocess import DataProcessor
from typing import List, Union, Tuple, Optional, Dict, Callable


class BertVerifierDataModel_new(BaseDataModel):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)

    def get_examples(self, path, type):
        examples = DataProcessor._read_jsonl(path)
        print(f"{len(examples)} examples")

        return examples

    # @staticmethod
    def collate_fn(batch, args, tokenizer):
        # batch_data = {}
        # for key in batch[0]:
        #     batch_data[key] = [example[key] for example in batch]
        
        temp_batch_data = []
        for i in range(len(batch)):
            t_batch_data = {}
            t_batch_data["question"] = "[QUES]" + batch[i]["instruction"]
            for response in batch[i]["responses"]:
                t_batch_data["question"] += "\n" + response
            t_batch_data["solution"] = batch[i]["next_response"] + "<|endoftext|>"
            t_batch_data["label"] = batch[i]["label"]
            temp_batch_data.append(t_batch_data)
        
        batch_data = {}
        for key in temp_batch_data[0]:
            batch_data[key] = [example[key] for example in temp_batch_data]

        inputs_encoding = tokenizer(
            batch_data['question'], 
            batch_data['solution'], 
            add_special_tokens=True, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )

        final_token_idx = inputs_encoding.attention_mask.sum(-1).view(-1, 1) - 1

        return dict(**batch_data, **inputs_encoding, verifier_labels=torch.FloatTensor(batch_data['label']), final_token_idx=final_token_idx)


class BertVerifierDataModel_compare(BaseDataModel):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)

    def get_examples(self, path, type):
        examples = DataProcessor._read_jsonl(path)
        print(f"{len(examples)} examples")

        return examples

    @staticmethod
    def collate_fn(batch, args, tokenizer):
        """
            [
                {
                    "question": "[QUES]Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage.  If she works 10 hours every day for 5 days, how much money does she make?", 
                    "next_response": "How many hours did Tina work every day?<|endoftext|>", 
                    "label": 0.5918266666580805
                }, 
                {
                    "question": "[QUES]Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage.  If she works 10 hours every day for 5 days, how much money does she make?", 
                    "next_response": "How much money does she make in 8 hours?<|endoftext|>", 
                    "label": 0.6099630889565345
                }
            ]
        """
        new_batch = []
        for b in batch:
            new_batch.append(b[0])
            new_batch.append(b[1])
        batch_data = {}
        for key in new_batch[0]:
            batch_data[key] = [example[key] for example in new_batch]

        batch_data['solution'] = batch_data["next_response"]
        inputs_encoding = tokenizer(
            batch_data['question'], 
            batch_data['solution'], 
            add_special_tokens=True, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )

        final_token_idx = inputs_encoding.attention_mask.sum(-1).view(-1, 1) - 1
        verifier_labels = torch.FloatTensor(batch_data["label"])

        return dict(
            **batch_data, 
            **inputs_encoding, 
            verifier_labels=verifier_labels, 
            final_token_idx=final_token_idx
        )

class BertVerifierDataModel(BaseDataModel):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)

    def get_examples(self, path, type):
        examples = DataProcessor._read_jsonl(path)
        print(f"{len(examples)} examples")

        return examples

    @staticmethod
    def collate_fn(batch, args, tokenizer):
        batch_data = {}
        for key in batch[0]:
            batch_data[key] = [example[key] for example in batch]

        inputs_encoding = tokenizer(
            batch_data['question'], 
            batch_data['solution'], 
            add_special_tokens=True, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )

        final_token_idx = inputs_encoding.attention_mask.sum(-1).view(-1, 1) - 1

        return dict(
            **batch_data, 
            **inputs_encoding, 
            verifier_labels=torch.FloatTensor(batch_data['is_correct']), 
            final_token_idx=final_token_idx
        )

if __name__ == '__main__':
    import argparse
    import pytorch_lightning as pl
    from transformers import BertTokenizer, AutoModelForMaskedLM
    from base_model import BaseModel
    from base_trainer import BaseTrainer
    from bert_verifier_modeling_gsm8k import BertModelForVerifier
    import transformers
    transformers.logging.set_verbosity_error()

    total_parser = argparse.ArgumentParser()
    # * data preprocessing args
    total_parser = BertVerifierDataModel.add_data_specific_args(total_parser)
    # * training args
    total_parser = BaseTrainer.add_trainer_specific_args(total_parser)
    # * model specific args
    total_parser = BaseModel.add_model_specific_args(total_parser)
    # * Bert specific args
    total_parser = BertModelForVerifier.add_model_specific_args(total_parser)

    args = total_parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.add_tokens(['[QUES]', '[ANS]', '[THOUGHT]', '[VERIFIER]'])

    bert = AutoModelForMaskedLM.from_pretrained(args.model_name)
    if bert.config.vocab_size < len(tokenizer):
        bert.resize_token_embeddings(new_num_tokens=len(tokenizer))
    verifier_head = nn.Linear(1, 1, bias=True)
    model = BertModelForVerifier(args, bert, tokenizer, verifier_head)

    verifier_data_model = BertVerifierDataModel(args, tokenizer)
    train_dataloader = verifier_data_model.train_dataloader()
    #  val_dataloader = verifier_data_model.val_dataloader()
    trainer = BaseTrainer(args, model)
    trainer.train(verifier_data_model)



