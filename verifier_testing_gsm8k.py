import os
import time
import math
import re
import json
import jsonlines
import itertools
import argparse
import torch
import torch.nn as nn
from transformers import (
        GPT2Tokenizer,
        GPT2TokenizerFast,
        AutoTokenizer, 
        AutoModelForCausalLM,
        BertTokenizer,
        AutoModelForMaskedLM,
        DebertaV2Tokenizer, 
        DebertaV2ForMaskedLM,
        DebertaV2ForTokenClassification,
        )
import pytorch_lightning as pl
from verifier_data_model import GPT2VerifierDataModel, VerifierPredictDataModel
from bert_verifier_data_model import BertVerifierDataModel, BertVerifierDataModel_new
from base_trainer import BaseTrainer
from base_model import BaseModel
from base_data_model import BaseDataModel
from verifier_modeling_gsm8k import GPT2ModelForVerifier
from bert_verifier_modeling_gsm8k import BertModelForVerifier

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def test(_args):
    hf_model = DebertaV2ForTokenClassification.from_pretrained(_args.model_path)
    tokenizer = DebertaV2Tokenizer.from_pretrained(_args.model_path, use_fast=True)
    verifier_head = None
    with open(_args.args_path, "r") as rf:
        args = json.load(rf)
    print("args = ", args)
    model = BertModelForVerifier(args, model=hf_model, tokenizer=tokenizer, verifier_head=verifier_head)
    state_dict = torch.load(_args.ckpt_path)["state_dict"]
    model.load_state_dict(state_dict)
    print(model)
    # 确保模型处于评估模式
    model.eval()

    with open(_args.data_path, "r") as rf:
        lines = rf.readlines()

    with open(_args.save_path, "w") as wf:
        for i, line in enumerate(lines):
            batch = [json.loads(line.strip())]
            data_dict = BertVerifierDataModel_new.collate_fn(batch, args, tokenizer)

            # 提取必要的输入数据
            input_ids = data_dict['input_ids']
            print(tokenizer.batch_decode(input_ids))
            attention_mask = data_dict['attention_mask']
            token_type_ids = data_dict['token_type_ids']
            verifier_labels = data_dict['verifier_labels']

            # 检查是否使用GPU
            if torch.cuda.is_available():
                model = model.cuda()
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()
                verifier_labels = verifier_labels.cuda()

            # 进行预测
            with torch.no_grad():
                # 调用模型的forward方法进行预测
                score = model(input_ids, attention_mask, token_type_ids,verifier_labels=verifier_labels)

            # 处理预测结果
            # 你可能需要根据你的模型和任务对这些输出进行适当的处理
            print("预测结果：",score)
            
            test_prob = score[1].tolist()[0][0]
            # 定义原始的batch数据和预测得分
            # 为batch中的每个字典添加pred_score字段
            for item in batch:
                item['pred_score'] = test_prob
                wf.write(json.dumps(item) + '\n')


import math
import re

def cal_MSE(args):
    with open(args.save_path, "r") as rf:
        lines = rf.readlines()

    sum_float, sum_float2, cnt = 0, 0, 0

    for line in lines:
        # 使用正则表达式找到所有的浮点数
        numbers = re.findall(r"[\d.]+e?[-+]?\d*", line)

        # 确保找到了预测值和真实值
        if len(numbers) >= 2:
            # 根据给定格式，假设第二个数字是我们需要的预测值，最后一个数字是真实值
            pred = float(numbers[4])  # 第二个数字作为预测值
            gt = float(numbers[-1])   # 最后一个数字作为真实值

            # 累加绝对差值和平方差
            sum_float += abs(pred - gt)
            sum_float2 += (pred - gt) ** 2
            cnt += 1

    # 计算MSE和RMSE
    MSE = sum_float2 / cnt if cnt > 0 else 0
    RMSE = math.sqrt(MSE) if cnt > 0 else 0

    print("MSE = ", MSE)
    print("RMSE = ", RMSE)

    return MSE, RMSE


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./verifier_outputs_test/microsoft-deberta-v3-large-prm800_con-01-29_19-32-03/hf_pretrained_epoch4_step39745")
    parser.add_argument("--args_path", type=str, default="./verifier_outputs_test/microsoft-deberta-v3-large-prm800_con-01-29_19-32-03/args.json")
    parser.add_argument("--ckpt_path", type=str, default="./verifier_outputs_test/microsoft-deberta-v3-large-prm800_con-01-29_19-32-03/last.ckpt")
    parser.add_argument("--save_path", type=str, default="./gsm8k_MCTS/02042024-230709-data/algo_output/jsonls/all284_gsm8k_predicted_mse.jsonl")
    # parser.add_argument("--data_path", type=str, default="./data/prm800/test_3cls.jsonl")
    parser.add_argument("--data_path", type=str, default="./gsm8k_MCTS/02042024-230709-data/algo_output/jsonls/all284_gsm8k_results.jsonl")
    args, left_argv = parser.parse_known_args()

    # test(args)
    # cal_MSE(args)

    model_path = "./verifier_outputs_test/microsoft-deberta-v3-large-prm800_con-01-29_19-32-03/hf_pretrained_epoch4_step39745"
    args_path = "./verifier_outputs_test/microsoft-deberta-v3-large-prm800_con-01-29_19-32-03/args.json"

    ckpt_path = "./verifier_outputs_test/microsoft-deberta-v3-large-prm800_con-01-29_19-32-03/last.ckpt"
    hf_model = DebertaV2ForTokenClassification.from_pretrained(model_path)
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_path, use_fast=True)
    verifier_head = None
    with open(args_path, "r") as rf:
        args = json.load(rf)
    model = BertModelForVerifier(args, model=hf_model, tokenizer=tokenizer, verifier_head=verifier_head)
    state_dict = torch.load(ckpt_path)["state_dict"]
    model.load_state_dict(state_dict)

    model.eval()
    print(model)
    
