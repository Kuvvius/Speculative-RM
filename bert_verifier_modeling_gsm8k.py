import os
import jsonlines
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import pytorch_lightning as pl
from bert_modeling_base import BertBaseModel
from calculator import batch_calculator_sample as sample


@torch.no_grad()
def gather_together(data):
    dist.barrier()

    world_size = dist.get_world_size()
    gather_data = [None for _ in range(world_size)]
    dist.all_gather_object(gather_data, data)

    return gather_data

def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)

class BertModelForVerifier(BertBaseModel):
    """
    initiates a PyTorch Lightning Bert-like base model for training Verifier, defines training and evaluation steps
    """
    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add Bert specific args
        Returns:
            parent_parser
        """
        parser = parent_parser.add_argument_group('BertModelForVerifier')
        parser.add_argument('--verifier_head', default=None, type=str, help="load a saved verifier head model")
        parser.add_argument('--verifier_loss', default="MSE", help="acceptable loss: [MSE, BCE]")
        parser.add_argument('--membank', default=0, type=int)

        return parent_parser

    def __init__(self, args, model=None, tokenizer=None, verifier_head=None):
        super().__init__(args, model, tokenizer)
        self.verifier_head = verifier_head
        self.verifier_idx = self.tokenizer.convert_tokens_to_ids("[VERIFIER]")
        if self.hparams.verifier_loss == "BCE":
            assert self.model.num_labels == 1

        # if args.get("membank", -1)>0:
        #     self.true_membank = torch.zeros(0, 1024)
        #     self.true_membank_ptr = 0
        #     self.false_membank = torch.zeros(0, 1024)
        #     self.false_membank_ptr = 0
        #     self.membank_size = args.get("membank", -1)

    @torch.no_grad()
    def get_negative_feats(self, labels, num_negative):
        labels = labels.cpu()
        negative_list = []
        if self.true_membank_ptr < num_negative:
            num_negative = self.true_membank_ptr
        if self.false_membank_ptr < num_negative:
            num_negative = self.false_membank_ptr

        for l in labels:
            if l == 0:
                perm_indx = torch.randperm(self.true_membank_ptr)
                negative_sample = self.true_membank[perm_indx[:num_negative]]
            elif l == 1:
                perm_indx = torch.randperm(self.false_membank_ptr)
                negative_sample = self.false_membank[perm_indx[:num_negative]]
            negative_list.append(negative_sample)
        
        return torch.stack(negative_list, dim=0)

    @torch.no_grad()
    def get_positive_feats(self, labels):
        labels = labels.cpu()
        positive_list = []
        for l in labels:
            if l == 1:
                positive_sample = torch.mean(self.true_membank, dim=0)
            elif l == 0:
                positive_sample = torch.mean(self.false_membank, dim=0)
            positive_list.append(positive_sample)
        
        return torch.stack(positive_list, dim=0)

    def cal_contrast_loss(self, features, labels):
        self.update_membank(features, labels)
        negative_feat = self.get_negative_feats(labels, 20)
        positive_feat = self.get_positive_feats(labels)
        all_feat = torch.cat((positive_feat.unsqueeze(dim=1), negative_feat), dim=1).cuda()
        seg_logits = torch.cosine_similarity(
            l2_normalize(features.unsqueeze(1)), l2_normalize(all_feat), dim=2
        )
        con_loss = F.cross_entropy(
            seg_logits, torch.zeros(features.shape[0]).long().cuda()
        )
        return con_loss
    
    @torch.no_grad()
    def update_membank(self, features, labels):
        true_feats = features[labels.to(bool)]
        false_feats = features[(1-labels).to(bool)]

        true_feats = true_feats.detach().clone().cpu()
        true_gathered_list = gather_together(true_feats)
        true_feats = torch.cat(true_gathered_list, dim=0)
        self.true_membank = torch.cat((self.true_membank, true_feats), dim=0)
        if self.true_membank.shape[0] >= self.membank_size:
            self.true_membank = self.true_membank[-self.membank_size:, :]
            self.true_membank_ptr = self.membank_size
        else:
            self.true_membank_ptr = self.true_membank.shape[0]  # move pointer

        false_feats = false_feats.detach().clone().cpu()
        false_gathered_list = gather_together(false_feats)
        false_feats = torch.cat(false_gathered_list, dim=0)
        self.false_membank = torch.cat((self.false_membank, false_feats), dim=0)
        if self.false_membank.shape[0] >= self.membank_size:
            self.false_membank = self.false_membank[-self.membank_size:, :]
            self.false_membank_ptr = self.membank_size
        else:
            self.false_membank_ptr = self.false_membank.shape[0]  # move pointer

    def get_inputs(self, batch):
        inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'token_type_ids': batch['token_type_ids'],
        }
        if self.hparams.verifier_loss == "BCE" and self.hparams.model_type == "deberta":
            inputs['final_token_idx'] = batch['final_token_idx']
        if 'verifier_labels' in batch:
            inputs['verifier_labels'] = batch['verifier_labels']

        return inputs

    def forward(self, input_ids, attention_mask, token_type_ids, verifier_labels=None, final_token_idx=None):
        """ forward step """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )

        if self.hparams.verifier_loss == "MSE":
            verifier_logits = output.logits[:, 0, self.verifier_idx]  # Expected shape = (bs, )
            verifier_predictions = self.verifier_head(verifier_logits.unsqueeze(-1)).squeeze(-1)  # Expected shape = (bs, )

            if verifier_labels is not None:
                loss_fct = nn.MSELoss()
                verifier_loss = loss_fct(verifier_predictions.view(-1), verifier_labels.view(-1))
        
        elif self.hparams.verifier_loss == "BCE":
            if self.hparams.model_type == "deberta":
                verifier_logits = output.logits.squeeze(-1)
                verifier_logits = torch.gather(verifier_logits, 1, final_token_idx)  # 这里是取了每组token的最后一个token的值 Expected shape = (bs, num_labels)
            else:
                verifier_logits = output.logits[:, 0]  # Expected shape = (bs, num_labels)
    
            if verifier_labels is not None:
                loss_fct = nn.BCEWithLogitsLoss()
                verifier_loss = loss_fct(verifier_logits.view(-1), verifier_labels.view(-1))
            verifier_predictions = torch.sigmoid(verifier_logits)
        
        elif self.hparams.verifier_loss == "LineMSE":
            # print("I am here LineMSE")
            # verifier_logits = output.logits[:, 0]
            # verifier_predictions = torch.sigmoid(verifier_logits)
            # if verifier_labels is not None:
            #     loss_fct = nn.MSELoss()
            #     verifier_loss = loss_fct(verifier_predictions.view(-1), verifier_labels.view(-1))

            verifier_logits = output.logits[:, 0]
            verifier_predictions = torch.sigmoid(verifier_logits)
            if verifier_labels is not None:
                with torch.no_grad():
                    positive_indx = []
                    negtive_indx = []
                    for i in range(0, len(verifier_labels), 2):
                        if verifier_labels[i] > verifier_labels[i+1]:
                            positive_indx.append(i)
                            negtive_indx.append(i+1)
                        else:
                            positive_indx.append(i+1)
                            negtive_indx.append(i)
                    positive_indx = torch.tensor(positive_indx).to(verifier_labels.device)
                    negtive_indx = torch.tensor(negtive_indx).to(verifier_labels.device)
                positive_logits = verifier_logits[positive_indx]
                negtive_logits = verifier_logits[negtive_indx]
                verifier_loss = torch.mean(-torch.log(torch.sigmoid((positive_logits-negtive_logits))))
        
        elif self.hparams.verifier_loss == "LineMSECon":
            verifier_logits = output.logits[:, 0]
            verifier_predictions = torch.sigmoid(verifier_logits)
            if verifier_labels is not None:
                loss_fct = nn.MSELoss()
                verifier_loss = loss_fct(verifier_predictions.view(-1), verifier_labels.view(-1))
                con_loss = self.cal_contrast_loss(output.hidden_states[-1][:, 0], verifier_labels)
                verifier_loss += 0.1 * con_loss

        elif self.hparams.verifier_loss == "Compare":
            verifier_logits = output.logits[:, 0]
            verifier_predictions = torch.sigmoid(verifier_logits)
            positive_logits = verifier_logits[positive_indx]
            negtive_logits = verifier_logits[negtive_indx]
            verifier_loss = torch.mean(-torch.log(torch.sigmoid((positive_logits-negtive_logits))))

        if verifier_labels is not None:
            self.log("verifier_loss", verifier_loss.item(), prog_bar=True, logger=True, on_step=True, batch_size=input_ids.size(0))
            loss = verifier_loss 
        else:
            loss = None

        return loss, verifier_predictions

    def predict_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        batch_size = input_ids.size(0)
        inputs = self.get_inputs(batch)
        del inputs['verifier_labels']
        _, verifier_predictions = self(**inputs)

        verifier_file = os.path.join(self.hparams.data_dir, self.hparams.predict_data) + "_verifier_scored_" + str(self.global_rank)

        with jsonlines.open(verifier_file, 'a') as f:
            for idx in range(batch_size):
                f.write({"question": batch['question'][idx], "solution": batch['solution'][idx] ,"verifier_score": str(verifier_predictions[idx].item()),
                    "is_correct": batch['is_correct'][idx], "question_id": batch['question_id'][idx], "ground_truth": batch['ground_truth'][idx]})

    def save_hf_checkpoint(self) -> None:
        #  if self.trainer._accelerator_connector.cluster_environment.global_rank() == 0:
        """Save huggingface model checkpoint and tokenizer"""
        if self.global_rank == 0:
            save_path = os.path.join(
                self.trainer.checkpoint_callback.dirpath if self.trainer else self.hparams.save_dir,
                'hf_pretrained_epoch{}_step{}'.format(self.current_epoch, self.global_step))
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            if self.verifier_head:
                torch.save(self.verifier_head, os.path.join(save_path, "verifier_head.pth"))


