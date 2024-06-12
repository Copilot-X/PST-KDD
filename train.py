import datetime
import torch
from sklearn.model_selection import StratifiedKFold
import warnings
from torch import nn
from ai.optim import AdamWGC as AdamW
from transformers import AutoConfig, set_seed
from transformers import AutoTokenizer, BertModel, get_linear_schedule_with_warmup
from typing import Optional, Tuple, Union
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Model
from transformers.models.deberta.modeling_deberta import DebertaModel
from torch.utils.data import DataLoader, Dataset
import argparse
from sklearn.metrics import f1_score, classification_report
import pandas as pd
import json
import os
from tqdm import tqdm
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings('ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        """
    item 为数据索引，迭代取第item条数据
    """
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        #         print(encoding['input_ids'])
        return {
            'texts': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def build_data_loader(df, tokenizer, max_len, batch_size, is_shuffle=False):
    ds = MyDataset(texts=df['text'].values, labels=df['label'].values, tokenizer=tokenizer, max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=is_shuffle, pin_memory=True)


class MyModel(nn.Module):
    def __init__(self, model_name, n_classes):
        super().__init__()
        bert_config = AutoConfig.from_pretrained(model_name)
        # bert_config.update({'output_hidden_states': True})
        self.dropout = nn.Dropout(0.1)
        self.num_layer = 3
        if "scideberta" in model_name and "scideberta-full" not in model_name:
            self.model = DebertaModel.from_pretrained(model_name)
        elif "deberta" in model_name:
            self.model = DebertaV2Model.from_pretrained(model_name)
        else:
            self.model = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(bert_config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask, token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )

        last_hidden_state = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        logits = self.classifier(mean_embeddings)
        # print("==logits==", logits.shape)
        return logits


class FocalLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class FGM(object):
    def __init__(self, model, emb_name, epsilon=1.0):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD:
    def __init__(self, model, eps=1., alpha=0.3):
        self.model = (
            model.module if hasattr(model, "module") else model
        )
        self.eps = eps
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, emb_name='word_embeddings', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.eps:
            r = self.eps * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class WarmUp_LinearDecay:
    def __init__(self, optimizer: AdamW, init_rate, warm_up_epoch, decay_epoch, train_data_length,
                 min_lr_rate=5e-6):
        self.optimizer = optimizer
        self.init_rate = init_rate
        self.epoch_step = train_data_length / args.batch_size
        self.warm_up_steps = self.epoch_step * warm_up_epoch
        self.decay_steps = self.epoch_step * decay_epoch
        self.min_lr_rate = min_lr_rate
        self.optimizer_step = 0
        self.all_steps = args.epoch * (train_data_length / args.batch_size)

    def step(self):
        self.optimizer_step += 1
        if self.optimizer_step <= self.warm_up_steps:
            rate = (self.optimizer_step / self.warm_up_steps) * self.init_rate
        elif self.warm_up_steps < self.optimizer_step <= self.decay_steps:
            rate = self.init_rate
        else:
            rate = (1.0 - ((self.optimizer_step - self.decay_steps) / (
                    self.all_steps - self.decay_steps))) * self.init_rate
            if rate < self.min_lr_rate:
                rate = self.min_lr_rate
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self.optimizer.step()


def run():
    save_dir = f"outputs/{model_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fout = open(f"{save_dir}/train_log.txt", 'a', encoding='utf-8')
    now_time = str(datetime.datetime.now()).split(".")[0]
    fout.write(json.dumps({"datetime": now_time, "model_name": model_name}, ensure_ascii=False) + "\n")
    data_df = pd.read_csv('data/train.csv')

    all_labels = sorted(set(data_df['label'].tolist()))
    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label = {v: k for k, v in label2id.items()}

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # 使用GroupKFold进行交叉验证
    for idx, (train_index, val_index) in enumerate(kf.split(data_df, data_df['label'])):
        if idx == 0:
            continue
        print(f"===第{idx + 1}===epoch" + "\n")
        df_train = data_df.loc[train_index]
        df_val = data_df.loc[val_index]
        train_data_loader = build_data_loader(df_train, tokenizer, args.max_length, args.batch_size, True)
        dev_data_loader = build_data_loader(df_val, tokenizer, args.max_length, args.batch_size, False)
        model = MyModel(args.pretrained_model_path, args.num_class)
        model.to('cuda')
        # 优化器
        no_decay = ["bias", "LayerNorm.weight"]

        bert_param_optimizer = list(model.named_parameters())

        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay,
             'lr': args.learning_rate},
            {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0,
             'lr': args.learning_rate}
        ]
        t_total = len(df_train) * args.epoch
        optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.learning_rate)
        # schedule = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)
        schedule = WarmUp_LinearDecay(
            optimizer=optimizer,
            init_rate=args.learning_rate,
            warm_up_epoch=args.warm_up_epoch,
            decay_epoch=args.decay_epoch,
            train_data_length=len(df_train),
        )
        weight = [1., 3]
        loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float()).to(device)
        if args.adv == "fgm":
            fgm = FGM(model, emb_name='word_embeddings.')
            print("--启动fgm对抗训练--")
        elif args.adv == "pgd":
            pgd = PGD(model=model)
            pgd_k = 3
            print("--启动pgd对抗训练--")

        best_f1 = 0
        for epoch in tqdm(range(args.epoch)):
            model.train()
            tk = tqdm(train_data_loader, total=len(train_data_loader), position=0, leave=True)
            for step_idx, item in enumerate(tk):
                input_ids = item["input_ids"].to(device)
                attention_mask = item["attention_mask"].to(device)
                labels = item["labels"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                # print("==logits==", logits)
                # print("==labels==", labels)
                loss = loss_func(logits, labels)
                # loss = loss_func(logits.view(-1, args.num_class), labels.view(-1))

                loss = loss.float().mean().type_as(loss)
                loss.backward()

                if args.adv == "fgm":
                    # FGM
                    fgm.attack()  # 在embedding上添加对抗扰动
                    logits_adv = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss_adv = loss_func(logits_adv, labels.to(device))
                    # loss_adv = loss_func(logits_adv.view(-1, args.num_class), labels.view(-1))
                    loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    fgm.restore()  # 恢复embedding参数

                elif args.adv == "pgd":
                    pgd.backup_grad()
                    for _t in range(pgd_k):
                        pgd.attack(is_first_attack=(_t == 0))
                        if _t != pgd_k - 1:
                            model.zero_grad()
                        else:
                            pgd.restore_grad()

                        logits_adv = model(input_ids=input_ids, attention_mask=attention_mask)
                        loss_adv = loss_func(logits_adv, labels.to(device))
                        loss_adv.backward()

                    pgd.restore()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)
                schedule.step()
                optimizer.zero_grad()

                tk.set_postfix(loss=loss.item())

            # 每轮epoch在验证集上计算分数
            eval_f1 = evaluate(dev_data_loader, model, loss_func)
            fout.write(json.dumps({"epoch": epoch, "eval_f1": eval_f1}, ensure_ascii=False) + "\n")

            fout.flush()
            print(f"{epoch}  the eval_f1 is {eval_f1}, saving model !!")

            if eval_f1 > best_f1:
                best_f1 = eval_f1
                torch.save(model.state_dict(), f=f"{save_dir}/best_model_{idx}.pth")

        # 清空显存
        torch.cuda.empty_cache()
        del model
        fout.write("\n\n")
        # break


def evaluate(dev_data_loader, model, criterion):
    eval_loss = 0.0
    eval_steps = 0
    pres, trues = [], []
    model.eval()
    for item in dev_data_loader:
        with torch.no_grad():
            input_ids = item["input_ids"].to(device)
            attention_mask = item["attention_mask"].to(device)
            labels = item["labels"].to(device)
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        loss = criterion(logits, labels)
        eval_loss += loss.item()
        eval_steps += 1
        _, pre = torch.max(logits, dim=1)
        pre = pre.cpu().numpy().tolist()
        true = labels.cpu().numpy().tolist()
        pres.extend(pre)
        trues.extend(true)

    report = classification_report(y_true=trues, y_pred=pres)
    f1 = f1_score(trues, pres, average='macro')
    print(report)
    return f1


def get_probs(model, data_loader):
    model = model.eval()
    prediction_probs = []
    with torch.no_grad():
        for d in tqdm(data_loader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = outputs.softmax(1)
            prediction_probs.extend(probs)
    prediction_probs = torch.stack(prediction_probs).cpu()
    return prediction_probs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--k_fold_num", type=int, default=5, help="k_fold_num")
    parser.add_argument("--num_class", type=int, default=2, help="num_class")
    parser.add_argument("--batch_size", type=int, default=8, help="batch_size")  # 设置batch_size
    parser.add_argument("--max_length", type=int, default=512, help="max_length")
    parser.add_argument("--epoch", type=int, default=5, help="epoch")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="transformer层学习率")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--pretrained_model_path", type=str, default="pretrain_models/scideberta-cs",
                        help="pretrained_model_path")
    parser.add_argument("--clip_norm", type=float, default=1.0, help="clip_norm")
    parser.add_argument("--warm_up_epoch", type=int, default=1, help="warm_up_steps")
    parser.add_argument("--decay_epoch", type=int, default=8, help="decay_steps")
    parser.add_argument("--adv", type=str, default="pgd", help="fgm or pgd")
    args = parser.parse_args()
    set_seed(args.seed)
    print("===seed===", args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    model_name = args.pretrained_model_path.split('/')[-1]
    # 训练
    run()
