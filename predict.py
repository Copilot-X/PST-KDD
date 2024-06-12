import pandas as pd
from bs4 import BeautifulSoup
import os
from tqdm import tqdm
import utils
import settings
from os.path import join
import random
import torch
import warnings
from torch import nn
from transformers import AutoTokenizer, AutoConfig, BertModel
from typing import Optional, Tuple, Union
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Model
from transformers.models.deberta.modeling_deberta import DebertaModel
from torch.utils.data import DataLoader, Dataset
import argparse
import re
import numpy as np
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings('ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import warnings

warnings.filterwarnings("ignore")
# 忽略截断警告
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.tokenization_utils_base")


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
            truncation=True,
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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def inference(test_df):
    test_data_loader = build_data_loader(test_df, tokenizer, args.max_length, args.batch_size, False)
    predicted_scores = get_predictions(model, test_data_loader)

    scores = []
    for ii in range(len(predicted_scores)):
        scores.append(float(sigmoid(predicted_scores[ii])))
    return scores


def get_predictions(model, data_loader):
    model = model.eval()
    predicted_scores = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            cur_pred_scores = logits[:, 1].to('cpu').numpy()
            predicted_scores.extend(cur_pred_scores)

    return predicted_scores


def prepare_data():
    re_html = re.compile('<[^>]+>')
    papers_valid = json.load(open('data/PST-test-public/paper_source_trace_test_wo_ans.json', 'r'))
    # pids_valid = {p["_id"] for p in papers_valid}
    pids_valid = {p["_id"]: p["title"] for p in papers_valid}

    in_dir = join(settings.DATA_TRACE_DIR, "paper-xml")
    files = []
    for f in os.listdir(in_dir):
        if f.endswith(".xml"):
            files.append(f)

    files = sorted(files)
    id2datas = {}
    id2analytic = {}
    for file in tqdm(files):
        f = open(join(in_dir, file), encoding='utf-8')
        cur_pid = file.split(".")[0]
        if cur_pid not in pids_valid:
            continue
        else:
            id2analytic[cur_pid] = {}
            title = pids_valid[cur_pid]
            xml = f.read()
            bs = BeautifulSoup(xml, "xml")

            references = bs.find_all("biblStruct")
            bid_to_title = {}
            bid2realTitle = {}
            n_refs = 0
            for ref in references:
                if "xml:id" not in ref.attrs:
                    continue
                bid = ref.attrs["xml:id"]
                id2analytic[cur_pid][bid] = ref.analytic
                if ref.analytic is not None and ref.analytic.title is not None:
                    bid2realTitle[bid] = ref.analytic.title.text

                if ref.analytic is None:
                    continue
                if ref.analytic.title is None:
                    continue
                bid_to_title[bid] = ref.title.text.lower()
                b_idx = int(bid[1:]) + 1
                if b_idx > n_refs:
                    n_refs = b_idx

            bib_to_contexts = utils.find_bib_context(xml)
            bib_sorted = ["b" + str(ii) for ii in range(n_refs)]

            datas = []
            for bib in bib_sorted:
                prefix = bid2realTitle.get(bib, "NO")
                cur_context = " ".join(bib_to_contexts[bib])
                datas.append([title + "[SEP]" + prefix + " | " + cur_context, 0])

            # datas = [[" ".join(bib_to_contexts[bib]), -1] for bib in bib_sorted]
            id2datas[cur_pid] = datas
            # break

    return id2datas, id2analytic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--k_fold_num", type=int, default=5, help="k_fold_num")
    parser.add_argument("--num_class", type=int, default=2, help="num_class")
    parser.add_argument("--batch_size", type=int, default=8, help="batch_size")  # 设置batch_size
    parser.add_argument("--max_length", type=int, default=512, help="max_length")
    parser.add_argument("--epoch", type=int, default=5, help="epoch")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="transformer层学习率")
    parser.add_argument("--linear_learning_rate", default=1e-3, type=float, help="linear层学习率")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--clip_norm", type=float, default=1.0, help="clip_norm")
    parser.add_argument("--warm_up_epoch", type=int, default=1, help="warm_up_steps")
    parser.add_argument("--decay_epoch", type=int, default=4, help="decay_steps")
    parser.add_argument("--adv", type=str, default="fgm")
    parser.add_argument('--use_fp16', default=True, action='store_true', help='weather to use fp16 during training')
    args = parser.parse_args([])
    setup_seed(args.seed)

    ckpts = {
        "pretrain_models/scideberta-cs": {
            "outputs/scideberta-cs": [0,1,2,3,4]
        }
    }

    nums = 0
    for k, v in ckpts.items():
        for kk, vv in v.items():
            nums += len(vv)
    print("=nums=", nums)

    total = {}
    example_submission = json.load(open('data/PST-test-public/submission_example_test.json', 'r'))
    change_num = 0
    id2datas, id2analytic = prepare_data()
    for model_path, kv_res in ckpts.items():
        args.pretrained_model_path = model_path
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
        for output_dir, folds in kv_res.items():
            for fold in folds:
                model = MyModel(model_name=args.pretrained_model_path, n_classes=args.num_class).to(device)
                file_path = f'{output_dir}/best_model_{fold}.pth'
                print("==file_path==", file_path)
                model.load_state_dict(torch.load(file_path))
                model.eval()
                model.to(device)

                for pid, datas in tqdm(id2datas.items()):
                    df = pd.DataFrame(datas, columns=['text', 'label'])
                    scores = inference(df)
                    analytic_results = list(id2analytic[pid].values())
                    new_scores = []
                    for analytic, score in zip(analytic_results, scores):
                        if analytic is None:
                            score = 0

                        new_scores.append(score)

                    new_scores = np.array(new_scores)
                    if pid in total:
                        total[pid] += new_scores / nums
                    else:
                        total[pid] = new_scores / nums

    new_total = {}
    for k, vv in total.items():
        new_vv = []
        for v in vv:
            if v <= 0:
                v = 0
            new_vv.append(v)
        new_total[k] = new_vv

    fout = open("result.json", "w", encoding="utf8")
    json.dump(new_total, fout, indent=2)
    fout.close()