import pandas as pd
import random
from bs4 import BeautifulSoup
import json
import os
from tqdm import tqdm
import utils
import settings
from fuzzywuzzy import fuzz
import numpy as np
from os.path import join
from collections import defaultdict as dd
import re


def prepare_train_test_data_for_bert():
    re_html = re.compile('<[^>]+>')
    datas = []
    pid_to_source_titles = dd(list)
    papers_train = json.load(open('data/paper_source_trace_train_ans.json', 'r'))
    for p in papers_train:
        pid = p["_id"]
        for ref in p["refs_trace"]:
            pid_to_source_titles[pid].append(ref["title"].lower())

    # pids_train = {p["_id"] for p in papers_train}
    pids_train = {p["_id"]: p["title"] for p in papers_train}

    in_dir = join(settings.DATA_TRACE_DIR, "paper-xml")
    files = []
    for f in os.listdir(in_dir):
        if f.endswith(".xml"):
            files.append(f)

    files = sorted(files)
    for file in tqdm(files):
        f = open(join(in_dir, file), encoding='utf-8')
        cur_pid = file.split(".")[0]
        if cur_pid not in pids_train:
            continue

        title = pids_train[cur_pid]
        xml = f.read()
        bs = BeautifulSoup(xml, "xml")

        source_titles = pid_to_source_titles[cur_pid]
        if len(source_titles) == 0:
            continue
        # print("=cur_pid=", cur_pid)
        references = bs.find_all("biblStruct")
        bid_to_title = {}
        n_refs = 0
        bid2realTitle = {}
        for ref in references:
            if "xml:id" not in ref.attrs:
                continue
            bid = ref.attrs["xml:id"]
            # if ref.analytic is None:
            #     continue
            # if ref.analytic.title is None:
            #     continue
            if ref.analytic is not None and ref.analytic.title is not None:
                bid2realTitle[bid] = ref.analytic.title.text
            bid_to_title[bid] = ref.title.text.lower()
            b_idx = int(bid[1:]) + 1
            if b_idx > n_refs:
                n_refs = b_idx

        flag = False
        cur_pos_bib = []
        bid2titles = {}
        for bid in bid_to_title:
            cur_ref_title = bid_to_title[bid]
            for label_title in source_titles:
                if fuzz.ratio(cur_ref_title, label_title) >= 80:
                    flag = True
                    if bid in cur_pos_bib:
                        continue
                    cur_pos_bib.append(bid)
                    bid2titles[bid] = [title, label_title]

        cur_neg_bib = sorted(list(set(bid_to_title.keys()) - set(cur_pos_bib)))
        if not flag:
            continue
        if len(cur_pos_bib) == 0 or len(cur_neg_bib) == 0:
            continue
        bib_to_contexts = utils.find_bib_context(xml)

        n_pos = len(cur_pos_bib)
        n_neg = n_pos * 4
        random.seed(42)
        cur_neg_bib_sample = random.sample(list(cur_neg_bib), min(n_neg, len(cur_neg_bib)))
        for bib in cur_pos_bib:
            prefix = bid2realTitle.get(bib, "NO")
            cur_context = " ".join(bib_to_contexts[bib])
            if cur_context.strip() == "":
                continue
            datas.append([title + "[SEP]" + prefix + " | " + cur_context, 1])

        for bib in cur_neg_bib_sample:
            prefix = bid2realTitle.get(bib, "NO")
            cur_context = " ".join(bib_to_contexts[bib])
            if cur_context.strip() == "":
                continue
            datas.append([title + "[SEP]" + prefix + " | " + cur_context, 0])
    # random.seed(42)
    # random.shuffle(datas)
    df = pd.DataFrame(datas, columns=['text', 'label'])
    df.to_csv("data/train.csv", index=False)


prepare_train_test_data_for_bert()