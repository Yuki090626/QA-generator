import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
from tqdm.notebook import tqdm, trange
from transformers import BertJapaneseTokenizer
from .eval import eval_vae
from .trainer import VAETrainer
from .utils import batch_to_device, get_harv_data_loader, get_squad_data_loader
from .models import DiscreteVAE, return_mask_lengths
import collections
import json
from transformers import BertJapaneseTokenizer
from tqdm.notebook import tqdm
from .squad_utils import evaluate, write_predictions
from .eval import Result, to_string


def QA_generation(filename, modelname, savename):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1004, type=int)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--train_dir', default='../data/kosodate/train.json')
    parser.add_argument('--dev_dir', default='../data/kosodate/test.json')

    parser.add_argument("--max_c_len", default=384, type=int, help="max context length")
    parser.add_argument("--max_q_len", default=64, type=int, help="max query length")

    parser.add_argument("--model_dir", default="../save/vae-checkpoint-jp", type=str)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--lr", default=1e-3, type=float, help="lr")
    parser.add_argument("--batch_size", default=32, type=int, help="batch_size")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay")
    parser.add_argument("--clip", default=5.0, type=float, help="max grad norm")

    parser.add_argument("--bert_model", default='cl-tohoku/bert-base-japanese-whole-word-masking', type=str)
    parser.add_argument('--enc_nhidden', type=int, default=300)
    parser.add_argument('--enc_nlayers', type=int, default=1)
    parser.add_argument('--enc_dropout', type=float, default=0.2)
    parser.add_argument('--dec_a_nhidden', type=int, default=300)
    parser.add_argument('--dec_a_nlayers', type=int, default=1)
    parser.add_argument('--dec_a_dropout', type=float, default=0.2)
    parser.add_argument('--dec_q_nhidden', type=int, default=900)
    parser.add_argument('--dec_q_nlayers', type=int, default=2)
    parser.add_argument('--dec_q_dropout', type=float, default=0.3)
    parser.add_argument('--nzqdim', type=int, default=50)
    parser.add_argument('--nza', type=int, default=20)
    parser.add_argument('--nzadim', type=int, default=10)
    parser.add_argument('--lambda_kl', type=float, default=0.1)
    parser.add_argument('--lambda_info', type=float, default=1.0)

    args = parser.parse_args([])


    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    tokenizer = BertJapaneseTokenizer.from_pretrained(args.bert_model)

    args.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    trainer = VAETrainer(args)
    vae = DiscreteVAE(args).to(args.device)
    vae.load_state_dict(torch.load(modelname)['state_dict'])
    trainer.vae = vae


    def document_preprocess(documents, device):
        res = []
        for d in documents:
            res.append(['[CLS]'])
            res[-1].extend(d)
            res[-1].append('[SEP]')
            while len(res[-1]) < 256 + 2:
                res[-1].append('[PAD]')
            res[-1] = tokenizer.convert_tokens_to_ids(res[-1])
        return torch.tensor(res, dtype=torch.long, device=device)


    def docs_to_qas(c_ids, trainer, tokenizer):

        res = []

        tmp = [[] for i in range(3)]

        for n in range(3):
            batch_prior_q_ids, batch_prior_start, batch_prior_end, prior_z_prob = trainer.generate_prior(c_ids)

            for i in range(len(c_ids)):
                dic = {}
                dic['context'] = to_string(c_ids[i], tokenizer).replace('[UNK]', '')
                dic['question'] = to_string(batch_prior_q_ids[i], tokenizer)
                dic['answer'] = to_string(c_ids[i][batch_prior_start[i]:(batch_prior_end[i] + 1)], tokenizer).replace('[UNK]', '')
                tmp[n].append(dic)

        for i in range(len(c_ids)):
            for n in range(3):
                res.append(tmp[n][i])

        return res


    def qa_generation(text_dir, save_dir, trainer, tokenizer, save_encoding='utf-8'):
        all_text = open(text_dir, 'r', encoding='utf-8').read()

        # 大区分
        text_group = all_text.split('\n\n')
        text_group_tokens = [tokenizer.tokenize(t) for t in text_group]

        # context_length <= 256 となるように分割する
        documents = []
        for t in text_group_tokens:
            if len(t) <= 256:
                documents.append(t)
            else:
                tmp = t[:]
                while len(tmp) > 256:
                    documents.append(tmp[:256])
                    # 半分ずらす
                    tmp = tmp[128:]
                if len(tmp) > 0:
                    documents.append(tmp)

        # ドキュメントの前処理
        document_ids = document_preprocess(documents, args.device)

        batch = []
        idx = 0
        while idx < len(document_ids):
            batch.append(document_ids[idx:idx + 32])
            idx += 32

        results = []
        answers = [] # 重複対応
        for data in batch:
            qacs = docs_to_qas(data, trainer, tokenizer)
            for qac in qacs:
                if qac['answer'][:20] not in answers:
                    answers.append(qac['answer'][:20])
                    results.append(qac)
            # results.extend(docs_to_qas(data, trainer, tokenizer))

        df = pd.DataFrame(results, columns=['question', 'answer'])
        df.to_csv(save_dir, encoding=save_encoding)

        return results

    results = qa_generation(filename, savename, trainer, tokenizer, save_encoding='cp932')

    return results