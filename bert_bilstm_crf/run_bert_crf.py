"""
token classification task （stable version 20200915）

this file works with ner_data_util.py

@author: youngf
@create time: 2020-09-11
@last update: 2020-09-15
"""

import os
import torch
import shutil
import logging
import argparse
import torch.optim as optim
from transformers import BertTokenizer, BertModel, BertConfig, BertTokenizerFast
import time
from bert_lstm_crf import BERT_LSTM_CRF
import numpy as np
import random
from metric import fmeasure_from_singlefile
from ner_data_util import NerTokenizer


# if running on cpu, comment this line
# if running on gpu, choose the wished gpu number
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 6'

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def evaluate(model, ner_tokenizer, device, eval_dataloader, args):
    """ evaluate base on cache file `eval_result.data` """
    if os.path.exists('eval_result.data'):
        os.remove('eval_result.data')

    if not args.do_train and args.checkpoint_name:
        logging.info('loading weights from {}'.format(args.checkpoint_name))
        model.load_state_dict(
            torch.load(os.path.join(args.dump_model_path, args.checkpoint_name), map_location=device, strict=False))

    logging.info('start to eval...')
    model.eval()
    with torch.no_grad():
        with open('eval_result.data', 'a', encoding='utf8') as fout:
            for batch_id, batch in enumerate(eval_dataloader):
                input_ids, input_masks, tag_ids = tuple([t.to(device) for t in batch])
                label_preds = model('inference', input_ids, input_masks=input_masks)

                current_batch_size = len(input_ids)
                for i in range(current_batch_size):
                    source_str = ner_tokenizer.ptm_tokenizer.convert_ids_to_tokens(input_ids[i])
                    source_tags = [ner_tokenizer.id2tag[x.item()] for x in tag_ids[i]]
                    pred_tags = [ner_tokenizer.id2tag[x] for x in label_preds[i]]

                    # automatically align the length according to pred_tags
                    for source_ch, source_tag, pred_tag in zip(source_str[1:], source_tags[1:], pred_tags[1:]):
                        fout.write('{} {} {}\n'.format(source_ch, source_tag, pred_tag))
                    fout.write('\n')

        acc, p, r, f1 = fmeasure_from_singlefile('eval_result.data', "BMES")
        logging.info('evaluation result: acc={:.4f}, precision={:.4f}, recall={:.4f}, f1={:.4f}'.format(acc, p, r, f1))
    return f1


def train(model, ner_tokenizer, device, train_data_loader, eval_dataloader, args):
    logging.info('start to train...')

    # count the number of model parameters; divide parameters into two groups
    bert_params = []
    other_params = []
    num_total_params = 0
    for k, v in model.named_parameters():
        num_total_params += v.numel()
        if k.startswith('bert'):
            bert_params.append(v)
        else:
            other_params.append(v)
    logging.info('total params: {}'.format(num_total_params))
    optimizer = optim.Adam([{"params": bert_params, "lr": args.bert_learning_rate},
                            {"params": other_params, "lr": args.other_learning_rate}])

    best_metric_score = 0
    for epoch in range(args.train_epochs):
        for batch_id, batch in enumerate(train_data_loader):
            model.train()
            model.zero_grad()
            input_ids, input_masks, tag_ids = tuple([t.to(device) for t in batch])
            loss = model('train', input_ids, tags=tag_ids,
                         input_masks=input_masks)  # logits: batch_size, length, num_tags

            if batch_id and batch_id % args.display_steps == 0:
                template = 'epochs={:2}/{:2}, batch={:3}/{:3}, loss={:.6f}'
                logging.info(template.format(epoch, args.train_epochs, batch_id, len(train_data_loader), loss))
            loss.backward()
            optimizer.step()

        logging.info('epoch evaluation:')
        metric_score = evaluate(model, ner_tokenizer, device, eval_dataloader, args)
        if args.dump_model and metric_score > best_metric_score:  # only dump model which gains better f1 scores
            best_metric_score = metric_score
            torch.save(model.state_dict(), os.path.join(args.dump_model_path, args.model_name + '-' + str(epoch)))
            logging.info(os.path.join(args.dump_model_path, args.model_name + '-' + str(epoch)) + ' dumped!')


def predict(model, ner_tokenizer, device, args):
    """ real time predicting your input in command line """
    if args.checkpoint_name:
        logging.info('loading weights from {}'.format(args.checkpoint_name))
        model.load_state_dict(
            torch.load(os.path.join(args.dump_model_path, args.checkpoint_name), map_location=device, strict=False))

    logging.info('preparation finished!')
    model.eval()
    with torch.no_grad():
        while True:
            query = input('input sentence:')
            if query != 'q' or query != 'quit':
                tokens = list(query.strip())
                input_ids, input_masks, token2char = ner_tokenizer.ch_seq_to_feature(tokens, [], 100)  # random number 100 can avoid output
                input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
                input_masks = torch.tensor([input_masks], dtype=torch.bool).to(device)

                pred = model('inference', input_ids, input_masks=input_masks)[0]
                token_tags = [ner_tokenizer.id2tag[tag_id] for tag_id in pred]  # n+2
                token2char = token2char[:len(token_tags)]

                # print("tokens:", ner_tokenizer.ptm_tokenizer.convert_ids_to_tokens(input_ids[0].tolist())[:len(pred)])
                # print("token tags:", token_tags)
                # print("token2char:", token2char)

                ch_tag = ner_tokenizer.token_tag_to_ch_tag(token_tags, token2char)  # cls and sep
                print(ch_tag)
                entity_list = ner_tokenizer.parse_entity_from_ch_tag(ch_tag)
                print(entity_list)
                if not entity_list:
                    print('no entity found.')
                else:
                    for s, e, c in entity_list:
                        print(query[s:e], c)
            else:
                print('system exited, see you again~~')
                break


def random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args, ner_tokenizer, pretrained_model):
    random_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info('device: {}'.format(device))
    model = BERT_LSTM_CRF(bert=pretrained_model,
                          bert_hidden_size=args.bert_hidden_size,
                          num_tags=len(ner_tokenizer.tag2id),
                          use_lstm=args.use_lstm,
                          lstm_hidden_size=args.lstm_hidden_size,
                          id2tag=ner_tokenizer.id2tag,
                          dropout_rate=args.dropout_rate).to(device)

    if args.do_train or args.do_eval:
        logging.info('loading eval corpus...')
        eval_dataloader = ner_tokenizer.build_dataloader(args.eval_file_path, args.batch_size, "sequential")
    if args.do_train:
        logging.info('loading train corpus...')
        train_dataloader = ner_tokenizer.build_dataloader(args.train_file_path, args.batch_size, "random")
        train(model, ner_tokenizer, device, train_dataloader, eval_dataloader, args)
    elif args.do_eval:
        evaluate(model, ner_tokenizer, device, eval_dataloader, args)
    elif args.do_predict:
        predict(model, ner_tokenizer, device, args)
    else:
        raise Exception("either `do_train` or `do_eval` or `do_predict` must be `True`")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--verbose_logging", default=False, action="store_true", help="output data examples")
    parser.add_argument("--do_train", default=False, action="store_true", help="train mode")
    parser.add_argument("--do_eval", default=False, action="store_true", help="eval mode")
    parser.add_argument("--do_predict", default=False, action="store_true", help="predict mode")
    parser.add_argument("--use_lstm", default=False, action="store_true", help="whether add lstm layers")
    parser.add_argument("--dump_model", default=False, action="store_true", help="whether dump model")

    parser.add_argument("--checkpoint_name", default="robert-lstm-crf-wwm-large-2", type=str, help="specify model name")
    parser.add_argument("--train_epochs", default=40, type=int, help="maximum epochs")
    parser.add_argument("--dataset", default="cluener", type=str, help="dataset name")
    parser.add_argument("--dump_model_path", default="models", type=str, help="directory")

    parser.add_argument("--model_card", default="hfl/chinese-roberta-wwm-ext-large", type=str)  # bert-base-chinese
    parser.add_argument("--model_name", default="roberta-wwm-ext-bilstm-crf", type=str)

    # model parameters
    parser.add_argument("--bert_hidden_size", default=1024, type=int)  # 768
    parser.add_argument("--lstm_hidden_size", default=128, type=int)
    parser.add_argument("--dropout_rate", default=0, type=float)
    parser.add_argument("--max_seq_len", default=60, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--seed", default=100, type=int)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--bert_learning_rate", default=1e-5, type=float)
    parser.add_argument("--other_learning_rate", default=0.01, type=float)

    pre_args = parser.parse_args()

    # prepare dependent parameters
    dataset_to_file_path = {
        "people": {
            "train": "../resource/people/train.data",
            "eval": "../resource/people/dev.data",
            "test": "../resource/people/test.data"
        },
        "alarm": {
            "train": "corpus/alarm/alarm.train.data",
            "eval": "corpus/alarm/alarm.test.data",
            "test": ""
        },
        "resume": {
            "train": "corpus/resume/train.char.bmes",
            "eval": "corpus/resume/dev.char.bmes",
            "test": "corpus/resume/test.char.bmes"
        },
        "cluener": {
            "train": "../resource/cluener/train.bmes.data",
            "eval": "../resource/cluener/dev.bmes.data",
            "test": "../resource/cluener/test.bmes.data"
        }
    }

    if pre_args.dataset == "people":
        scheme = "BMES"
        categories = ["ORG", "PER", "LOC"]
        display_steps = 100
    elif pre_args.dataset == "alarm":
        tag2id = {'O': 0, 'B-T': 1, 'M-T': 2, 'E-T': 3, 'S-T': 4}
        display_steps = 5
    elif pre_args.dataset == 'resume':
        tag2id = {'O': 0, 'B-NAME': 1, 'M-NAME': 2, 'E-NAME': 3, 'S-NAME': 4, 'B-TITLE': 5, 'M-TITLE': 6, 'E-TITLE': 7,
                  'S-TITLE': 8, 'B-CONT': 9, 'M-CONT': 10, 'E-CONT': 11, 'S-CONT': 12, 'B-EDU': 13, 'M-EDU': 14,
                  'E-EDU': 15, 'S-EDU': 16, 'B-ORG': 17, 'M-ORG': 18, 'E-ORG': 19, 'S-ORG': 20, 'B-RACE': 21,
                  'M-RACE': 22, 'E-RACE': 23, 'S-RACE': 24, 'B-PRO': 25, 'M-PRO': 26, 'E-PRO': 27, 'S-PRO': 28,
                  'B-LOC': 29, 'M-LOC': 30, 'E-LOC': 31, 'S-LOC': 32}
        display_steps = 5
    elif pre_args.dataset == 'cluener':
        scheme = "BMES"
        categories = ['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position',
                      'scene']
        display_steps = 10
    else:
        raise Exception("invalid dataset name")

    parser.add_argument("--train_file_path", default=dataset_to_file_path[pre_args.dataset]["train"], type=str)
    parser.add_argument("--eval_file_path", default=dataset_to_file_path[pre_args.dataset]["eval"], type=str)
    parser.add_argument("--test_file_path", default=dataset_to_file_path[pre_args.dataset]["test"], type=str)
    parser.add_argument("--display_steps", default=display_steps, type=int)

    arguments = parser.parse_args()

    logging.info('loading pretrained model ...')
    if arguments.model_card == 'hfl/chinese-roberta-wwm-ext-large':
        tokenizer = BertTokenizerFast('../pretrained_model/roberta/hfl-roberta-wwm-ext-large/vocab.txt')
        config = BertConfig.from_pretrained('../pretrained_model/roberta/hfl-roberta-wwm-ext-large/config.json')
        pretrained_model = BertModel.from_pretrained(
            '../pretrained_model/roberta/hfl-roberta-wwm-ext-large/pytorch_model.bin', config=config)
    elif arguments.model_card == 'bert-base-chinese':
        tokenizer = BertTokenizerFast('../pretrained_model/bert-base-chinese/bert-base-chinese.vocab')
        config = BertConfig.from_pretrained('../pretrained_model/bert-base-chinese/bert-base-chinese-config.json')
        pretrained_model = BertModel.from_pretrained(
            '../pretrained_model/bert-base-chinese/bert-base-chinese-pytorch_model.bin', config=config)
    else:
        raise Exception('unknown model card')

    ner_tokenizer = NerTokenizer(categories, scheme, arguments.max_seq_len, tokenizer, True)
    main(arguments, ner_tokenizer, pretrained_model)


'''
guide to use this script

python run_bert_crf_beta0911.py  --do_eval \
                        --checkpoint_name robert-crf-wwm-large-2 \
                        --batch_size 32 \
                        --bert_hidden_size 1024 \
                        --dataset cluener


python run_bert_crf_beta0911.py  --do_predict \
                        --use_lstm \
                        --checkpoint_name robert-lstm-crf-wwm-large-2

python run_bert_crf_beta0911.py \
                                --do_train \
                                --use_lstm \
                                --dump_model \
                                --dataset people \
                                --max_seq_len 150 \
                                --batch_size 32 \
                                --bert_hidden_size 768 \
                                --model_card bert-base-chinese \
                                --model_name bert-lstm-crf \
                                --other_learning_rate 0.01
'''
