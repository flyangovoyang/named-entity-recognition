# running script for bert-bilstm-crf
# @create time: 2020-07-10
# @last update time: 2020-09-07
# @author: youngf

import os
import torch
import shutil
import logging
import argparse
import torch.optim as optim
from transformers import BertTokenizer, BertModel, BertConfig, BertTokenizerFast
import time
from bert_lstm_crf import BERT_LSTM_CRF
from data_util import *
import numpy as np
import random
from metric import fmeasure_from_singlefile


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def train(model, tokenizer, device, train_data_loader, eval_dataloader, args):
    logging.info('start to train...')

    crf_params_list = ['crf_layer.transitions', 'crf_layer._constraint_mask', 'crf_layer.start_transitions',
                       'crf_layer.end_transitions']  # , 'linear_weight', 'linear_bias'
    crf_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in crf_params_list, model.named_parameters()))))
    other_params = list(
        map(lambda x: x[1], list(filter(lambda kv: kv[0] not in crf_params_list, model.named_parameters()))))

    optimizer = optim.Adam([
        {"params": crf_params, "lr": args.crf_learning_rate},
        {"params": other_params}
    ], lr=args.learning_rate)

    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    best_metric_score = 0
    for epoch in range(args.train_epochs):
        for batch_id, batch in enumerate(train_data_loader):
            model.train()
            model.zero_grad()
            input_tokens, input_ids, _, tag_ids, input_masks = batch
            # preprocessing
            input_tokens = transform_input_tokens(input_tokens)
            input_ids = input_ids.to(device)
            tag_ids = tag_ids.to(device)
            input_masks = input_masks.to(device)
            loss = model('train', input_ids, tags=tag_ids,
                         input_masks=input_masks)  # logits: batch_size, length, num_tags
            if batch_id and batch_id % args.display_steps == 0:
                template = 'epochs={:2}/{:2}, batch={:3}/{:3}, loss={:.6f}'
                logging.info(template.format(epoch, args.train_epochs, batch_id, len(train_data_loader), loss))
            loss.backward()
            optimizer.step()

        logging.info('epoch evaluation:')
        metric_score = evaluate(model, tokenizer, device, eval_dataloader, args)
        if args.dump_model and metric_score > best_metric_score:
            best_metric_score = metric_score
            torch.save(model.state_dict(), os.path.join(args.dump_model_path, args.model_name + '-' + str(epoch)))
            logging.info(os.path.join(args.dump_model_path, args.model_name + '-' + str(epoch)) + ' dumped!')


def transform_input_tokens(input_tokens):
    batch_size = len(input_tokens[0])
    batch_data = []
    for i in range(batch_size):
        batch_data.append([x[i] for x in input_tokens])
    return batch_data


def predict(model, tokenizer, device, args):
    if os.path.exists('predict_result.data'):
        os.remove('predict_result.data')

    if args.init_checkpoint:
        logging.info('loading model from {}'.format(args.init_checkpoint))
        model.load_state_dict(
            torch.load(os.path.join(args.dump_model_path, args.init_checkpoint),
                       map_location=device)
        )

    model.eval()
    logging.info('start to predict...')
    with torch.no_grad():
        with open('predict_result.data', 'w', encoding='utf8') as fout:
            with open(args.test_file_path, 'r', encoding='utf8') as fin:
                sent = []
                for line in fin:
                    line = line.strip()
                    if not line:
                        if sent:
                            input_tokens = ['[CLS]'] + sent + ['[SEP]']
                            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                            input_masks = [True] + [True]*len(sent) + [False]
                            outputs = model('inference', torch.tensor([input_ids], dtype=torch.long).to(device))
                            preds = outputs[0][1:]
                            pred_tags = [args.id2tag[tag_id] for tag_id in preds]
                            for token, pred_tag in zip(sent, pred_tags):
                                fout.write('{} {}\n'.format(token, pred_tag))
                            fout.write('\n')
                            sent = []
                    else:
                        sent.append(line)


def evaluate(model, tokenizer, device, eval_dataloader, args):
    if os.path.exists('eval_result.data'):
        os.remove('eval_result.data')

    if not args.do_train and args.init_checkpoint:
        logging.info('loading model from {}'.format(args.init_checkpoint))
        model.load_state_dict(
            torch.load(os.path.join(args.dump_model_path, args.init_checkpoint),
                       map_location=device)
        )

    model.eval()
    logging.info('start to eval...')
    with torch.no_grad():
        with open('eval_result.data', 'a', encoding='utf8') as fout:
            for batch_id, batch in enumerate(eval_dataloader):
                input_tokens, input_ids, tags, tag_ids, input_masks = batch
                # preprocessing
                input_tokens = transform_input_tokens(input_tokens)
                input_ids = input_ids.to(device)
                input_masks = input_masks.to(device)
                label_preds = model('inference', input_ids, input_masks=input_masks)

                current_batch_size = len(input_ids)
                for i in range(current_batch_size):
                    source_str = input_tokens[i]
                    source_tags = [args.id2tag[x.item()] for x in tag_ids[i]]
                    pred_tags = [args.id2tag[x] for x in label_preds[i]]

                    # automatically align the length according to pred_tags
                    for source_ch, source_tag, pred_tag in zip(source_str[1:], source_tags[1:], pred_tags[1:]):
                        fout.write('{} {} {}\n'.format(source_ch, source_tag, pred_tag))
                    fout.write('\n')

        acc, p, r, f1 = fmeasure_from_singlefile('eval_result.data', "BMES")
        logging.info('evaluation result: acc={:.4f}, precision={:.4f}, recall={:.4f}, f1={:.4f}'.format(acc, p, r, f1))
    return f1


def main(args, processor, tokenizer):
    # output config information
    logging.info('running parameters:')
    # TODO: print all the configuration
    logging.info('')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load pretrained model
    logging.info('loading pretrained model...')
    if args.model_card == 'hfl/chinese-roberta-wwm-ext-large':
        config = BertConfig.from_pretrained('hfl-roberta-wwm-ext-large/config.json')
        bert = BertModel.from_pretrained('hfl-roberta-wwm-ext-large/pytorch_model.bin', config=config)
    else:
        bert = BertModel.from_pretrained(args.model_card)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info('device: {}'.format(device))
    model = BERT_LSTM_CRF(bert,
                          args.num_tags,
                          args.lstm_hidden_size,
                          id2tag=args.id2tag,
                          dropout_rate=args.dropout_rate).to(device)

    if args.do_train or args.do_eval:
        logging.info('loading eval corpus...')
        eval_dataloader = processor.get_dataloader(src_file_path=args.eval_file_path,
                                                   tokenizer=tokenizer,
                                                   max_seq_len=args.max_seq_len,
                                                   batch_size=args.batch_size,
                                                   sampler="sequential")
    if args.do_train:
        logging.info('loading train corpus...')

        train_dataloader = processor.get_dataloader(src_file_path=args.train_file_path,
                                                    tokenizer=tokenizer,
                                                    max_seq_len=args.max_seq_len,
                                                    batch_size=args.batch_size,
                                                    sampler="random")
        train(model, tokenizer, device, train_dataloader, eval_dataloader, args)
    elif args.do_eval:
        evaluate(model, tokenizer, device, eval_dataloader, args)
    elif args.do_predict:
        predict(model, tokenizer, device, args)
    else:
        raise Exception("either `do_train` or `do_eval` must be `True`")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--verbose_logging",
                        default=False,
                        action="store_true",
                        help="whether output data examples")
    parser.add_argument("--do_train",
                        default=False,
                        action="store_true",
                        help="train mode")
    parser.add_argument("--do_eval",
                        default=False,
                        action="store_true",
                        help="eval mode")
    parser.add_argument("--do_predict",
                        default=True,
                        action="store_true",
                        help="predict mode")
    parser.add_argument("--train_epochs",
                        default=40,
                        type=int,
                        help="")
    parser.add_argument("--dataset",
                        default="cluener",
                        type=str,
                        help="")
    parser.add_argument("--dump_model_path",
                        default="models",
                        type=str,
                        help="")
    parser.add_argument("--init_checkpoint",
                        default="seed-40-dropout-0-14",
                        type=str,
                        help="")
    parser.add_argument("--dump_model",
                        default=False,
                        action="store_true",
                        help="whether dump model during the training or evaluation")
    parser.add_argument("--lstm_hidden_size",
                        default=128,
                        type=int)
    parser.add_argument("--dropout_rate",
                        default=0,
                        type=float)
    parser.add_argument("--max_seq_len",
                        default=60,
                        type=int)
    parser.add_argument("--batch_size",
                        default=32,
                        type=int)
    parser.add_argument("--seed",
                        default=100,
                        type=int)
    parser.add_argument("--optimizer",
                        default="adam",
                        type=str)
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float)
    parser.add_argument("--crf_learning_rate",
                        default=0.01,
                        type=float)
    parser.add_argument("--model_card",
                        default="hfl/chinese-roberta-wwm-ext-large",  # bert-base-chinese
                        type=str)
    parser.add_argument("--model_name",
                        default="roberta-wwm-ext-bilstm-crf",
                        type=str)

    pre_args = parser.parse_args()

    # prepare dependent parameters
    dataset_to_file_path = {
        "people": {
            "train": "corpus/people/train.data",
            "eval": "corpus/people/dev.data",
            "test": "corpus/people/test.data"
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
        tag2id = {'O': 0, 'B-ORG': 1, 'M-ORG': 2, 'E-ORG': 3, 'S-ORG': 4,
                  'B-PER': 5, 'M-PER': 6, 'E-PER': 7, 'S-PER': 8,
                  'B-LOC': 9, 'M-LOC': 10, 'E-LOC': 11, 'S-LOC': 12}
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
        scheme = ['B', 'M', 'E', 'S']
        categories = ['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position',
                      'scene']
        tag2id = {'O': 0}
        for c in categories:
            for s in scheme:
                tag2id[s+'-'+c] = len(tag2id)
        display_steps = 10
    else:
        raise Exception("invalid dataset name")

    id2tag = {v: k for k, v in tag2id.items()}
    parser.add_argument("--train_file_path", default=dataset_to_file_path[pre_args.dataset]["train"], type=str)
    parser.add_argument("--eval_file_path", default=dataset_to_file_path[pre_args.dataset]["eval"], type=str)
    parser.add_argument("--test_file_path", default=dataset_to_file_path[pre_args.dataset]["test"], type=str)
    parser.add_argument("--tag2id", default=tag2id, type=dict, help="")
    parser.add_argument("--id2tag", default=id2tag, type=dict, help="")
    parser.add_argument("--num_tags", default=len(tag2id), type=int, help="")
    parser.add_argument("--display_steps", default=display_steps, type=int, help="")

    arguments = parser.parse_args()

    if arguments.model_card == 'hfl/chinese-roberta-wwm-ext-large':
        tokenizer = BertTokenizerFast('hfl-roberta-wwm-ext-large/vocab.txt')
    else:
        tokenizer = BertTokenizer.from_pretrained(arguments.model_card)

    processor = Processor(arguments.tag2id, tokenizer, arguments.verbose_logging)
    main(arguments, processor, tokenizer)


'''
python run_bert_crf.py --do_train --dump_model --model_name seed-40-dropout-0
'''
