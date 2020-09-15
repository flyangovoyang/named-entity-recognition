"""
tokenizer for token classification task

@author: youngf
@create time: 2020-09-10
@last update: 2020-09-11
"""
import torch
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
import logging
from transformers import BertTokenizerFast

logging.basicConfig(format='[%(asctime)s]-%(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)


class NerFeature:
    """ vectorized input """
    def __init__(self, uid, input_ids, input_masks, tag_ids=None, token2char=None):
        self.uid = uid
        self.input_ids = input_ids
        self.tag_ids = tag_ids
        self.input_masks = input_masks
        self.token2char = token2char


class NerTokenizer:
    """
    read corpus, build vocab, tokenizer sentence, build data loader
    build tag2id, id2tag
    """

    def __init__(self, categories, tagging_schema, max_seq_len, ptm_tokenizer=None, debug=False):
        self.tag2id = self.build_tag2id(categories, tagging_schema)
        self.ptm_tokenizer = ptm_tokenizer
        self.id2tag = {v: k for k, v in self.tag2id.items()}
        self.max_seq_len = max_seq_len
        self.tagging_schema = tagging_schema  # currently only support BMES, BIO need to update `token_tag_to_ch_tag`
        self.debug = debug

    @staticmethod
    def build_tag2id(labels, schema):
        tag_to_id = {'O': 0}
        if schema == 'BIO':
            position_symbols = ['B', 'I', 'O']
        elif schema == 'BMES':
            position_symbols = ['B', 'M', 'E', 'S']
        else:
            raise RuntimeError('tagging_schema should be either `BIO` or `BMES`')
        for label in labels:
            for symbol in position_symbols:
                tag_to_id[symbol + '-' + label] = len(tag_to_id)
        return tag_to_id

    def ch_seq_to_feature(self, sent, tags, uid):
        """
        vectorize character sequence (including padding, truncating and vectorization) it by pretrained model tokenizer

        For Chinese NER task, tokenize the character sequence to token sequence and polish the tag sequence according
        to `tokens_to_chars`. After that, four quotations `“`, `”`, `’`, `‘` should be converted to their English
        counterparts to avoid `[unk]` noise
        """
        sent_str = ''.join(sent)

        en_style_sent_str = sent_str.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")
        batch_encoding = self.ptm_tokenizer.encode_plus(en_style_sent_str, pad_to_max_length=True,
                                                        max_length=self.max_seq_len, truncation=True)
        sent_tokens = self.ptm_tokenizer.tokenize(en_style_sent_str)
        token_ids = batch_encoding['input_ids']
        input_masks = batch_encoding['attention_mask']
        tokens_len = len(token_ids)

        if not tags:  # when predicting, tags is empty
            token2char = []
            for i in range(len(token_ids)):
                token2char.append(tuple(batch_encoding.token_to_chars(i)))
            if uid < 5:
                print('example-{}'.format(uid))
                print('sent_str({}):{}'.format(len(sent_str), sent_str))
                print('tags({}):{}'.format(len(tags), tags))
                print('sent_tokens({}):{}'.format(len(sent_tokens), sent_tokens))
                print('token_ids({}):{}'.format(len(token_ids), token_ids))
                print('input_masks({}):{}'.format(len(input_masks), input_masks))
                print('token2char({}):{}'.format(len(token2char), token2char))
                print()
            return token_ids, input_masks, token2char

        tokens_multi_tags = [[] for _ in range(len(token_ids))]
        token_merged_tags = ['O'] * len(token_ids)
        for i in range(tokens_len):
            span = batch_encoding.token_to_chars(i)  # when exceeds the character sequence, span=(0, 0)
            for j in range(span[0], span[1]):
                cur_category = tags[j] if tags[j] == 'O' else tags[j][2:]
                tokens_multi_tags[i].append(cur_category)  # make sure category symbol starts from index 2
        for i in range(tokens_len):
            tag_set = set(tokens_multi_tags[i]) - {'O'}  # remove the 'O' and count non 'O' categories
            if len(tag_set) == 1:
                token_merged_tags[i] = tag_set.pop()  # get the only one category
            elif len(tag_set) > 1:
                token_merged_tags[i] = 'mix'
                output_template = 'more than one ({}) categories on the single token {}, which is unexpected.'
                logging.warning(output_template.format(tag_set, sent_tokens[i + 1]))
        token_tags = NerTokenizer.generate_position_symbol(token_merged_tags, self.tagging_schema)
        token_tag_ids = [self.tag2id[tag] for tag in token_tags]

        if uid < 5:
            print('example-{}'.format(uid))
            print('sent_str({}):{}'.format(len(sent_str), sent_str))
            print('tags({}):{}'.format(len(tags), tags))
            print('sent_tokens({}):{}'.format(len(sent_tokens), sent_tokens))
            print('token_ids({}):{}'.format(len(token_ids), token_ids))
            print('token_tags({}):{}'.format(len(token_tags), token_tags))
            print('token_tag_ids({}):{}'.format(len(token_tag_ids), token_tag_ids))
            print('input_masks({}):{}'.format(len(input_masks), input_masks))
            print()
        return token_ids, input_masks, token_tag_ids

    @staticmethod
    def token_tag_to_ch_tag(token_tags, token2char):
        """ convert token tags seq to character tags seq """
        assert len(token_tags) == len(token2char)
        token_length = len(token_tags)  # n+2: [CLS] a1 ... an [SEP]
        ch_tags = ['O']*(token_length-2)
        for i in range(1, token_length-1):  # a1, ..., an
            if token_tags[i] != 'O':
                """
                |     token1    | token2 |     token3    |            token4
                |     B-xxx     | M-xxx  |     E-xxx     |            S-xxx
                
                | ch11  | ch12  |  ch2   |  ch31 | ch32  |        token41 token42
                | B-xxx | M-xxx | M-xxx  | M-xxx | E-xxx |         B-xxx   E-xxx
                
                for ch32, token3's end tag(E-xxx) should be preserved
                """
                pos_symbol = token_tags[i][0]
                category = token_tags[i][2:]
                s, e = token2char[i]
                if pos_symbol == 'B':
                    for j in range(s, e):
                        ch_tags[j] = token_tags[i] if j == s else 'M-'+category  # (B) -> (B, [M, M, ...])
                elif pos_symbol == 'M':
                    for j in range(s, e):
                        ch_tags[j] = token_tags[i]
                elif pos_symbol == 'E':
                    for j in range(s, e):
                        ch_tags[j] = token_tags[i] if j == e-1 else 'M-'+category  # (E) -> ([M, ...,] E)
                else:  # S
                    if e == s+1:
                        ch_tags[s] = token_tags[i]  # (S) -> (S)
                    else:
                        for j in range(s, e):  # (S) -> B[, [M, ] E]
                            if j == s:
                                ch_tags[j] = 'B-'+category
                            elif j == e-1:
                                ch_tags[j] = 'E-'+category
                            else:
                                ch_tags[j] = 'M-'+category
        return ch_tags

    @staticmethod
    def parse_entity_from_ch_tag(ch_tags):
        entity_pairs = []
        index = 0
        while index < len(ch_tags):
            if ch_tags[index] != 'O':
                if ch_tags[index][0] == 'S':
                    entity_pairs.append((index, index+1, ch_tags[index][2:]))
                elif ch_tags[index][0] == 'B':
                    s = index
                    while index < len(ch_tags) and (not ch_tags[index].startswith('E')) and ch_tags[index] != 'O':
                        index += 1
                    if index >= len(ch_tags) or ch_tags[index] == 'O':
                        entity_pairs.append((s, index, ch_tags[s][2:]))
                    else:
                        entity_pairs.append((s, index+1, ch_tags[s][2:]))
            index += 1
        return entity_pairs

    @staticmethod
    def generate_position_symbol(tags, tagging_schema):
        """ add position symbol to tag sequence (name, name, name, O, pos, pos) => (B-name, M-name, ....) """
        if tagging_schema == 'BIO':
            pass
        else:
            # forward scan
            mem = ''
            mark = [0] * len(tags)
            for i in range(len(tags)):
                if tags[i] != 'O':
                    if tags[i] == mem:
                        mark[i] = mark[i - 1] + 1
                    else:
                        mark[i] = 1
                mem = tags[i]
            # backward scan
            next_mark = 0
            symbol_res = ['O'] * len(tags)
            for i in range(len(tags) - 1, -1, -1):
                if mark[i] == 1:
                    if next_mark <= 1:
                        symbol_res[i] = 'S'
                    else:  # 2
                        symbol_res[i] = 'B'
                elif mark[i] >= 2:
                    if next_mark > mark[i]:
                        symbol_res[i] = 'M'
                    else:
                        symbol_res[i] = 'E'
                next_mark = mark[i]
            # merge
            merge_res = []
            for symbol, category in zip(symbol_res, tags):
                if symbol == 'O':
                    merge_res.append('O')
                else:
                    merge_res.append(symbol + '-' + category)
            return merge_res

    def read_corpus_file(self, file_path):
        """
        load examples from corpus file
        :param file_path: corpus file path
        :param ptm_tokenizer: pretrained model's tokenizer
        :param tagging_schema: `BIO` or `BMES`
        :return: for data with annotation , return features;
                    for data without annotation, return features and token2chars
        """
        tokens = []
        tags = []
        uid = 0
        features = []
        predict_mode = False  # automatically detect whether current corpus file has annotation
        with open(file_path, 'r', encoding='utf8') as fin:
            for line in fin:
                line = line.strip()
                if not line and tokens:
                    if tags and len(tokens) != len(tags):
                        logging.warning('length inconsistent: tokens:{}, tags:{}'.format(tokens, tags))
                        tokens, tags = [], []
                        continue
                    if self.ptm_tokenizer is not None:
                        if predict_mode:
                            input_ids, input_masks, token2char = self.ch_seq_to_feature(tokens, tags, uid)
                            features.append(NerFeature(uid=uid, input_ids=input_ids, input_masks=input_masks,
                                                       token2char=token2char))
                        else:
                            input_ids, input_masks, tag_ids = self.ch_seq_to_feature(tokens, tags, uid)
                            features.append(
                                NerFeature(uid=uid, input_ids=input_ids, input_masks=input_masks, tag_ids=tag_ids))
                    else:
                        # TODO: feature implementation
                        raise Exception('TBD')

                    uid += 1
                    tokens, tags = [], []
                else:
                    line_list = line.split(' ')
                    if len(line_list) == 1:
                        predict_mode = True
                        ch = line_list[0]
                        tokens.append(ch)
                    else:
                        ch, tag = line_list[0], line_list[-1]  # make sure character tag locates in the last column
                        tokens.append(ch)
                        tags.append(tag)
        print('loading {} features in file {}'.format(len(features), file_path))
        return features

    def build_dataloader(self, corpus_file_path, batch_size, sample="random"):
        features = self.read_corpus_file(corpus_file_path)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_masks = torch.tensor([f.input_masks for f in features], dtype=torch.bool)
        if features[0].tag_ids is not None:
            all_tag_ids = torch.tensor([f.tag_ids for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_masks, all_tag_ids)
        else:
            all_token2char = torch.tensor([f.token2char for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_masks, all_token2char)
        sampler = RandomSampler(dataset) if sample == 'random' else SequentialSampler(dataset)
        dataloader = DataLoader(dataset, batch_size, sampler)
        return dataloader


def main():
    ptm_tokenizer = BertTokenizerFast('../pretrained_model/roberta/hfl-roberta-wwm-ext-large/vocab.txt')
    ner_tokenizer = NerTokenizer(
        ['address', 'name', 'organization', 'game', 'scene', 'book', 'company', 'position', 'government', 'movie'],
        max_seq_len=100,
        tagging_schema='BMES',
        ptm_tokenizer=ptm_tokenizer,
        debug=True
    )
    # features = ner_tokenizer.read_corpus_file('../resource/cluener/test.bmes.data')
    # dataloader = ner_tokenizer.build_dataloader(features, 3, 'random')
    # for input_ids, input_masks, tag_ids in dataloader:
    #     print(input_ids, input_masks, tag_ids)
    #     assert 0
    print(ner_tokenizer.parse_entity_from_ch_tag(['B-a', 'E-a']))


if __name__ == '__main__':
    main()
