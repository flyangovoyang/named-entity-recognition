# softlexicon protype (unfinished)
# @last update time: 2020-09-07
# @author: youngf

import os
import json
import pickle
import numpy as np
from collections import defaultdict
from torch.utils.data.dataset import Dataset
from torch.utils.data import RandomSampler, DataLoader
import torch


class TrieNode:
    # Initialize your data structure here.
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_word = False


class Trie:
    """ build trie tree based on lexicon """

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        """ insert a word """
        current = self.root
        for letter in word:
            current = current.children[letter]
        current.is_word = True

    def search(self, word):
        """ search to see whether word exists in trie tree """
        current = self.root
        for letter in word:
            current = current.children.get(letter)
            if current is None:
                return False
        return current.is_word

    def startsWith(self, prefix):
        """ check whether prefix `prefix` exists in trie tree """
        current = self.root
        for letter in prefix:
            current = current.children.get(letter)
            if current is None:
                return False
        return True

    def scan(self, word):
        """ two bool values represents `startswith` and `in` """
        current = self.root
        for letter in word:
            current = current.children.get(letter)
            if current is None:
                return False, False
        return True, current.is_word

    # def enumerateMatch(self, word, space="_", backward=False):  # space=''
    #     """ word存在于trie树中的所有前缀词 """
    #     matched = []
    # 
    #     while len(word) > 0:
    #         if self.search(word):
    #             matched.append(space.join(word[:]))
    #         del word[-1]  # 删掉word最后一个字
    #     return matched


class Node:
    """ each character has four word sets, which is represented as a Node instance """
    def __init__(self):
        self.B = set()
        self.M = set()
        self.E = set()
        self.S = set()

    def __repr__(self):
        return '(B:{}, M:{}, E:{}, S:{})'.format(self.B, self.M, self.E, self.S)


class NerExample:
    def __init__(self, sent, ch_word_sets, tags):
        self.sent = sent
        self.ch_word_sets = ch_word_sets
        self.tags = tags


class SoftLexiconTokenizer:
    def __init__(self,
                 lexicon_path,
                 vocab_cache_path=None,
                 corpus_file_path=None,
                 embedding_file_path=None,
                 embedding_cache_path=None,
                 embedding_dim=300,
                 min_count=0,
                 verbose=True):
        self.tag2id = {"O": 0, "B-T": 1, "M-T": 2, "E-T": 3, "S-T": 4}
        self.id2tag = {v: k for k, v in self.tag2id.items()}
        self.word2id = {}
        self.max_seq_len = 100
        self.max_word_sets_len = 2
        self.trie_tree = None
        self.lexicon2weight = None
        self.verbose = verbose
        self.embedding = None
        # load lexicon
        self.load_lexicon(lexicon_path)
        # build vocab
        self.build_vocab(vocab_cache_path, corpus_file_path, min_count)
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.build_embedding(embedding_cache_path, embedding_file_path, embedding_dim)

    def load_lexicon(self, lexicon_file_path):
        lexicon2weight = {}
        with open(lexicon_file_path, 'r', encoding='utf8') as fin:
            for line in fin:
                line = line.strip().lower()
                line_list = line.split()
                assert 0 < len(line_list) < 3
                if len(line_list) == 2:
                    lexicon2weight[line_list[0]] = float(line_list[1])
                else:
                    lexicon2weight[line_list[0]] = 1.0

        trie_tree = Trie()
        for word in lexicon2weight:
            trie_tree.insert(word)

        lexicon2weight['none'] = 0

        self.trie_tree = trie_tree
        self.lexicon2weight = lexicon2weight
        if self.verbose:
            print('trie_tree built, lexicon length: {}'.format(len(self.lexicon2weight)))

    def build_vocab(self, vocab_cache_path, corpus_file_path, min_count):
        """ build word2id from cache or from train corpus """
        if vocab_cache_path is not None and os.path.exists(vocab_cache_path):
            vocab = list()
            with open(vocab_cache_path, 'r', encoding='utf8') as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    vocab.append(line)
        else:
            count = {}
            with open(corpus_file_path, 'r', encoding='utf8') as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    ch = line.split()[0]
                    if ch in count:
                        count[ch] = 1
                    else:
                        count[ch] += 1
            sorted_kv_pairs = sorted(count.items(), key=lambda x: x[1], reverse=True)
            filter_kv_pairs = list(filter(lambda x: x[1] >= min_count, sorted_kv_pairs))
            vocab = ['[PAD]', '[UNK]']
            for k, _ in filter_kv_pairs:
                vocab.append(k)
            with open(vocab_cache_path, 'w', encoding='utf8') as fout:
                for word in vocab:
                    fout.write('{}\n'.format(word))

        self.word2id = {word: index for index, word in enumerate(vocab)}  # dict remember keys' insert order
        for word in self.lexicon2weight:
            self.word2id[word] = len(self.word2id)
        if self.verbose:
            print('vocab built, word2id length: {}'.format(len(self.word2id)))

    def build_embedding(self, embedding_cache_path, embedding_file_path, embedding_dim):
        if embedding_cache_path is not None and os.path.exists(embedding_cache_path):
            embedding = pickle.load(open(embedding_cache_path, 'rb'))
        else:
            words = list(self.word2id.keys())  # dict remember keys' insert order
            embedding = np.zeros([len(self.word2id), embedding_dim], dtype=np.float)
            covered = 0
            with open(embedding_file_path, 'r', encoding='utf8') as fin:
                for line in fin:
                    line_list = line.strip().split(' ')
                    if len(line_list) == 2:
                        continue
                    if line_list[0] in words:
                        try:
                            embedding[self.word2id[line_list[0]]] = list(map(float, line_list[1:]))
                        except ValueError:
                            print(line_list)
                            print(line)
                        covered += 1
            if self.verbose:
                print('embedding oov: {:.4f}%'.format(100 - 100 * covered / len(self.word2id)))
            pickle.dump(embedding, open(embedding_cache_path, 'wb'))
        self.embedding = embedding

    def get_char_word_sets(self, seq):
        """
        seq: input token sequence
        return: word_sets: list of nodes(length equals seq), each node contains B, M, E, S and weight five attributes
        """

        def span(seq, i, j):
            return ''.join(seq[i:j + 1])

        char_word_sets = [Node() for _ in range(len(seq))]

        for i in range(len(seq)):
            flag_startwith, flag_in = self.trie_tree.scan(seq[i])
            if not flag_startwith:
                continue
            # S
            if flag_in:
                char_word_sets[i].S.add(seq[i])
            # B
            j = i + 1
            while True:
                if j >= len(seq):
                    break
                flag_startwith, flag_in = self.trie_tree.scan(span(seq, i, j))
                if flag_startwith:
                    # only when flag_in is true, update word_sets
                    if flag_in:
                        try:
                            char_word_sets[i].B.add(span(seq, i, j))
                        except TypeError:
                            print(span(seq, i, j))
                        for k in range(i + 1, j):
                            char_word_sets[k].M.add(span(seq, i, j))
                        char_word_sets[j].E.add(span(seq, i, j))
                    j += 1
                else:
                    break

        max_word_sets_len = max([max([len(node.M), len(node.B), len(node.E), len(node.S)]) for node in char_word_sets])
        if max_word_sets_len > self.max_word_sets_len:
            print('meeting new max word sets length: {}'.format(max_word_sets_len))
        return char_word_sets

    def read_corpus(self, corpus_file_path):
        assert self.trie_tree is not None and self.lexicon2weight is not None

        ner_examples = []
        sent = []
        tags = []
        with open(corpus_file_path, 'r', encoding='utf8') as fin:
            for line in fin:
                line = line.strip()
                if not line and sent:  # empty line
                    assert len(sent) == len(tags)
                    ch_word_sets = self.get_char_word_sets(sent)
                    ner_examples.append(NerExample(sent=sent, ch_word_sets=ch_word_sets, tags=tags))
                    sent = []
                    tags = []
                else:
                    line_list = line.split()
                    assert len(line_list) >= 2
                    sent.append(line_list[0])
                    tags.append(line_list[-1])
        return ner_examples

    def encode(self, example):
        sent = example.sent
        tags = example.tags
        ch_word_sets = example.ch_word_sets

        # padding
        input_mask = [True] * min(len(sent), self.max_seq_len) + [False] * max(0, self.max_seq_len - len(sent))
        sent = sent[:self.max_seq_len] + ['[PAD]'] * max(0, self.max_seq_len - len(sent))
        tags = tags[:self.max_seq_len] + ['O'] * max(0, self.max_seq_len - len(tags))
        b_seq, m_seq, e_seq, s_seq = [], [], [], []
        for node in ch_word_sets[:self.max_seq_len]:
            b, m, e, s = list(node.B), list(node.M), list(node.E), list(node.S)
            b_seq.append(b[:self.max_word_sets_len] + ['[PAD]'] * max(0, self.max_word_sets_len - len(b)))
            m_seq.append(m[:self.max_word_sets_len] + ['[PAD]'] * max(0, self.max_word_sets_len - len(m)))
            e_seq.append(e[:self.max_word_sets_len] + ['[PAD]'] * max(0, self.max_word_sets_len - len(e)))
            s_seq.append(s[:self.max_word_sets_len] + ['[PAD]'] * max(0, self.max_word_sets_len - len(s)))

        for _ in range(0, max(0, self.max_seq_len - len(ch_word_sets))):
            b_seq.append(['[PAD]'] * self.max_word_sets_len)
            m_seq.append(['[PAD]'] * self.max_word_sets_len)
            e_seq.append(['[PAD]'] * self.max_word_sets_len)
            s_seq.append(['[PAD]'] * self.max_word_sets_len)

        # vectorization
        input_sent = [self.word2id.get(ch, 1) for ch in sent]  # unk
        input_tags = [self.tag2id[tag] for tag in tags]
        arr = np.array([b_seq, m_seq, e_seq, s_seq])
        res = np.frompyfunc(lambda x: self.word2id[x], 1, 1)(arr)
        input_b, input_m, input_e, input_s = res[0].tolist(), res[1].tolist(), res[2].tolist(), res[3].tolist()
        return input_sent, input_tags, input_mask, input_b, input_m, input_e, input_s

    def test_encode(self, examples):
        """ test code, just ignore """
        print(examples[0].sent)
        print(examples[0].tags)
        print(examples[0].ch_word_sets)
        print('=' * 30)
        a, b, c, d, e, f, g = self.encode(examples[0])
        print('input_sent({}):{}'.format(len(a), a))
        print('input_tags({}):{}'.format(len(b), b))
        print('input_mask({}):{}'.format(len(c), c))
        print('input_b({}):{}'.format(len(d), d))
        print('input_m({}):{}'.format(len(e), e))
        print('input_e({}):{}'.format(len(f), f))
        print('input_s({}):{}'.format(len(g), g))

    def decode(self):
        pass


class AlarmDataset(Dataset):
    def __init__(self, features):
        self.input_sent = []
        self.input_tags = []
        self.input_masks = []
        self.input_b = []
        self.input_m = []
        self.input_e = []
        self.input_s = []
        for f in features:
            sent, tags, mask, b, m, e, s = f
            self.input_sent.append(sent)
            self.input_tags.append(tags)
            self.input_masks.append(mask)
            self.input_b.append(b)
            self.input_m.append(m)
            self.input_e.append(e)
            self.input_s.append(s)

    def __getitem__(self, index):
        return torch.tensor(self.input_sent[index]), \
               torch.tensor(self.input_tags[index]), \
               torch.tensor(self.input_masks[index]), \
               torch.tensor(self.input_b[index]), \
               torch.tensor(self.input_m[index]), \
               torch.tensor(self.input_e[index]), \
               torch.tensor(self.input_s[index])

    def __len__(self):
        return len(self.input_sent)


class SoftLexiconModel(torch.nn.Module):
    def __init__(self, embedding, embedding_dim):
        super(SoftLexiconModel, self).__init__()
        self.emb_layer = torch.nn.Embedding.from_pretrained(embedding)
        self.input_size = 5 * embedding_dim


def main(verbose=False):
    tokenizer = SoftLexiconTokenizer(lexicon_path='../resource/alarm/lexicon.data',
                                     vocab_cache_path='cache/alarm-vocab.data',
                                     corpus_file_path='../resource/alarm/alarm.train.data',
                                     embedding_file_path='../resource/Tencent_AILab_ChineseEmbedding.txt',
                                     embedding_cache_path='cache/alarm-embedding.pickle',
                                     embedding_dim=200,
                                     verbose=True)
    examples = tokenizer.read_corpus('../resource/alarm/alarm.train.data')
    dataset = AlarmDataset([tokenizer.encode(example) for example in examples])

    # print(tokenizer.id2word[874])
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    # features = [tokenizer.encode(example) for example in examples]
    # dataset = AlarmDataset(examples)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)
    for batch in dataloader:
        sent, tags, mask, b, m, e, s = tuple([t.to(device) for t in batch])
        loss = model('train', sent, tags, mask, b, m, e, s)

    # print(tokenizer.)
    # tokenizer.load_lexicon_from_file(dict_file_path)
    # ner_examples = tokenizer.read_corpus(train_file_path)
    # print(tokenizer.max_word_sets_len)
    # for i in range(10):
    #     for a, b in zip(ner_examples[i].sent, ner_examples[i].ch_word_sets):
    #         print(a, b)


if __name__ == '__main__':
    main(verbose=True)
