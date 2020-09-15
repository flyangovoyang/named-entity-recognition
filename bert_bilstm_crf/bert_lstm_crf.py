import torch.nn as nn
import torch
from torch.nn import Linear, Dropout, LSTM
from crf import ConditionalRandomField as CRF


class BERT_LSTM_CRF(nn.Module):
    def __init__(self, bert, bert_hidden_size, num_tags, use_lstm=None, lstm_hidden_size=None, id2tag=None,
                 dropout_rate=0.0):
        """
        crf is required, lstm is optional
        """
        super(BERT_LSTM_CRF, self).__init__()
        self.bert = bert
        self.use_lstm = use_lstm
        if self.use_lstm:
            self.lstm = LSTM(input_size=bert_hidden_size,
                             hidden_size=lstm_hidden_size,
                             batch_first=True,
                             bidirectional=True,
                             num_layers=1)
            self.linear = Linear(2 * lstm_hidden_size, num_tags)
        else:
            self.linear = Linear(bert_hidden_size, num_tags)
        self.dropout = Dropout(p=dropout_rate)
        self.crf_layer = CRF(num_tags)

    def forward(self, mode, input_ids, tags=None, input_masks=None):
        """
        when in `inference` mode, return list [B, ?]
        """
        bert_hidden_states, _ = self.bert(input_ids)
        if self.use_lstm:
            lstm_hidden_states, _ = self.lstm(bert_hidden_states)
            token_logits = self.linear(self.dropout(lstm_hidden_states))
        else:
            token_logits = self.linear(self.dropout(bert_hidden_states))

        if mode == 'train':
            loss = -self.crf_layer.forward(inputs=token_logits, tags=tags, mask=input_masks) / token_logits.size(0)
            return loss
        elif mode == 'inference':
            crf_res = self.crf_layer.viterbi_tags(logits=token_logits, mask=input_masks)
            return [label for label, score in crf_res]  # each example's label length is decided according to mask
        else:
            raise Exception('only support `train` or `inference` modes.')
