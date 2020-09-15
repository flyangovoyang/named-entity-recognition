# named-entity-recognition

This is a stable version of code for sequence tagging task.

## BERT-BiLSTM-CRF

directory `bert_bilstm_crf` consists of four python scripts:

- `crf.py`: utility function and class for conditional random field layer, which is called by `bert_lstm_crf.py`
- `bert_lstm_crf.py`: bert+bilstm+crf model implementation, in which bilstm layer is optional
- `ner_data_util.py`: text sequence's encoding and decoding, and build dataloader 
- `run_bert_crf.py`: running script, more guides are within it.

when running this model, please download pretrained model(not limited to BERT, roberta is supported too.) files from huggingface transformer, which includes three files: config, vocab, model binary file.

modify all the directory paths to fits your own running environment.

## SoftLexicon

directory `soft_lexicon` currently consists of only one python script:

- `protype.py`: input preparation for SoftLexicon

