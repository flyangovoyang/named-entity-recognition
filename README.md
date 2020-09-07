# named-entity-recognition

## Choice

Model|Thinking
:--:|:--
BiLSTM-CRF|embedding combination can improve performance
Lattice LSTM|model overcomplicated, slow inference
Softlexicon|need lexicon character + word embedding, powerful when there are many word combinations for character sequence
Character Embedding + BERT-BiLSTM-CRF|using bert as a feature extractor, which means BERT's parameters are kept frozen all the time
BERT-CRF|
BERT-BiLSTM-CRF|
RoBERTa-CRF|
RoBERTa-BiLSTM-CRF|

others:

- CRF is highly recommended, cause I found it could bring further improvements for serveral models.
- When using pretrained language model, I am not willing to integrate traditional word embedding anymore, although this choice needs to be verified.
