import torch
import torchtext
import spacy

text_field = torchtext.data.Field(sequential=True, tokenize='spacy', include_lengths=True)
label_field = torchtext.data.Field(sequential=False, use_vocab=False)
train, val, test = torchtext.data.TabularDataset.splits(path='./data/', train='train.tsv',
                                                        validation='validation.tsv', test='test.tsv',
                                                        skip_header=True, format='TSV', fields=[('text', text_field),
                                                                                                ('label', label_field)])
text_field.build_vocab(train)
glove_model = torchtext.vocab.GloVe(name='6B', dim=100)
text_field.vocab.load_vectors(glove_model)

baseline = torch.load('model_baseline.pt')
rnn = torch.load('model_rnn.pt')
cnn = torch.load('model_cnn.pt')

nlp = spacy.load('en')

while True:
    print("Enter a sentence")
    sentence = input()
    tokens = nlp(sentence)
    x_list = [[text_field.vocab.stoi[token.text]] for token in tokens]
    x = torch.tensor(x_list, dtype=torch.long)
    length = torch.tensor([len(x_list)], dtype=torch.long)

    for model_type in ['baseline', 'rnn', 'cnn']:
        p = eval('{}(x, length).detach().numpy()'.format(model_type))
        if p > 0.5:
            print('Model {}: subjective ({})'.format(model_type, p))
        else:
            print('Model {}: objective ({})'.format(model_type, p))
