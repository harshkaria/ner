import torch
import sklearn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

# Find all sentences from the training data

from collections import defaultdict

def get_sentences_and_states(file_path, vocab = None):
  sentences = []
  states = []
  states_set = set()
  if vocab == None:
    vocab = set()
  with open(file_path, "r") as file:
    lines = file.readlines()
    # singular sentence
    sentence = []
    # singular tag of format (prev tag, curr tag)
    state_array = []
    for line in lines:
      # New sentence
      if len(line.strip()) == 0 and len(sentence) > 0:
        sentences.append([_ for _ in sentence])
        states.append([_ for _ in state_array])
        # print(sentences, states)
        sentence = []
        state_array = []
        continue
      word_split = line.strip().split(" ")
      # print(word_split)
      word = word_split[1] if len(word_split) >= 2 else None
      curr_state = word_split[2] if len(word_split) >= 3 else None
      if curr_state not in states_set:
        states_set.add(curr_state)
      if word not in vocab:
        vocab.add(word)
      # print(word, curr_state)
      sentence.append(word)
      state_array.append(curr_state)
    sentences.append([_ for _ in sentence])
    states.append([_ for _ in state_array])
    return sentences, states, states_set, vocab

def decode_sentences_and_tokens(sentences, tokens, true_tags):
  decoded_sents = []
  decoded_tokens = []
  true_tokens = []
  for sentence, token_arr, true_token_arr in zip(sentences, tokens, true_tags):
    sentence_decoded = [idx2word[idx] for idx in sentence]
    token_decoded = [idx2state[idx] for idx in token_arr]
    true_tags_decoded = [idx2state[idx] for idx in true_token_arr]
    decoded_sents.extend([sentence_decoded])
    decoded_tokens.extend([token_decoded])
    true_tokens.extend([true_tags_decoded])
  return decoded_sents, decoded_tokens, true_tokens

def detach(all_sentences, decoded_tags, gt):
  all_sentences = [torch.squeeze(sentence, 0) for sentence in all_sentences]
  all_sentences = [sentence.cpu().detach().numpy() for sentence in all_sentences]
  out_states = [tokens.cpu().detach().numpy() for tokens in decoded_tags]
  gt_states = [tokens.cpu().detach().numpy() for tokens in gt]
  return all_sentences, out_states, gt_states

def write_inference_to_file(decoded_tags, true_tags, test_sentences, filename, eval = True):
    with open(filename, "w") as file:
        for sentence, decoded_sentence_tags, gt_tags in zip(test_sentences, decoded_tags, true_tags):
            for i, (word, pred, gt) in enumerate(zip(sentence, decoded_sentence_tags, gt_tags), 1):
                if eval:
                  line = f'{i} {word} {gt} {pred}'
                else:
                  line = f'{i} {word} {pred}'
                file.write(line + '\n')
            file.write('\n')

import json
def load_dicts_from_json():
    # Load word2idx from JSON
    with open('word2idx.json', 'r') as f:
        word2idx = json.load(f)

    # Load state2idx from JSON
    with open('state2idx.json', 'r') as f:
        state2idx = json.load(f)

    # Load idx2word from JSON
    with open('idx2word.json', 'r') as f:
        idx2word = json.load(f)
        idx2word = {int(k): v for k, v in idx2word.items()}

    # Load idx2state from JSON
    with open('idx2state.json', 'r') as f:
        idx2state = json.load(f)
        idx2state = {int(k): v for k, v in idx2state.items()}

    return word2idx, state2idx, idx2word, idx2state

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from typing import List, Tuple

class SentencesDataset(Dataset):
  def __init__(self, sentences: List[str], states: List[List[str]], all_states: List[str], vocab: List[str], state2idx, word2idx, idx2word):
    self.sentences = sentences
    self.states = states
    self.all_states = all_states
    self.vocab = vocab
    self.state2idx = state2idx
    self.word2idx = word2idx
    self.idx2word = idx2word

  def __len__(self):
    return len(self.sentences)
  
  def __getitem__(self, idx):
    sentence = self.sentences[idx]
    encoded_sentence = [self.word2idx[word] for word in sentence]

    tags = self.states[idx]
    tags_encoded = []
    for tag in tags:
      one_hot = torch.zeros(len(self.all_states) + 1)
      one_hot[self.state2idx[tag]] = 1
      tags_encoded.append(one_hot)
    
    return encoded_sentence, tags_encoded

def collate_fn(batch):
    # Get the sequences and labels from the batch
    sequences, labels = zip(*batch)

    # Find the maximum sequence length in the batch
    max_len = max([len(seq) for seq in sequences])

    # Pad the sequences and labels to the maximum length
    padded_sequences = pad_sequence([torch.tensor(seq) for seq in sequences], batch_first=True, padding_value=0)
    
    for label in labels:
      while len(label) < max_len:
          zeros = torch.zeros(10)
          zeros[0] = 1
          label.append(zeros)
    labels = torch.stack([torch.stack(lst) for lst in labels])
    return padded_sequences, labels

sentences, states, all_states, vocab = get_sentences_and_states("./data/train")
dev_sentences, dev_states, _, vocab = get_sentences_and_states("./data/dev", vocab)
test_sentences, test_states, _, vocab = get_sentences_and_states("./data/test", vocab)

# No tags provided for test data
for sublist in test_states:
    # Iterate through each element in the sublist
    for i in range(len(sublist)):
        # Set the element to '<PAD>'
        sublist[i] = '<PAD>'

word2idx, state2idx, idx2word, idx2state = load_dicts_from_json()

dev_dataset = SentencesDataset(dev_sentences, dev_states, all_states, vocab, state2idx = state2idx, word2idx = word2idx, idx2word = idx2word)

dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False, collate_fn=lambda batch: collate_fn(batch))

test_dataset = SentencesDataset(test_sentences, test_states, all_states, vocab, state2idx = state2idx, word2idx = word2idx, idx2word = idx2word)

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda batch: collate_fn(batch))

import numpy as np
from sklearn.metrics import confusion_matrix

@torch.no_grad()
def infer(model, test_dataloader, trainer = None):

    running_loss = 0
    running_acc = 0
    test_output = []
    true_res = []
    incorrect_examples = []
    incorrect_labels = []
    incorrect_pred = []
    all_x = []

    for x, y in test_dataloader:

        x = x.long().to(device)
        y = y.to(device).float()
        all_x.append(x)
        output = model(x)
        output = output.view(-1, output.size(-1))
        y = y.view(-1, y.size(-1))
        pred = torch.argmax(output, dim=1)
        y = torch.argmax(y, dim=1)
        test_output.append(pred)
        true_res.append(y)
        if trainer:
          running_acc += trainer.calc_accuracy_test(output, y)
          loss = trainer.loss_fn(output, y)
          running_loss += loss.item()
        del x, y, output

    test_loss = running_loss / len(test_dataloader)
    test_acc = running_acc / len(test_dataloader)

    # calculate F1 score, precision, and recall
    test_output_cf = torch.cat(test_output, dim=0).cpu().numpy()
    true_res_cf = torch.cat(true_res, dim=0).cpu().numpy()

    conf_mat = confusion_matrix(true_res_cf, test_output_cf)
    true_positives = np.diag(conf_mat)
    false_positives = np.sum(conf_mat, axis=0) - true_positives
    false_negatives = np.sum(conf_mat, axis=1) - true_positives

    precision = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives))
    recall = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_negatives))
    f1 = 2 * precision * recall / (precision + recall)

    return test_loss, test_acc, test_output, true_res, f1, precision, recall, all_x

"""## Simple BLSTM """

import torch.nn as nn
class SimpleBidirectionalLSTM(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim, n_layers, device, dropout_pct, num_classes):
        super(SimpleBidirectionalLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device

        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
        self.dropout_pct = dropout_pct

        self.bilstm = nn.LSTM(input_size=embedding_dim,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              batch_first=True,
                              bidirectional=True)

        # Linear layer
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)

        # ELU activation function
        self.elu = nn.ELU()

        # Dropout layer
        self.dropout = nn.Dropout(self.dropout_pct)

        # Classifier layer
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, text):
        # print(text)
        # print(text.shape)
        # Embedding layer
        embedded = self.embedding_layer(text)

        # BiLSTM layer
        bilstm_output, _ = self.bilstm(embedded)

        # Apply dropout layer
        dropped = self.dropout(bilstm_output)

        # Apply linear layer
        linear_output = self.linear(dropped)

        # Apply ELU activation function
        elu_output = self.elu(linear_output)

        # Apply classifier layer to every time step
        output = self.classifier(elu_output)
        #print(output.shape)

        return output

blstm_model_1 = torch.load('blstm1.pt', map_location=torch.device('cpu'))

blstm_model_1.eval()

dev_loss, dev_acc, dev_output, dev_true_res, dev_f1, dev_precision, dev_recall, all_sentences = infer(model=blstm_model_1, test_dataloader=dev_dataloader)
decoded_sentences, decoded_tags_out, decoded_tags_true = detach(all_sentences, dev_output, dev_true_res)
decoded_sentences, decoded_tags_out, decoded_tags_true = decode_sentences_and_tokens(decoded_sentences, decoded_tags_out, decoded_tags_true)

test1_loss, test1_acc, test1_output, test1_true_res, test1_f1, test1_precision, test1_recall, test_sentences = infer(model=blstm_model_1, test_dataloader=test_dataloader)
decoded_sentences_t, decoded_tags_out_t, decoded_tags_true_t = detach(test_sentences, test1_output, test1_true_res)
decoded_sentences_t, decoded_tags_out_t, decoded_tags_true_t = decode_sentences_and_tokens(decoded_sentences_t, decoded_tags_out_t, decoded_tags_true_t)

write_inference_to_file(decoded_tags_out, decoded_tags_true, decoded_sentences, "dev1.out", eval = False)

write_inference_to_file(decoded_tags_out_t, decoded_tags_true_t, decoded_sentences_t, "test1.out", eval = False)

"""## BLSTM with GloVE"""

class GloVEBidirectionalLSTM(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim, n_layers, device, dropout_pct, num_classes, embedding_mat):
        super(GloVEBidirectionalLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
      

        # Create embedding layer
        self.embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_mat), freeze=True)
        self.dropout_pct = dropout_pct

        self.bilstm = nn.LSTM(input_size=embedding_dim,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              batch_first=True,
                              bidirectional=True)

        # Linear layer
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)

        # ELU activation function
        self.elu = nn.ELU()

        # Dropout layer
        self.dropout = nn.Dropout(self.dropout_pct)

        # Classifier layer
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, text):
        # print(text)
        # print(text.shape)
        # Embedding layer
        embedded = self.embedding_layer(text)

        # BiLSTM layer
        bilstm_output, _ = self.bilstm(embedded)

        # Apply dropout layer
        dropped = self.dropout(bilstm_output)

        # Apply linear layer
        linear_output = self.linear(dropped)

        # Apply ELU activation function
        elu_output = self.elu(linear_output)

        # Apply classifier layer to every time step
        output = self.classifier(elu_output)
        #print(output.shape)

        return output

blstm_model_2 = torch.load('blstm2.pt', map_location=torch.device('cpu'))

blstm_model_2.eval()

dev_loss_blstm2, dev_acc_blstm2, dev_output_blstm2, dev_true_res_blstm2, dev_f1_blstm2, dev_precision_blstm2, dev_recall_blstm2, all_sentences_blstm2 = infer(model=blstm_model_2, test_dataloader=dev_dataloader)

decoded_sentences_blstm2, decoded_tags_out_blstm2, decoded_tags_true_blstm2 = detach(all_sentences_blstm2, dev_output_blstm2, dev_true_res_blstm2)
decoded_sentences_blstm2, decoded_tags_out_blstm2, decoded_tags_true_blstm2 = decode_sentences_and_tokens(decoded_sentences_blstm2, decoded_tags_out_blstm2, decoded_tags_true_blstm2)

test_loss_blstm2, test_acc_blstm2, test_output_blstm2, test_true_res_blstm2, test_f1_blstm2, test_precision_blstm2, test_recall_blstm2, test_sentences_blstm2 = infer(model=blstm_model_2, test_dataloader=test_dataloader)

decoded_sentences_blstm2_t, decoded_tags_out_blstm2_t, decoded_tags_true_blstm2_t = detach(test_sentences_blstm2, test_output_blstm2, test_true_res_blstm2)
decoded_sentences_blstm2_t, decoded_tags_out_blstm2_t, decoded_tags_true_blstm2_t = decode_sentences_and_tokens(decoded_sentences_blstm2_t, decoded_tags_out_blstm2_t, decoded_tags_true_blstm2_t)

write_inference_to_file(decoded_tags_out_blstm2, decoded_tags_true_blstm2, decoded_sentences_blstm2, "dev2.out", eval = False)

write_inference_to_file(decoded_tags_out_blstm2_t, decoded_tags_true_blstm2_t, decoded_sentences_blstm2_t, "test2.out", eval = False)