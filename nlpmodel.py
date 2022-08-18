from functools import partial
import itertools
import torch
import csv
import pickle
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.utils import get_tokenizer
tokenizer = get_tokenizer('basic_english')
from torchtext.vocab import GloVe
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import r2_score
import numpy as np
from nltk.corpus import stopwords

# ==============
# i. CONSTANTS
# ==============
VECTORS_CACHE_DIR = './.vector_cache'

UNK, PAD, LBS, LBE, SBS, SBE, PART, SW = 0, 1, 2, 3, 4, 5, 6, 7

TOKEN_CHANGES = {'motherfuckin' : 'fucking', 'drippin' : 'dripping', 
  'prayin' : 'praying', 'countin' : 'counting', 'knowin' : 'knowing', 
  'listenin' : 'listening', 'showin' : 'showing', 'whippin' : 'whipping', 
  'spendin' : 'spending', 'stuntin' : 'stunting', 'starin' : 'staring', 
  'trappin' : 'trapping', 'wonderin' : 'wondering', 'mothafuckin' : 'fucking', 
  'motherfucking' : 'fucking','winnin' : 'winning', 'grindin' : 'grinding', 
  'pourin' : 'pouring', 'breathin' : 'breathing', 'lettin' : 'letting', 
  'switchin' : 'switching', 'flexin' : 'flexing', 'speakin' : 'speaking',
  'the—' : 'the', 'thе' : 'the', 'aight' : 'alright', 'a-' : 'a',
  'hunnid' : 'hundred', 'prolly' : 'probably'}

# ==============
# A. Collate functions
# ==============

# A1. Helpers
def make_labels_tensor(batch, rate_type):
    labels = []
    for num, sample in enumerate(batch):
        if rate_type == 'c_rate':
            labels.append(int(sample[1]))
        elif rate_type == 'u_rate':
            labels.append(int(sample[2] * 10))
        else:
            raise RuntimeError("Please make sure rate_type is either 'c_rate' or 'u_rate'")
    return torch.tensor(labels, dtype = torch.float)

def token_to_char(token, char_vocab, reverse = False):
    if reverse:
        return torch.LongTensor([char_vocab[letter] for letter in token[::-1]])
    else:
        return torch.LongTensor([char_vocab[letter] for letter in token])

# A2. Collate Functions
def collate_into_bow(batch, vocab, rate_type, **kwargs):
    stops = set(stopwords.words('english'))
    treat_stops = kwargs.get('treat_stops', False)
    bows = torch.zeros(len(batch), len(vocab))
    labels = make_labels_tensor(batch, rate_type)
    for num, sample in enumerate(batch):
        for token in tokenizer(sample[3]):
            if treat_stops == 'see' and token in stops:
                bows[num, SW] += 1
            elif treat_stops == 'remove' and token in stops:
                pass
            else:
                bows[num, vocab[token]] += 1   
        bows[num,:] = bows[num, :] / (bows[num, :].sum())
    return labels, bows

def collate_into_cbow_unfrozen(batch, vocab, rate_type, **kwargs):
    labels = make_labels_tensor(batch, rate_type)
    word_idxs = []
    all_tokens = 0
    for num, sample in enumerate(batch):
        tokens = [TOKEN_CHANGES.get(i, i) for i in tokenizer(sample[3])]
        all_tokens += len(tokens)
        token_idxs = [vocab[token] for token in tokens]
        word_idxs.append(torch.LongTensor(token_idxs))
    word_idxs = pad_sequence(word_idxs, batch_first = False, padding_value = PAD)
    print(all_tokens / (num +1))
    return labels, word_idxs


def collate_batch_rnn_firstk(batch, vocab, rate_type, **kwargs):
    chunk_size = kwargs['chunk_size']
    chunk_portion = kwargs['chunk_portion']
    treat_stops = kwargs['treat_stops']
    labels = make_labels_tensor(batch, rate_type)
    word_idxs = []
    for num, sample in enumerate(batch):
        tokens = [TOKEN_CHANGES.get(i, i) for i in tokenizer(sample[3])]
        token_idxs = tokenize_treating_stops(tokens, vocab, treat_stops)
        if isinstance(chunk_size, int) or chunk_size == 'all':
            if chunk_size == 'all' or len(token_idxs) <= chunk_size:
                pass
            elif chunk_portion == 'first':
                token_idxs = token_idxs[:chunk_size]
            elif chunk_portion == 'last':
                token_idxs = token_idxs[-chunk_size:]
            elif chunk_portion == 'mid':
                token_idxs = token_idxs[:chunk_size // 2] + token_idxs[-chunk_size // 2 : ]
        else:
            print("Please enter a valid value (either an integer or string 'all') for chunk_size.")
            print("If you've already done os, please make sure that you've specified which chunk portion you want to use.")
            print("Valid values for chunk_portion include: ('first', 'last', 'mid'). None is an acceptable value if chunk_size is 'all'.")
            raise TypeError
        word_idxs.append(torch.LongTensor(token_idxs))
    word_idxs = pad_sequence(word_idxs, batch_first = False, padding_value = PAD)
    return labels, word_idxs

def tokenize_treating_stops(tokens, vocab, treat_stops):
    stops = set(stopwords.words('english'))
    token_idxs = []
    for token in tokens:
        if treat_stops == 'see' and token in stops:
            token_idxs.append(SW)
        elif treat_stops == 'remove' and token in stops:
            pass
        else:
            token_idxs.append(vocab[token])
    return token_idxs

# =======
# Still unchecked below
# =======

def collate_batch_chars(batch, vocab, rate_type, **kwargs):
    firstk = kwargs['chunk_size']
    char_vocab = kwargs['char_vocab']
    labels = make_labels_tensor(batch, rate_type)
    word_idxs = []
    for num, sample in enumerate(batch):
        tokens = [TOKEN_CHANGES.get(i, i) for i in tokenizer(sample[3])]
        tokens = tokens[:firstk]
        token_idxs = [vocab[token] for token in tokens]
        char_idxs = [token_to_char(token, char_vocab) for token in tokens]
        char_idxs = pad_sequence(char_idxs, batch_first = False, padding_value = PAD)
        char_idxs_b = [token_to_char(token, char_vocab, True) for token in tokens]
        char_idxs_b = pad_sequence(char_idxs_b, batch_first = False, padding_value = PAD)
        word_idxs.append(torch.LongTensor(token_idxs))
    word_idxs = pad_sequence(word_idxs, batch_first = False, padding_value = PAD)
    return labels, word_idxs, char_idxs, char_idxs_b

def collate_batch_song(batch, vocab, rate_type, **kwargs):
    chunk_size = kwargs['chunk_size']
    chunk_portion = kwargs['chunk_portion']
    treat_stops = kwargs['treat_stops']
    names = []
    labels = []
    songs = []
    for num, sample in enumerate(batch):
        if rate_type == 'c_rate':
            score = int(sample[1])
        else:
            score = int(sample[2]) * 10

        tokens = [TOKEN_CHANGES.get(i, i) for i in tokenizer(sample[3])]
        token_idxs = tokenize_treating_stops(tokens, vocab, treat_stops)
        song_tokens = []

        for token_idx in token_idxs:
            song_tokens.append(token_idx)
            if token_idx == SBE:
                labels.append(score)
                names.append(sample[0])
                songs.append(torch.LongTensor(song_tokens))
                song_tokens = []
    songs = pad_sequence(songs, batch_first = False, padding_value = PAD)
    print(len(labels))
    print(songs.shape)
    print(len(names))
    return torch.tensor(labels, dtype = torch.float), songs, names


def collate_batch_rnn_firstk_song(batch, vocab, rate_type, **kwargs):
    firstk = kwargs['chunk_size']
    labels = make_labels_tensor(batch, rate_type)
    word_idxs = []
    for num, sample in enumerate(batch):
        tokens = [TOKEN_CHANGES.get(i, i) for i in tokenizer(sample[3])]
        songs = []
        song_tokens = []
        for token in tokens:
            song_tokens.append(token)
            if token == '</sb>':
                songs.append(song_tokens)
                song_tokens = []
        # Tokens allowed to accumulate per song
        tps = int(firstk / len(songs)) 
        # Get first tps tokens for each song
        songs_concat = list(itertools.chain(*[song[:tps] + song[-1:] for song in songs])) 
        token_idxs = [vocab[token] for token in songs_concat]
        word_idxs.append(torch.LongTensor(token_idxs))
    word_idxs = pad_sequence(word_idxs, batch_first = False, padding_value = PAD)
    return labels, word_idxs

def collate_batch_chunk(batch, vocab, rate_type, **kwargs):
    chunk_size = kwargs['chunk_size']
    labels = []
    word_idxs = []
    for num, sample in enumerate(batch):
        if rate_type == 'c_rate':
            score = int(sample[1])
        else:
            score = int(sample[2]) * 10

        tokens = [TOKEN_CHANGES.get(i, i) for i in tokenizer(sample[3])]
        for i in range(0, len(tokens) // chunk_size + 1):
            labels.append(score)
            curr_tokens = tokens[i * chunk_size : i * chunk_size + chunk_size]
            token_idxs = [vocab[token] for token in curr_tokens]
            word_idxs.append(torch.LongTensor(token_idxs))
    word_idxs = pad_sequence(word_idxs, batch_first = False, padding_value = PAD)
    return torch.tensor(labels, dtype = torch.float), word_idxs

def collate_rnn_whole(batch, vocab, rate_type, **kwargs):
    labels = make_labels_tensor(batch, rate_type)
    word_idxs = []
    for num, sample in enumerate(batch):
        tokens = [TOKEN_CHANGES.get(i, i) for i in tokenizer(sample[3])]
        token_idxs = [vocab[token] for token in tokens]
        word_idxs.append(torch.LongTensor(token_idxs))
    word_idxs = pad_sequence(word_idxs, batch_first = False, padding_value = PAD)
    return labels, word_idxs

# ==============
# B. PyTorch Modules
# ==============
class BoWClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_dim = 100, nonlinearity = nn.ReLU(), use_cuda = False, **kwargs):
        super(BoWClassifier, self).__init__()
        self.use_cuda = use_cuda
        self.nonlinearity = nonlinearity
        self.linear1 = nn.Linear(vocab_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.char = False
        
    def forward(self, bow_vec, **kwargs):
        if self.use_cuda:
            bow_vec = bow_vec.cuda()
        out = self.nonlinearity(self.linear1(bow_vec))
        return self.linear2(out).unsqueeze(2)

class CBoWClassifier(nn.Module):
    def __init__(self, 
                vocab_size, 
                embedding_dim,
                hidden_dim = 100, 
                nonlinearity = nn.ReLU(),
                use_cuda = False,
                use_glove = False,
                glove_vectors = None,
                freeze_glove = False,
                **kwargs):
        super(CBoWClassifier, self).__init__()
        print()
        print("Model called is a CBOW")
        print(f"CBOW hidden size is {hidden_dim}")
        self.use_cuda = use_cuda
        self.char = False
        
        if use_glove:
            self.embeddings = nn.Embedding.from_pretrained(embeddings = glove_vectors, freeze = freeze_glove)
        else:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        self.nonlinearity = nonlinearity
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, input, **kwargs):
        if self.use_cuda:
            input = input.cuda()
        embeds = self.embeddings(input)
        embeds = torch.mean(embeds, dim = 0)
        out = self.nonlinearity(self.linear1(embeds))
        return self.linear2(out).unsqueeze(2)

class CharacterRNN(nn.Module):
    def __init__(self, 
                char_vocab_size,
                embedding_dim,
                hidden_dim):
        super(CharacterRNN, self).__init__()
        self.embeddings = nn.Embedding(char_vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim)
    
    def forward(self, input, **kwargs):
        embeds = self.embeddings(input)
        rnn_out, hidden = self.rnn(embeds)
        return hidden[0].squeeze(0).unsqueeze(1 )

class RNNClassifier(nn.Module):
    def __init__(self,
                 rnn_type,
                 vocab_size,
                 embedding_dim,
                 hidden_dim,
                 num_layers,
                 bidirectional,
                 char_vocab_size = None,
                 char = False,
                 dropout = 0.5,
                 use_cuda = False,
                 use_glove = False,
                 glove_vectors = None,
                 freeze_glove = False):
        super(RNNClassifier, self).__init__()
        print("The model called is an RNN with the following properties:")
        print(f"RNN Type: {rnn_type}\nVocab Size: {vocab_size}\nEmbedding Size: {embedding_dim}")
        print(f"Hidden Size: {hidden_dim}\nNumber Layers: {num_layers}\nBidirectional: {bidirectional}")
        print(f"Use GloVE Embeddings: {use_glove}")
        print(f"Character-level RNN: {char}")
        self.use_cuda = use_cuda
        self.char = char
        self.bidirectional = bidirectional
        if use_glove:
            self.embeddings = nn.Embedding.from_pretrained(embeddings = glove_vectors, freeze = freeze_glove)
        else:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        rnn_dict = {'LSTM' : nn.LSTM, 'GRU' : nn.GRU}
        self.rnn_type = rnn_type
        
        self.rnn = rnn_dict[rnn_type](embedding_dim, hidden_dim, num_layers = num_layers, dropout = dropout, bidirectional = bidirectional)

        self.f_char = None
        self.b_char = None
        if self.char:
            self.f_char = CharacterRNN(char_vocab_size, embedding_dim, embedding_dim)
            self.b_char = CharacterRNN(char_vocab_size, embedding_dim, embedding_dim)
            
        
        directions = 2 if bidirectional else 1
        self.hidden2score = nn.Linear(hidden_dim * directions, 1)
      
    def forward(self, input, **kwargs):
        chars_f = kwargs.get('chars_f', None)
        chars_b = kwargs.get('chars_b', None)
        embeds = self.embeddings(input)
        if self.char:
            charsf = self.f_char(chars_f)
            charsb = self.b_char(chars_b)
            print("Concatenating with character-level RNN output...")
            embeds = torch.cat((embeds, charsf, charsb))
        rnn_out, hidden = self.rnn(embeds)
        if self.bidirectional:
            hidden = torch.cat((hidden[0][-2,:,:], hidden[0][-1,:,:]), dim = 1).unsqueeze(0)
            # print(f"Hidden shape after concat: {hidden.shape}")
        else:
            hidden = hidden[0][-1, :, :].unsqueeze(0)
        score = self.hidden2score(hidden)
        return score
        
# ==============
# C. Other functions
# ==============
def train_an_epoch(dataloader, model, optimizer, clip, use_cuda):
    loss_function = nn.MSELoss()

    model.train() # Sets the module in training mode.
    for idx, data in enumerate(dataloader):
        label = data[0]
        text = data[1]
        chars_f = None
        chars_b = None
        if len(data) > 2:
            chars_f = data[2]
            chars_b = data[3]
        
        model.zero_grad()
        if use_cuda:
            label = label.cuda()
            text = text.cuda()
            if len(data) > 2:
                chars_f = chars_f.cuda()
                chars_b = chars_b.cuda()  
        log_probs = model(text, chars_f = chars_f, chars_b = chars_b)
        loss = loss_function(log_probs, torch.reshape(label, (log_probs.shape[0], log_probs.shape[1], -1)))
        if idx % 32 == 0 or model.char:
            print(f"{idx}: {text.shape}")
            print(loss)
            print()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

def get_accuracy(dataloader, model, info, use_cuda):
    model.eval()
    with torch.no_grad():
        denom = 0
        labels = torch.empty(0)
        vals = torch.empty(0)

        if use_cuda:
            labels = labels.cuda()
            vals = vals.cuda()

        for idx, data in enumerate(dataloader):
            label = data[0]
            text = data[1]
            chars_f = None
            chars_b = None
            if len(data) > 2:
                chars_f = data[2]
                chars_b = data[3]
            if use_cuda:
                model = model.cuda()
                label = label.cuda()
                text = text.cuda()
                if len(data) > 2:
                    chars_f = chars_f.cuda()
                    chars_b = chars_b.cuda()  
            
            val = model(text, chars_f = chars_f, chars_b = chars_b).view(label.shape)
            labels = torch.cat((labels, label), 0)
            vals = torch.cat((vals, val), 0)
            denom += len(val)
        base = torch.sum((labels - torch.mean(labels)) ** 2)
        explained = torch.sum((labels - vals) ** 2)
        my_r2 = 1 - explained / base

        labels_np = labels.cpu().detach().numpy()
        vals_np = vals.cpu().detach().numpy()
        skl_r2 = r2_score(labels_np, vals_np)
        
        print(f"-Over {denom} observations in dataset...")
        print(f"-Total sum of squared errors is {explained}.")
        print(f"    -MSE is {explained/denom}")
        print(f"-Total sum of squared deviations from mean is {base}")
        print(f"    -Variance is {base/denom}")
        print(f"-My calculated R^2 value is {my_r2.item()}")
        print(f"-sklearn's R^2 value is {skl_r2.item()}")
        with open('model_int_results.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(info + [(explained / denom).item(), (base / denom).item(), my_r2.item(), skl_r2] )        
        return (explained / denom).item(), (base / denom).item(),  my_r2.item(), skl_r2.item()

# ==============
# D. General Template
# ==============


# class KFoldData:
#     def __init__(self, a, b, c, methodology):
#         self.a = a
#         self.b = b
#         self.c = c
#         self.methodology = methodology
#         self.state = 0
#         # self.path = f'valid_folds_{methodology}_{wd_albs}.pickle'
        
#     def create_kfold_datasets(self):
#         self.state += 1
#         if self.state % 3 ==  1:
#             train = torch.utils.data.ConcatDataset([self.a, self.b])
#             print("Test is made up of c")
#             test = self.c
#         elif self.state % 3 ==  2:
#             train = torch.utils.data.ConcatDataset([self.a, self.c])
#             test = self.b
#             print("Test is made up of b")
#         elif self.state % 3 == 0:
#             train = torch.utils.data.ConcatDataset([self.b, self.c])
#             test = self.a
#             print("Test is made up of a")
#         return train, test   
    
class NlpModel():
    def __init__(self, train_collate, test_collate, train_fx, get_accuracy, model_type, model_kwargs, info, use_cuda, *args):
        self.train_collate = train_collate
        self.test_collate = test_collate
        self.train_fx = train_fx
        self.get_accuracy = get_accuracy
        self.model = model_type(**model_kwargs, use_cuda = use_cuda)
        self.info = info

        self.use_cuda = use_cuda

        self.model_type = model_type
        self.model_kwargs = model_kwargs
        
        
    def runModel(self, optimizer, train_dataloader, valid_dataloader, clip, epochs):
        model = self.model
        if self.use_cuda:
            model = model.cuda()
        
        accuracies = []
        best_model = None
        best = None
        best_epoch = 0
        for epoch in range(1, epochs + 1):
            self.train_fx(train_dataloader, model, optimizer, clip, self.use_cuda)
            info = self.info + [epoch]
            print(f"Results after epoch {epoch}")
            accuracy = self.get_accuracy(valid_dataloader, model, info, self.use_cuda)
            print("Yo this is the accuracy")
            print(accuracy)
            print("And this is accuracy[2]")
            print(accuracy[2])
            accuracies.append(accuracy[2])
            print("This is all of accuracies")
            print(accuracies)
            
            print("This is the max of accuracies")
            print(max(accuracies))
            
            if best is None:
                best = accuracy[2]
                best_epoch = 1
            
            if len(accuracies) == 1 or ((epoch > epochs *.4) and accuracy[2] >= best):
                best = accuracy[2]
                best_epoch = epoch
                print(f"New best accuracy is now {accuracy[2]}, picked at epoch {best_epoch}")
                best_model = type(model)(**self.model_kwargs) # get a new instance
                best_model.load_state_dict(model.state_dict())
                print("New best model has been selected")
                print()
            
            print("But this is best")
            print(best)

        return accuracy[0], accuracy[1], best_model, best, best_epoch


    # def datasets_3fold(self, reg_albums):
    #     third = len(reg_albums) // 3
    #     a, big = random_split(reg_albums, [third, len(reg_albums) - third])
    #     b, c = random_split(big, [third, len(big) - third])
    #     return a, b, c

    # def create_datasets_k(self, reg_albums, methodology, wd_albs):
    #     try:
    #         with open(f'valid_folds_{methodology}_{wd_albs}.pickle', 'rb') as handle:
    #             data = pickle.load(handle)   
    #     except:
    #         a, b, c = self.datasets_3fold(reg_albums)
    #         data = KFoldData(a, b, c, methodology)
        
    #     train, test = data.create_kfold_datasets()
    #     with open(f'valid_folds_{methodology}_{wd_albs}.pickle', 'wb') as handle:
    #         pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     return train, test

    def create_datasets(self, reg_albums, test_reg_albums, final):
        print("Creating datasets right now...")
        # if valid:
            # self.create_datasets_k(reg_albums, methodology, wd_albs)
        prop = 0.8

        if final:
            print("Running on 'final'")
            print(f"Final train set is {len(reg_albums)}")
            num_train = int(len(reg_albums) * .9)    
            num_valid = len(reg_albums) - num_train
            train_data, valid_data = random_split (reg_albums, [num_train, num_valid])
            print(f"Train is size: {len(train_data)}, valid is size: {len(valid_data)}, test is size: {len(test_reg_albums)}")
            return train_data, valid_data, test_reg_albums

        num_train_valid = int(len(reg_albums) * prop)
        num_test = len(reg_albums) - num_train_valid
        train_valid_data, test_data = random_split(reg_albums, [num_train_valid, num_test])

        num_train = int(num_train_valid * .9)
        num_valid = num_train_valid - num_train
        train_data, valid_data = random_split(train_valid_data, [num_train, num_valid])

        print(f"Train samples: {num_train}\nValid samples: {num_valid}\nTest samples: {num_test}")
        return train_data, valid_data, test_data
    
    def collate_datasets(self, train_data, valid_data, test_data, batch_size, test_batch_size, valid, **kwargs):
        train_collate_fn = partial(self.train_collate, **kwargs)
        test_collate_fn = partial(self.test_collate, **kwargs)
        
        print("Creating training dataset...")
        train_dataloader = DataLoader(train_data,
                                        batch_size=batch_size,
                                        shuffle=True, 
                                        collate_fn=train_collate_fn)
        print("Creating validation dataset...")
        valid_dataloader = DataLoader(valid_data, 
                                        batch_size=test_batch_size,
                                        shuffle=False, 
                                        collate_fn=test_collate_fn)
        # if valid:
        #     return train_dataloader, valid_dataloader, None
        
        print("Creating test dataset...")
        test_dataloader = DataLoader(test_data, 
                                        batch_size=test_batch_size,
                                        shuffle=False, 
                                        collate_fn=test_collate_fn)
        return train_dataloader, valid_dataloader, test_dataloader
