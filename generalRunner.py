"""
Note: this is .py equivalent of generalRunner.ipynb
"""
import sys
import os
import torch
import csv
import argparse
from functools import partial
import itertools
import uuid
import pickle
import numpy as np
        
from torchtext.data.utils import get_tokenizer
tokenizer = get_tokenizer('basic_english')
from torchtext.vocab import Vocab, build_vocab_from_iterator
from torchtext.vocab import GloVe
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataset import random_split
from torch.nn.utils.rnn import pad_sequence
import time
import importlib
from nltk.corpus import stopwords
from torch.utils.data.dataset import random_split



# ===============
# i. Colab
# ===============
COLAB = False

USE_CUDA = False
if COLAB:
    from google.colab import drive 
    drive.mount('/content/gdrive')
    PATH = 'gdrive/MyDrive/nlp22/project/'
    sys.path.append('gdrive/MyDrive/nlp22/project')

    USE_CUDA = torch.cuda.is_available()

    if USE_CUDA:
        DEVICE = torch.device('cuda')
        print("Using cuda.")
    else:
        DEVICE = torch.device('cpu')
        print("Using cpu.")

    os.chdir(os.path.join(os.getcwd(),'gdrive/MyDrive/nlp22/project'))
    
# ===============
# ii. Imports
# ===============
from album_loader import *
import lyric_loader
import nlpmodel
importlib.reload(nlpmodel)

# ===============
# iii. Constants
# ===============
VECTORS_CACHE_DIR = './.vector_cache'

UNK, PAD, LBS, LBE, SBS, SBE, PART = 0, 1, 2, 3, 4, 5, 6
FIRST_TOKENS = 5000
STRATEGY = f'FIRST {FIRST_TOKENS} - Embeddings On'
EMBEDDING_DIMENSIONS = 300

VECTORS_CACHE_DIR = './.vector_cache' # or modify to the correct location

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

RATE_TYPE = 'c_rate'

# ===========
# 1. Set-up functions
# ===========
def init_albums(path, file, standardize_parts, see_lbs, u_rate_min = 15):
    """
    Instantiates a set of Albums for regression purposes

    kwargs:
    file  -- file containing Albums info (albums_f.pickle)
    standardize_parts - signals whether to standardize parts in lyrics
    see_lbs - signals whether to see linen breaks in lyrics
    u_rate_min - minimium user ratings to be included in data set
    """
    print("Init albums was called...")
    albums_data = os.path.join(path, file)
    albums_pre = lyric_loader.RegAlbums(album_path = albums_data, 
                                        standardize_parts = standardize_parts, 
                                        see_line_breaks = see_lbs,
                                        u_rate_min = u_rate_min)
    reg_albums = albums_pre.reg_full_album_text() 
    return reg_albums

def weed_albums_fx(reg_albums, exc_sds):
    """
    Excludes a portion of medium-rated albums

    kwargs:
    reg_albums -- regression albums
    exc_sds -- how many standard deviations to exclude
    """
    a = np.array([int(i[1]) for i in reg_albums])
    
    denom = exc_sds * 2
    lo = a.mean() - (np.std(a) / denom)
    hi = a.mean() + (np.std(a) / denom) 
    test = [i for i in reg_albums if int(i[1]) < lo or int(i[1]) > hi]
    print(f"Albums reduced from {len(reg_albums)} to {len(test)}")
    return test

def yield_tokens(data_iter, chars):
    if chars:
        for _, _, _, text in data_iter:
            for i in text:
                if i == ' ':
                    continue
                yield i
    else:
        for _, _, _, text in data_iter:
            res = [TOKEN_CHANGES.get(i, i) for i in tokenizer(text)]
            yield res

def set_vocab(reg_albums, min_freq, chars, treat_stops):
    specials = ['<unk>', '<pad>', '<lb>', '</lb>', '<sb>', '</sb>', '[part]', '[sw]']
    yield_fx = yield_tokens
    if chars:
        specials = ['<unk>', '<pad>']
    vocab = build_vocab_from_iterator(yield_fx(reg_albums, chars = chars),
                specials = specials, min_freq = min_freq)
    vocab.set_default_index(vocab['<unk>'])
    return vocab

def create_train_test(standardize_parts, see_lbs):
    """
    Create separate train and test datasets
        according to pre-processing required
    
    kwargs:
    standardize_parts -- signals whether to standardize parts in lyrics
    see_lbs -- signals whether to see linen breaks in lyrics    
    """
    print("Create train test was called...")
    sp = 1 if standardize_parts else 0
    sl = 1 if see_lbs else 0
    u_rate_min = 10
    reg_albums = init_albums(path = '', file = 'albums_f.pickle', 
                standardize_parts = standardize_parts, see_lbs = see_lbs, u_rate_min = u_rate_min)

    fourth = len(reg_albums) // 4
    train_val, test = random_split(reg_albums, [len(reg_albums) - fourth, fourth])

    comb = (train_val, test)
    with open(f'train_val_test_{sp}_{sl}.pickle', 'wb') as handle:
        pickle.dump(comb, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return comb

def load_train_test(standardize_parts, see_lbs):
    """
    Load train and test datasets
    according to pre-processing required. Will
    create if not already created.
    
    kwargs:
    standardize_parts -- signals whether to standardize parts in lyrics
    see_lbs -- signals whether to see linen breaks in lyrics    
    """
    sp = 1 if standardize_parts else 0
    sl = 1 if see_lbs else 0
    try:
        with open(f'train_val_test_{sp}_{sl}.pickle', 'rb') as handle:
            comb = pickle.load(handle)   
    except:
        print(f"Creating train/val/test sets for standardize_parts: {standardize_parts}, see_lbs: {see_lbs}")
        comb = create_train_test(standardize_parts, see_lbs)
    train, test = comb
    return train, test

def create_datasets(reg_albums, test_reg_albums, final):
    """
    Create train, val and test data sets for THIS run of the model

    kwargs:
    reg_albums -- training data from load_train_test()
    test_reg_albums -- test data from load_train_test()
    final - dictates how train,test,val will be created
        - if True, test_reg_albums ("actual" test set) will be 
            used as test set
        - if False, assume hyperparameter tuning, so a portion
            of the train_val dataset will be cut out to
            evaluate parameter
    """
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

# =============
# 2. Main Function
# =============
def main(methodologies,
        methodology = 0,
        weed_albums = False,
        standardize_parts = True, 
        see_lbs = True,
        treat_stops = 'see',
        opt = 'adam', 
        lr = 0.005, 
        wd = 0,  
        chunk_size = 2500,
        chunk_portion = 'first',
        epochs = 10,
        batch_size = 16,
        test_batch_size = 16,
        clip = 1,
        rnn_type = 'LSTM',
        embedding_size = 300,
        use_glove = True, 
        bidirectional = False, 
        num_layers = 1,
        dropout = 0,
        use_cuda = USE_CUDA,
        rate_type = 'c_rate',
        valid = False,
        final = False,
        kfoldername = ''
       ):
    """
    Runs model from end to end, from importing regression albums
    to training, evalluation, and saving results
    """

    id = uuid.uuid4()
    char = False
    if methodology == 7:
        char = True
    if methodology == 8:
        chunk_size = 'all'

    methodology_name = methodologies[methodology][0]
    hidden_size = int(embedding_size * 2)
    if embedding_size > 300:
        use_glove = False
    print(f"Running following model:")
    print(f"Methodology: {methodology_name}\nWeed albums:{weed_albums}")
    print(f"\nOptimizer: {opt}\nLearning Rate: {lr}")
    print(f"Epochs: {epochs}\nEmbedding size: {embedding_size}\nHidden size: {hidden_size}")
    print(f"Batch size: {batch_size},\nTest batch size: {test_batch_size}")
    print(f"Bidirectional: {bidirectional},\nNumber of layers: {num_layers}")
    print(f"Character-level RNN included: {char}")
    print(f"Standardize_parts: {standardize_parts}\nSee line breaks: {see_lbs}\nTreat stops: {treat_stops}")
    print(f"Chunking portion used : {chunk_portion}")
    print(f"This is treated as validation verison? {valid}")
    print(f"This is treated as final verison? {final}")
    
    reg_albums, test_reg_albums = load_train_test(standardize_parts, see_lbs)
    print(f"Working with {len(reg_albums)} albums in total for reg_albums...")
    if final:
        print(f"Working with {len(test_reg_albums)} albums in total for test_reg_albums...")
    
    wd_albs = 0
    if weed_albums:
        reg_albums = weed_albums_fx(reg_albums, 1)
        wd_albs = 1

    train_dataset, valid_dataset, test_dataset = create_datasets(reg_albums, 
                                                                    test_reg_albums,
                                                                    final)
    
    train_text = train_dataset
    vocab = None
    glove_vectors = None
    char_vocab = None
    vocab = set_vocab(train_text, min_freq = 10, chars = False, treat_stops = treat_stops)
    if methodology > 0:
        glove = GloVe('6B',cache=VECTORS_CACHE_DIR)
        glove_vectors = glove.get_vecs_by_tokens(vocab.get_itos())
    if methodology == 7:
        char_vocab = set_vocab(reg_albums, min_freq = 1000, chars = True)
    
    print("Finished loading vocabs and GLoVE")

    vocab_size = len(vocab) if vocab is not None else 0 
    char_vocab_size = len(char_vocab) if char_vocab is not None else 0
    rnn_kwargs =  {'rnn_type': rnn_type, 
                    'vocab_size': vocab_size,
                    'char_vocab_size' : char_vocab_size,
                    'embedding_dim': embedding_size, 
                    'hidden_dim': hidden_size,
                    'num_layers': num_layers,
                    'bidirectional' : bidirectional,
                    'char' : char,
                    'use_glove' : use_glove,
                    'glove_vectors' : glove_vectors,
                    'freeze_glove' : False,
                    'dropout': 0}

    strategy = methodologies[methodology][0]
    info_pre = [id, methodology, strategy, standardize_parts, see_lbs, treat_stops, opt, lr, wd, chunk_size, chunk_portion,
                         epochs, batch_size, test_batch_size, hidden_size, use_glove, bidirectional, num_layers, 
                         rate_type, char, weed_albums]

    model_template = nlpmodel.NlpModel(*methodologies[methodology][1:], 
                                        rnn_kwargs, 
                                        info_pre,
                                        use_cuda = USE_CUDA)
    model = model_template.model

    print("Initialized model")

    optim_dict = {'adam' : torch.optim.Adam, 'sgd' : torch.optim.SGD}
    try:
        optimizer = optim_dict[opt.lower()](model.parameters(), lr = lr, weight_decay = wd)
    except:
        print("Please select a valid optimizer! Specify either 'adam' or 'sgd'.")

    print("Initialized optimizer")

    collate_kwargs = {'vocab' : vocab, 
                    'rate_type' : rate_type,
                    'char_vocab' : char_vocab, 
                    'chunk_size' : chunk_size,
                    'chunk_portion' : chunk_portion,
                    'treat_stops' : treat_stops}

    train_dataloader, valid_dataloader, test_dataloader = model_template.collate_datasets(train_dataset,
                                                                                    valid_dataset,
                                                                                    test_dataset,
                                                                                    batch_size,
                                                                                    test_batch_size,
                                                                                    valid,   
                                                                                    **collate_kwargs)
    print("Initialized dataloaders. Running models now...")
    last_mse, last_var, best_model, best_r2, best_epoch = model_template.runModel(optimizer, train_dataloader, valid_dataloader, clip, epochs)
    
    link = 'model_results_valid.csv'
        
    test_mse_new, test_var_new, my_r2, skl_r2  = None, None, None, None

    if final:    
        link = 'model_results_final.csv'
        torch.save(best_model, f'bestModel - {methodology}.pt')
        print("Here are results on final dataset...")
    test_mse_new, test_var_new, my_r2, skl_r2  = model_template.get_accuracy(test_dataloader, best_model, 
                                                                                    info_pre, model_template.use_cuda)

    with open(link, 'a') as csvfile:
        print(f"Writing to {link}")
        writer = csv.writer(csvfile)
        writer.writerow(info_pre + [last_mse, last_var, test_mse_new, test_var_new, my_r2, skl_r2, best_r2, best_epoch])

if __name__ == '__main__':
    """
    Define inputs to main function here

    - Methodology must be integer from 0-8,
    indicating which model to apply
    - 0, 1, 2, 3, and 8 are in paper
    """

    methodologies= {0 : ('BOW', nlpmodel.collate_into_bow, 
                                nlpmodel.collate_into_bow, 
                                nlpmodel.train_an_epoch, 
                                nlpmodel.get_accuracy, 
                                nlpmodel.BoWClassifier),
                1 : ('CBOW - unfrozen', 
                                nlpmodel.collate_into_cbow_unfrozen, 
                                nlpmodel.collate_into_cbow_unfrozen,
                                nlpmodel.train_an_epoch, 
                                nlpmodel.get_accuracy, 
                                nlpmodel.CBoWClassifier),
                2 : ('RNN - album firstk',  
                                nlpmodel.collate_batch_rnn_firstk, 
                                nlpmodel.collate_batch_rnn_firstk,
                                nlpmodel.train_an_epoch,
                                nlpmodel.get_accuracy, 
                                nlpmodel.RNNClassifier),
                3 : ('RNN - song firstk',  
                                nlpmodel.collate_batch_rnn_firstk_song, 
                                nlpmodel.collate_batch_rnn_firstk_song, 
                                nlpmodel.train_an_epoch,
                                nlpmodel.get_accuracy, 
                                nlpmodel.RNNClassifier),
                4: ('RNN - album firstk - test whole',  
                                nlpmodel.collate_batch_rnn_firstk, 
                                nlpmodel.collate_rnn_whole,
                                nlpmodel.train_an_epoch,
                                nlpmodel.get_accuracy, 
                                nlpmodel.RNNClassifier),
                5 : ('RNN - song firstk - test whole',  
                                nlpmodel.collate_batch_rnn_firstk_song, 
                                nlpmodel.collate_rnn_whole, 
                                nlpmodel.train_an_epoch,
                                nlpmodel.get_accuracy, 
                                nlpmodel.RNNClassifier),
                6 : ('RNN - chunk - test whole',
                                nlpmodel.collate_batch_chunk,
                                nlpmodel.collate_rnn_whole,
                                nlpmodel.train_an_epoch,
                                nlpmodel.get_accuracy,
                                nlpmodel.RNNClassifier),             
                7 : ('RNN - character',
                                nlpmodel.collate_batch_chars,
                                nlpmodel.collate_batch_chars,
                                nlpmodel.train_an_epoch,
                                nlpmodel.get_accuracy,
                                nlpmodel.RNNClassifier),
                8 : ('RNN - album all',  
                                nlpmodel.collate_batch_rnn_firstk, 
                                nlpmodel.collate_batch_rnn_firstk,
                                nlpmodel.train_an_epoch,
                                nlpmodel.get_accuracy, 
                                nlpmodel.RNNClassifier)
                }

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-methodology', type = int, default = 0, help = 'Methodology')
    parser.add_argument('-weed_albums', nargs = '?', const = '', help = 'If turned on, will weed out middle albums')
    parser.add_argument('-learn_rate', type = float, default = 0.015, help = 'Learning rate for optimizer')
    parser.add_argument('-optimizer', type = str, default = 'adam', help = 'Name of optimizer to use')
    parser.add_argument('-weight_decay', type = float, default = 0, help = 'Weight decay (l2 reg rate) applied')
    parser.add_argument('-no_parts', nargs = '?', const = '',  help = "Don't identify parts")
    parser.add_argument('-no_lb', nargs = '?', const = '',  help = "Don't identify linebreaks")
    parser.add_argument('-no_glove', nargs = '?', const = '',  help = "Don't use glove embeddings")
    parser.add_argument('-epochs', type = int, default = 20, help = 'Iterations through training data')
    parser.add_argument('-chunk_size', type = int, default = 2500, help = 'Size of chunk processed')
    parser.add_argument('-batch_size', type = int, default = 16, help = 'Batch size')
    parser.add_argument('-test_batch_size', type = int, default = 16, help = 'Test batch size')
    parser.add_argument('-embedding_size', type = int, default = 300, help = 'Embedding dimensions')
    parser.add_argument('-bidirectional', type = bool, default = False, help = 'Bidirectional RNN')
    parser.add_argument('-num_layers', type = int, default = 1, help = 'Layers in RNN')
    parser.add_argument('-treat_stops', type = str, default = None, help = 'How to treat stop words')
    parser.add_argument('-chunk_portion', type = str, default = 'first', help = 'Which portion to chunk')
    parser.add_argument('-valid', nargs = '?', const = '', help = 'Is this validation version?')
    parser.add_argument('-final', nargs = '?', const = '', help = 'Is this final version?')

    args = parser.parse_args()

    methodology = args.methodology
    weed_albums = False if args.weed_albums is None else True
    standardize_parts = True if args.no_parts is None else False
    see_lbs = True if args.no_lb is None else False
    treat_stops = args.treat_stops
    
    opt = args.optimizer
    lr = args.learn_rate
    wd = args.weight_decay
    chunk_size = args.chunk_size
    chunk_portion = args.chunk_portion
    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    epochs = args.epochs
    clip = 1

    rnn_type = 'LSTM'
    embedding_size = args.embedding_size
    use_glove = True if args.no_glove is None else False
    bidirectional = args.bidirectional
    num_layers = args.num_layers
    dropout = 0

    rate_type = RATE_TYPE

    valid = False if args.valid is None else True
    final = False if args.final is None else True


    main_kwargs = {'methodology' : methodology,
                'weed_albums' : weed_albums,
                'standardize_parts' : standardize_parts,
                'see_lbs' : see_lbs,
                'treat_stops' : treat_stops,
                'opt' : opt,
                'lr' : lr,
                'wd' : wd,
                'chunk_size' : chunk_size,
                'batch_size' : batch_size,
                'test_batch_size' : test_batch_size,
                'epochs' : epochs,
                'clip' : 1,
                'rnn_type': rnn_type, 
                'embedding_size' : embedding_size,
                'use_glove' : use_glove,
                'bidirectional' : bidirectional,
                'num_layers' : num_layers,
                'dropout' : dropout,
                'use_cuda' : USE_CUDA,
                'rate_type' : rate_type,
                'valid' : valid,
                'final' : final}
    
    # for parts in [(True, False), (True, True)]:
    for i in range(1):
        for lr in [0.00025, 0.0005, 0.001]:
            main_kwargs['lr'] = lr
            main(methodologies, **main_kwargs)
        
        for treat in ['see',' remove', None]:
            main_kwargs['treat_stops'] = treat
            main(methodologies, **main_kwargs)
