import torch
import os

class Dictionary(object):
    """
        Dictionary class provides a
        word-to-id dictionary as well as 
        a id-to-word dictionary 
        for words added with add_word()
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __len__(self):
        return len(self.word2idx)
    
class Corpus(object):
    def __init__(self, path='./data'):
        self.dictionary = Dictionary()
        self.train = os.path.join(path, 'train.txt')
        self.test = os.path.join(path, 'test.txt')

    def get_data(self, path, batch_size=20):
        # Add words to the dictionary
        with open(path, 'r') as f:
            # number of tokens in corpus
            tokens = 0
            # iterating through lines
            for line in f:
                #getting words in line and adding '<eos>' tag
                words = line.split() + ['<eos>']
                # summing up tokens
                tokens += len(words)
                for word in words: 
                    # adding words to dictionary
                    self.dictionary.add_word(word)  
        
        # Tokenize the file content
        # text converted to ids as integers tensor
        ids = torch.LongTensor(tokens)
        token = 0
        # opening file again
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    # adding line as int to ids
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        # ids.size(0) = num of tokens
        # calculating number of batches
        num_batches = ids.size(0) // batch_size
        # deleting incomplete last batch
        ids = ids[:num_batches*batch_size]
        # reshaping for batches
        return ids.view(batch_size, -1)
