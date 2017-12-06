import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from gensim.models import Word2Vec
import tqdm

def sen2vec(sen, model, vocab):
    sen = [[ float(num) for num in model[w]] for w in sen if w in vocab]

    return sen


def build(data_path, vocab_path):
    x_data=[]
    y_data=[]
    max_len = 0
    with open(data_path, 'r') as f:
        vocab_model = Word2Vec.load(vocab_path)
        vocab = list(vocab_model.wv.vocab)
        print(vocab_model)
       
        for line in f:
            label = int(line[0])    # reading labeled data file 
            line = ''.join([ch for ch in line if ch.isalpha() or ch==' ']).strip().split()
            x = sen2vec(line, vocab_model, vocab)
            max_len = max(max_len, len(x))
            #print(max_len)
            x_data.append(x)
            y_data.append(label)
            if label > 1:
                print("label value error!!!")
    total_len = len(y_data)
    embed_dim = len(x_data[0][0]) 
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    print(x_data.shape)
    np.save("./gensim_x.npy", np.array(x_data))
    np.save("./gensim_y.npy", np.array(y_data))
    

class Dataset(Dataset):
    def __init__(self):
        print("Preparing dataset")
        self.y = []
        self.x = []
        self.max_len = 0
        self.total_len = 0
        self.embed_dim = 0
        self.build()

    def __len__(self):
        return self.total_len

    def build(self):
        self.x = np.load("./gensim_x.npy").tolist()
#       for j in range(len(self.x)):
#            for i in range(len(self.x[j])):
#                w = self.x[j][i]
#                self.x[j][i] = [float(num) for num in w]
        self.y = np.load("./gensim_y.npy").tolist()
        
        for sen in self.x:
            self.max_len = max(self.max_len, len(sen))
        self.total_len = len(self.y)
        self.embed_dim = len(self.x[0][0])


    def __getitem__(self, index):
        length = len(self.x[index])
        return torch.FloatTensor([[0.0]*self.embed_dim]*(self.max_len-length) + self.x[index]), \
                torch.LongTensor([self.y[index]])

import random 
from math import floor

def train_valid_split(dataset, test_size=0.25, shuffle=False, random_seed=0):
    length = dataset.__len__()
    indices = list(range(1, length))
    if shuffle == True:
        random.seed(random_seed)
        random.shuffle(indices)
    if type(test_size) is float:
        split = floor(test_size*length)
    elif type(test_size) is int:
        split = test_size
    else:
        raise ValueError('%s should be an int or a float'%str)
    return indices[split:], indices[:split]


def main():
    vocab_path = './word2vec.bin'
    dataset = Dataset()
    build('./training_label.txt', vocab_path)
    #pickle.dump(dataset, open("dataset_train.pkl", "wb"))

if __name__ == "__main__":
    main()
