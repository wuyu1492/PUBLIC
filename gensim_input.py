from gensim.models import Word2Vec

label_path = "./training_label.txt"
nolabel_path = "./training_nolabel.txt"
test_path = "./testing_data.txt"

def make_sen(label_path, unlabel_path, test_path):
    sentences = []
    with open(label_path,'r') as f:
        print("Processing label file...")
        for line in f:
            line = ''.join(ch for ch in line if ch.isalpha() or ch == ' ')
            sen = line.strip().split()
            sentences.append(sen)

    with open(nolabel_path,'r') as f:
        print("Processing nolabel file...")
        for line in f:
            line = ''.join(ch for ch in line if ch.isalpha() or ch == ' ')
            sen = line.strip().split()
            sentences.append(sen)

    with open(test_path,'r') as f:
        print("Processing test file...")
        for line in f:
            line = ''.join(ch for ch in line if ch.isalpha() or ch == ' ')
            sen = line.strip().split()
            sentences.append(sen)
    return sentences

sentences = make_sen(label_path, nolabel_path, test_path)
    
print("Building model...")
model = Word2Vec(sentences, size=128, window=5, min_count=5)
print(model)

model.save('word2vec.bin')

#new_model = Word2Vec.load('word2vec.bin')
