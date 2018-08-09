import dynet as dy
import numpy as np
import codecs
import scipy.stats as sp
import math

vec_dimension = 200
number_of_segmentation = 10
model = dy.Model()

morph_embeddings = model.add_lookup_parameters((63460, int(vec_dimension / 4)))  # randomly initialized

# input dim (morpheme embedding dimension): 200/4, output dimension: 300
fwdLSTM = dy.LSTMBuilder(1, int(vec_dimension / 4), vec_dimension, model)
bwdLSTM = dy.LSTMBuilder(1, int(vec_dimension / 4), vec_dimension, model)

# in order to reduce the segmentation dimension to half.
hidden_layer = model.add_parameters((vec_dimension, 2*vec_dimension))

# attention parameters: w-> for the hidden layer, v-> for the output layer
attention_w = model.add_parameters((vec_dimension, vec_dimension))
attention_v = model.add_parameters((1, vec_dimension))

model.populate('saved-model')

f_init = fwdLSTM.initial_state()
b_init = bwdLSTM.initial_state()

word2segmentations = {}
word2sgmt = {}
morph_indices = {}


def generate_word_vec(word):
    seg = word2sgmt[word]
    state_f = fwdLSTM.initial_state()
    state_b = bwdLSTM.initial_state()
    for s in range(len(seg)):
        morph = seg[s]
        morph_indice = morph_indices[morph[0]]
        state_f = state_f.add_input(morph_embeddings[int(morph_indice)])
        morph = seg[len(seg)-s-1]
        morph_indice = morph_indices[morph[0]]
        state_b = state_b.add_input(morph_embeddings[int(morph_indice)])

    bi=dy.concatenate([state_f.output(), state_b.output()])
    H = dy.parameter(hidden_layer)  # reduce the segmentation representation to its half.
    return H * bi  # H: hidden layer parameters


def read_gold_segmentations():
    f = codecs.open('turkish_new_data_gold_segmented.txt', encoding='utf-8')
    for line in f:
        line = line.rstrip('\n')
        word, sgmnts = line.split(':')
        sgmt = sgmnts.split('+')
        sgmt = sgmt[0].split('-')
        word2segmentations[word] = list(s for s in sgmt)
        sgmt = list(s.split('-') for s in sgmt)
        word2sgmt[word] = sgmt


def read_morph_indices():
    f = codecs.open('morphem-indices.txt', encoding='utf-8')
    for line in f:
        line = line.rstrip('\n')
        morph, indice = line.split(':')
        morph_indices[morph] = indice

def can_generate_word_vector(word):
    seg = word2sgmt[word]
    for s in range(len(seg)):
        morph = seg[s]
        if not morph[0] in morph_indices:
            return 0
    return 1


def evaluate_word_similarity():
    f = codecs.open('WordSimT.txt', encoding='utf-8')
    sim = []
    jud = []
    for line in f:
        line = line.rstrip('\n')
        wordpair, score = line.split('\t')
        word1, word2 = wordpair.split(':')
        if(can_generate_word_vector(word1)==1 and can_generate_word_vector(word2)==1):
            vector1 = generate_word_vec(word1)
            vector2 = generate_word_vec(word2) ## if both the word vectors could be generated
            sim.append(cal_cos(vector1, vector2))
            jud.append(float(score)/10)
    print(sp.spearmanr(sim, jud)[0])

def cal_cos(w1, w2):
    return dy.cdiv(dy.dot_product(w1, w2), (dy.l2_norm(w1) * dy.l2_norm(w2))).scalar_value()


def save_morpheme_vectors():
    fout = "morpheme-vectors.txt"
    fo = open(fout, "w")
    for k in morph_indices.keys():
        fo.write( k + ': ')
        fo.write( str(morph_embeddings[int(morph_indices[k])].vec_value()) + '\n')
    fo.close()

def read_morpheme_vectors():
    vocab = {}
    vectors = []

    with open("morpheme-vectors.txt") as f:
        f.readline()
        for i, line in enumerate(f):
            line = line.replace('[', '').replace(']', '').replace(',','')
            fields = line.strip().split(" ")
            vocab[fields[0]] = i
            vectors.append(list(map(float, fields[1:])))
    return vocab, vectors

read_gold_segmentations()
read_morph_indices()
evaluate_word_similarity()
save_morpheme_vectors()
vocab, vectors = read_morpheme_vectors()

lar = morph_indices['lar']
ler = morph_indices['ler']
print(dy.cdiv(dy.dot_product(morph_embeddings[int(lar)], morph_embeddings[int(ler)]), (dy.l2_norm(morph_embeddings[int(lar)])*dy.l2_norm(morph_embeddings[int(ler)]))).scalar_value())

#f_init = fwdLSTM.initial_state()
#b_init = bwdLSTM.initial_state()
