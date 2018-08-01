from __future__ import print_function
from __future__ import unicode_literals

import codecs
import sys

import dynet_config
import dynet as dy
import numpy as np
from numpy.linalg import norm
from keras.preprocessing import sequence
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Learn morph2vec: morpheme-based representation learning ')
    parser.add_argument('--input', type=str, required=True, help = "path to training file")
    parser.add_argument('--wordVector', type=str, required=True, help = "path to word2vec vector file")
    parser.add_argument('--output', '-o', type=str, required=True, help = "output directory where the weight file will be saved")
    parser.add_argument('--segNo', type=int, default=10, help = "number of segmentations provided for a given word during training")
    parser.add_argument('--batch', type=int, default=32,help = "batch size")
    parser.add_argument('--epoch', type=int, default=5,help = "number of epochs")
    parser.add_argument('--dim', type=int, default=200, help="dimension")

    args = parser.parse_args()

    number_of_segmentation = args.segNo
    gensim_model = args.wordVector
    training_file = args.input
    output_file =args.output
    batch_size = args.batch
    number_of_epoch = args.epoch
    dim = args.dim
    print('===================================  Prepare data...  ==============================================')
    print('')

    word2sgmt = {}
    word2segmentations = {}
    seq = []
    morphs = []

    f = codecs.open(training_file, encoding='utf-8')
    for line in f:
        line = line.rstrip('\n')
        word, sgmnts = line.split(':')
        sgmt = sgmnts.split('+')
        word2segmentations[word] = list(s for s in sgmt)
        sgmt = list(s.split('-') for s in sgmt)
        word2sgmt[word] = sgmt
        seq.extend(sgmt)

   # timesteps_max_len = 0

    for sgmt in seq:
        #if len(sgmt) > timesteps_max_len: timesteps_max_len = len(sgmt)
        for morph in sgmt:
            morphs.append(morph)

    print('number of words: ', len(word2sgmt))

    morph_indices = dict((c, i + 1) for i, c in enumerate(set(morphs)))
    morph_indices['###'] = 0

    indices_morph = dict((i+1, c) for i, c in enumerate(set(morphs)))

    print('number of morphemes: ', len(morphs))
    print('number of unique morphemes: ', len(set(morphs)))

    x_train = [[] for i in range(number_of_segmentation)]

    word_indices = {}
    index = 0
    for word in word2sgmt:
        word_indices[word] = index
        for i in range(len(word2sgmt[word])):
            x_train[i].append([morph_indices[c] for c in word2sgmt[word][i]])
        index = index+1

    for i in range(number_of_segmentation):
        x_train[i] = np.array(x_train[i])

#    for i in range(len(x_train)):
#        x_train[i] = sequence.pad_sequences(x_train[i], maxlen=timesteps_max_len)

    #print(x_train[0][0])
    #print(x_train[0][0][0])
    print('')
    print('==========================  Load pre-trained word vectors...  ======================================')
    print('')

    vocab = {}
    y_train = []

    with open("tvec.txt") as f:
        f.readline()
        for i, line in enumerate(f):
            fields = line.strip().split(" ")
            vocab[fields[0]] = i
            y_train.append(list(map(float, fields[1:])))

#    print('number of pre-trained vectors: ', len(w2v_model.vocab))

    print('number of pre-trained vectors: ', len(vocab)) # 've':1, 'kitap':2
    print('number of words found: ', len(y_train)) # [[f1,f2,....,f200],[f1,f2,....,f200],.....]
    y_train = np.array(y_train)
    print('shape of Y: ', y_train.shape)

    print('')
    print('===================================  Save Input and Output...  ===============================================')
    print('')

    np.save("x_train", x_train)
    np.save("y_train", y_train)

    print('')
    print('===================================  Build model...  ===============================================')
    print('')

    model = dy.Model()
    parameters = dy.ParameterCollection()

    morph_embeddings = model.add_lookup_parameters((len(set(morphs)), int(dim/4))) # randomly initialized

    def morph_rep(m): # returns the embedding of morpheme 'm' (m: morpheme index)
        #m_index = morph_indices[m]
        return morph_embeddings[m]

    # input dim (morpheme embedding dimension): 300/4, output dimension: 300
    fwdLSTM = dy.LSTMBuilder(1, int(dim/4), 300, model)
    bwdLSTM = dy.LSTMBuilder(1, int(dim/4), 300, model)

    # in order to reduce the segmentation dimension to half.
    hidden_layer = model.add_parameters((300, 600))

    # attention parameters: w-> for the hidden layer, v-> for the output layer
    attention_w = model.add_parameters((300, 300))
    attention_v = model.add_parameters((1, 300))
    trainer = dy.SimpleSGDTrainer(model)

    #build the LSTM for all segmentations of a single word
    def build_segmentation_graph(word_index, fwdLSTM, bwdLSTM):
        # iterate for each segmentation
        segmentation_outputs_f = []
        segmentation_outputs_b = []

        f_init = fwdLSTM.initial_state()
        b_init = bwdLSTM.initial_state()

        for i in range(number_of_segmentation):

            seg = x_train[i][word_index]
            # for each unit (i.e. segment) in the LSTM forward computation
            state_f = f_init
            state_b = b_init
            for s in range(len(seg)):
                state_f = state_f.add_input(morph_rep(seg[s]))
                state_b = state_b.add_input(morph_rep(seg[len(seg)-s-1]))

            segmentation_outputs_f.append(state_f.output())
            segmentation_outputs_b.append(state_b.output())
        bi = [dy.concatenate([f,b]) for f, b in zip(segmentation_outputs_f, segmentation_outputs_b)]
        return bi

    def attention(segmentation_outputs):

        w = dy.parameter(attention_w)
        v = dy.parameter(attention_v)
        # a hidden layer with tanh and an output layer with softmax
        unnormalized = []
        for segmentation in segmentation_outputs:
            unnormalized.append(v * dy.tanh(w*segmentation))
        att_weights = dy.softmax(dy.concatenate(unnormalized))
        return att_weights

    for ITER in range(50):
        for word, segmentations in word2sgmt.items():
            word_index = word_indices[word]
            print(word)

            # generate the representations for each segmentation
            segmentation_outputs = build_segmentation_graph(word_index, fwdLSTM, bwdLSTM)

            H = dy.parameter(hidden_layer)
            # reduce each segmentation representation to its half.
            for seg in range(len(segmentation_outputs)):
                segmentation_outputs[seg] = H * segmentation_outputs[seg] # H: hidden layer parameters

            # apply attention to learn the weights for each each segmentation
            att_weights = attention(segmentation_outputs)

            # compute weighted sum
            weighted_sum=dy.cmult(att_weights[0], segmentation_outputs[0])
            for i in range(1, number_of_segmentation):
                weighted_sum += dy.cmult(att_weights[i], segmentation_outputs[i])

            # compute cosine proximity loss and backpropagate
            y_word_vec = y_train[vocab[word]]
            y = dy.vecInput(300)
            y.set(y_word_vec)
            loss = dy.cdiv(dy.dot_product(weighted_sum, y), (dy.squared_norm(weighted_sum)*dy.squared_norm(y)))
            loss.backward()
            trainer.update()
            dy.renew_cg()  # build a new LSTM for each word

if __name__ == '__main__': main()


### To do:
# 1. Add l2 regularization
