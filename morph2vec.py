from __future__ import print_function
from __future__ import unicode_literals

import codecs
import sys
import dynet_config
import dynet as dy
import numpy as np
from numpy.linalg import norm
import argparse
import dynet_config
import random


def main():
    parser = argparse.ArgumentParser(
        description='Learn morph2vec: morpheme-based representation learning ')
    parser.add_argument('--input', type=str, required=True, help = "path to training file")
    parser.add_argument('--wordVector', type=str, required=True, help = "path to word2vec vector file")
    parser.add_argument('--output', '-o', type=str, required=True, help = "output directory where the weight file will be saved")
    parser.add_argument('--batch', type=int, default=32,help = "batch size")
    parser.add_argument('--epoch', type=int, default=5,help = "number of epochs")
    parser.add_argument('--dim', type=int, default=200, help="dimension")

    args = parser.parse_args()

    gensim_model = args.wordVector
    training_file = args.input
    output_file =args.output
    batch_size = args.batch
    number_of_epoch = args.epoch
    dim = args.dim
    print('===================================  Prepare data...  ==============================================')
    print('')

    word2sgmt = {}
    seq = []
    morphs = []
    word_indices = {}
    indices_word = {}

    f = codecs.open(training_file, encoding='utf-8')
    index = 0
    for line in f:
        line = line.rstrip('\n')
        word, sgmnts = line.split(':')
        sgmt = sgmnts.split('+')
        sgmt = list(s.split('-') for s in sgmt if len(s)>0 and s!="###")
        word_indices[word] = index
        indices_word[index] = word
        word2sgmt[word] = sgmt
        seq.extend(sgmt)
        index = index + 1

    for sgmt in seq:
        for morph in sgmt:
            morphs.append(morph)

    print('number of words: ', len(word2sgmt))

    morph_indices = dict((c, i) for i, c in enumerate(set(morphs)))
    indices_morph = dict((i, c) for i, c in enumerate(set(morphs)))

    print('number of morphemes: ', len(morphs))
    print('number of unique morphemes: ', len(set(morphs)))

    def save_morpheme_dictionary():
        fout = "morphem-indices.txt"
        fo = open(fout, "w")
        for k, v in morph_indices.items():
            fo.write(str(k) + ':' + str(v) + '\n')
        fo.close()

    save_morpheme_dictionary()

    for word in word2sgmt:
        for i in range(len(word2sgmt[word])):
            word2sgmt[word][i] = [morph_indices[c] for c in word2sgmt[word][i]]

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

    print('number of words found: ', len(y_train)) # [[f1,f2,....,f200],[f1,f2,....,f200],.....]
    y_train = np.array(y_train)
    print('shape of Y: ', y_train.shape)

    print('')
    print('===================================  Build model...  ===============================================')
    print('')

    model = dy.Model()
    parameters = dy.ParameterCollection()

    morph_embeddings = model.add_lookup_parameters((len(set(morphs))+1, int(dim/4))) # randomly initialized

    def morph_rep(m): # returns the embedding of morpheme 'm' (m: morpheme index)
        return morph_embeddings[m]

    # input dim (morpheme embedding dimension): 200/4, output dimension: 300
    fwdLSTM = dy.LSTMBuilder(1, int(dim/4), dim, model)
    bwdLSTM = dy.LSTMBuilder(1, int(dim/4), dim, model)

    # in order to reduce the segmentation dimension to half.
    hidden_layer = model.add_parameters((dim, 2*dim))

    # attention parameters: w-> for the hidden layer, v-> for the output layer
    attention_w = model.add_parameters((dim, dim))
    attention_v = model.add_parameters((1, dim))
    trainer = dy.AdamTrainer(model)
    trainer.set_sparse_updates(False)   # sparse updates off

    #build the LSTM for all segmentations of a single word
    def build_segmentation_graph(word, fwdLSTM, bwdLSTM):
        word_segs = word2sgmt[word]
        # iterate for each segmentation
        segmentation_outputs_f = []
        segmentation_outputs_b = []

        f_init = fwdLSTM.initial_state()
        b_init = bwdLSTM.initial_state()

        for i in range(len(word_segs)):

            seg = word_segs[i]
            # for each unit (i.e. segment) in the LSTM forward computation
            state_f = f_init
            state_b = b_init
            for s in range(len(seg)):
                state_f = state_f.add_input(morph_rep(seg[s]))
                state_b = state_b.add_input(morph_rep(seg[len(seg)-s-1]))

            segmentation_outputs_f.append(state_f.output())
            #print(state_f.output().vec_value())
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

    def pick_cos_prox(pred, gold):

        def l2_normalize(x):
            epsilon = np.finfo(float).eps * dy.ones(pred.dim()[0])
            norm = dy.sqrt(dy.sum_elems(dy.square(x)))
            sign = dy.cdiv(x, dy.bmax(dy.abs(x), epsilon))
            return dy.cdiv(dy.cmult(sign, dy.bmax(dy.abs(x), epsilon)), dy.bmax(norm, epsilon[0]))

        y_true = l2_normalize(pred)
        y_pred = l2_normalize(gold)

        return dy.mean_elems(dy.cmult(y_true, y_pred))

    def cosine_proximity(pred, gold):

        def l2_normalize(x):
            square_sum = dy.sqrt(dy.bmax(dy.sum_elems(dy.square(x)), np.finfo(float).eps * dy.ones((1))[0]))
            return dy.cdiv(x, square_sum)

        y_true = l2_normalize(pred)
        y_pred = l2_normalize(gold)
        return -dy.sum_elems(dy.cmult(y_true, y_pred)) 

    print("Start training...")
    for ITER in range(number_of_epoch):
        unique_words = list(word2sgmt.keys())
        random.shuffle(unique_words)
        print(ITER)
        for word in unique_words:
            # generate the representations for each segmentation
            segmentation_outputs = build_segmentation_graph(word, fwdLSTM, bwdLSTM)

            H = dy.parameter(hidden_layer)
            # reduce each segmentation representation to its half.
            for seg in range(len(segmentation_outputs)):
                segmentation_outputs[seg] = H * segmentation_outputs[seg] # H: hidden layer parameters

            # apply attention to learn the weights for each each segmentation
            att_weights = attention(segmentation_outputs)

            # compute weighted sum
            weighted_sum=dy.cmult(att_weights[0], segmentation_outputs[0])
            for i in range(1, len(segmentation_outputs)):
                weighted_sum += dy.cmult(att_weights[i], segmentation_outputs[i])

            # compute cosine proximity loss and backpropagate
            y_word_vec = y_train[vocab[word]]
            y = dy.vecInput(200)
            y.set(y_word_vec)
            loss = cosine_proximity(weighted_sum, y)
            loss.backward()
            trainer.update()
            dy.renew_cg()

    model.save("saved-model")
if __name__ == '__main__': main()


### To do:
# 1. Add l2 regularization (or add noise to the lookup parameters)
