# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:23:06
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-04-01 15:52:06
from __future__ import print_function
from __future__ import absolute_import
import sys
import numpy as np


def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def sentence_preprocessing(input_sentence_string):
    ## add your own sentence preprocessing script here
    if sys.version_info[0] < 3:
        input_sentence_string = input_sentence_string.decode('utf-8')
    return input_sentence_string


def word_preprocessing(input_word_string):
    ## add your own word preprocessing script here
    if sys.version_info[0] < 3:
        input_word_string = input_word_string.decode('utf-8')
    return input_word_string


def generate_words_chars(original_words, word_alphabet, char_padding_size, char_padding_symbol, \
                         char_alphabet, number_normalized):
    words = []
    word_Ids = []
    chars = []
    char_Ids = []
    for word in original_words:
        words.append(word)
        if number_normalized:
            word = normalize_word(word)
        word_Ids.append(word_alphabet.get_index(word))
        ## get char
        char_list = []
        char_Id = []
        for char in word:
            char_list.append(char)
        if char_padding_size > 0:
            char_number = len(char_list)
            if char_number < char_padding_size:
                char_list = char_list + [char_padding_symbol] * (char_padding_size - char_number)
            assert (len(char_list) == char_padding_size)
        for char in char_list:
            char_Id.append(char_alphabet.get_index(char))
        chars.append(char_list)
        char_Ids.append(char_Id)
    return words, word_Ids, chars, char_Ids


def read_instance(input_file, word_alphabet, char_alphabet, feature_alphabets, label_alphabet, number_normalized,
                  max_sent_length, sentence_classification=False, split_token='\t', predict_line=None,
                  char_padding_size=-1,
                  char_padding_symbol='</pad>'):
    feature_num = len(feature_alphabets)

    instence_texts = []
    instence_Ids = []
    words = []
    features = []
    chars = []
    labels = []
    word_Ids = []
    feature_Ids = []
    char_Ids = []
    label_Ids = []

    if predict_line is None:
        in_lines = open(input_file, 'r', encoding="utf8").readlines()
    else:
        in_lines = predict_line

    ## if sentence classification data format, splited by split token
    if sentence_classification:
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split(split_token)
                sent = pairs[0]
                sent = sentence_preprocessing(sent)
                original_words = sent.split()
                words, word_Ids, chars, char_Ids = generate_words_chars(original_words, word_alphabet,
                                                                        char_padding_size, char_padding_symbol, \
                                                                        char_alphabet, number_normalized)

                if len(words) > max_sent_length:
                    original_words = original_words[:max_sent_length]
                    words, word_Ids, chars, char_Ids = generate_words_chars(original_words, word_alphabet,
                                                                            char_padding_size, char_padding_symbol, \
                                                                            char_alphabet, number_normalized)

                label = pairs[-1]
                label_Id = label_alphabet.get_index(label)
                ## get features
                feat_list = []
                feat_Id = []
                for idx in range(feature_num):
                    feat_idx = pairs[idx + 1].split(']', 1)[-1]
                    feat_list.append(feat_idx)
                    feat_Id.append(feature_alphabets[idx].get_index(feat_idx))
                ## combine together and return, notice the feature/label as different format with sequence labeling task
                instence_texts.append([words, feat_list, chars, label])
                instence_Ids.append([word_Ids, feat_Id, char_Ids, label_Id, words, features, chars, labels])


                words = []
                features = []
                chars = []
                char_Ids = []
                word_Ids = []
                feature_Ids = []
                label_Ids = []

    else:
        ### for sequence labeling data format i.e. CoNLL 2003
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split()
                word = pairs[0]
                word = word_preprocessing(word)
                words.append(word)
                if number_normalized:
                    word = normalize_word(word)
                label = pairs[-1]
                labels.append(label)
                word_Ids.append(word_alphabet.get_index(word))
                label_Ids.append(label_alphabet.get_index(label))
                ## get features
                feat_list = []
                feat_Id = []
                for idx in range(feature_num):
                    feat_idx = pairs[idx + 1].split(']', 1)[-1]
                    feat_list.append(feat_idx)
                    feat_Id.append(feature_alphabets[idx].get_index(feat_idx))
                features.append(feat_list)
                feature_Ids.append(feat_Id)
                ## get char
                char_list = []
                char_Id = []
                for char in word:
                    char_list.append(char)
                if char_padding_size > 0:
                    char_number = len(char_list)
                    if char_number < char_padding_size:
                        char_list = char_list + [char_padding_symbol] * (char_padding_size - char_number)
                    assert (len(char_list) == char_padding_size)
                else:
                    ### not padding
                    pass
                for char in char_list:
                    char_Id.append(char_alphabet.get_index(char))
                chars.append(char_list)
                char_Ids.append(char_Id)
            else:
                if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)):
                    instence_texts.append([words, features, chars, labels])
                    instence_Ids.append([word_Ids, feature_Ids, char_Ids, label_Ids, words, features, chars, labels])
                else:
                    words = words[:max_sent_length]
                    word_Ids = word_Ids[:max_sent_length]

                    features = words[:max_sent_length]
                    feature_Ids = feature_Ids[:max_sent_length]
                    labels = labels[:max_sent_length]
                    label_Ids = label_Ids[:max_sent_length]

                    char_list = []
                    char_Id = []
                    for char in word:
                        char_list.append(char)
                    if char_padding_size > 0:
                        char_number = len(char_list)
                        if char_number < char_padding_size:
                            char_list = char_list + [char_padding_symbol] * (char_padding_size - char_number)
                        assert (len(char_list) == char_padding_size)
                    else:
                        ### not padding
                        pass
                    for char in char_list:
                        char_Id.append(char_alphabet.get_index(char))
                    chars.append(char_list)
                    char_Ids.append(char_Id)
                    instence_texts.append([words, features, chars, labels])
                    instence_Ids.append([word_Ids, feature_Ids, char_Ids, label_Ids, words, features, chars, labels])

                words = []
                features = []
                chars = []
                labels = []
                word_Ids = []
                feature_Ids = []
                char_Ids = []
                label_Ids = []
        #the last sample we have handle it
        if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)):
            instence_texts.append([words, features, chars, labels])
            instence_Ids.append([word_Ids, feature_Ids, char_Ids, label_Ids, words, features, chars, labels])
        else:
            words = words[:max_sent_length]
            word_Ids = word_Ids[:max_sent_length]

            features = words[:max_sent_length]
            feature_Ids = feature_Ids[:max_sent_length]
            labels = labels[:max_sent_length]
            label_Ids = label_Ids[:max_sent_length]

            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol] * (char_padding_size - char_number)
                assert (len(char_list) == char_padding_size)
            else:
                ### not padding
                pass
            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)
            instence_texts.append([words, features, chars, labels])
            instence_Ids.append([word_Ids, feature_Ids, char_Ids, label_Ids, words, features, chars, labels])

        words = []
        features = []
        chars = []
        labels = []
        word_Ids = []
        feature_Ids = []
        char_Ids = []
        label_Ids = []
    return instence_texts, instence_Ids


def read_instance_from_list(input_list, word_count_dict, word_cutoff, word_alphabet, char_alphabet, feature_alphabets,
                            label_alphabet, number_normalized, max_sent_length, sentence_classification=False,
                            split_token='\t', char_padding_size=-1, char_padding_symbol='</pad>'):
    '''
          
          input_list: [sent_list, label_list, feature_list]
              sent_list: list of list [[word1, word2,...],...,[wordx, wordy]...]
              label_list:     if sentence_classification: 
                                   list of labels [label1, label2,...labelx, labely,...]
                              else: 
                                   list of list [[label1, label2,...],...,[labelx, labely,...]]
              feature_list:   if sentence_classification: 
                                   list of labels [[feat1, feat2,..],...,[feat1, feat2,..]], len(feature_list)= sentence_num
                              else: 
                                   list of list [[[feat1, feat2,..],...,[feat1, feat2,..]],...,[[feat1, feat2,..],...,[feat1, feat2,..]]], , len(feature_list)= sentence_num
          word_count_dict: word occurence number dict
          word_cutoff: int, threshold of cutoff low frequence word
    '''
    sent_list, label_list, feature_list = input_list
    feature_num = len(feature_alphabets)
    instence_texts = []
    instence_Ids = []
    words = []
    features = []
    chars = []
    labels = []
    word_Ids = []
    feature_Ids = []
    char_Ids = []
    label_Ids = []

    ## if sentence classification data format, splited by split token
    if sentence_classification:
        for sent, label, feature in zip(sent_list, label_list, feature_list):
            if len(sent) > max_sent_length:
                sent = sent[:max_sent_length]
            for word in sent:
                if (word not in word_count_dict) or (word_count_dict[word] > word_cutoff):
                    words.append(word)
                    if number_normalized:
                        word = normalize_word(word)
                    word_Ids.append(word_alphabet.get_index(word))
                    char_list = []
                    char_Id = []
                    for char in word:
                        char_list.append(char)
                    if char_padding_size > 0:
                        char_number = len(char_list)
                        if char_number < char_padding_size:
                            char_list = char_list + [char_padding_symbol] * (char_padding_size - char_number)
                        assert (len(char_list) == char_padding_size)
                    for char in char_list:
                        char_Id.append(char_alphabet.get_index(char))
                    chars.append(char_list)
                    char_Ids.append(char_Id)

            if len(words) == 0:
                for word in sent:
                    words.append(word)
                    if (word not in word_count_dict) or (word_count_dict[word] > word_cutoff):
                        if number_normalized:
                            word = normalize_word(word)
                        word_Ids.append(word_alphabet.get_index(word))
                    else:
                        word_Ids.append(word_alphabet.get_index(word_alphabet.UNKNOWN))
                    char_list = []
                    char_Id = []
                    for char in word:
                        char_list.append(char)
                    if char_padding_size > 0:
                        char_number = len(char_list)
                        if char_number < char_padding_size:
                            char_list = char_list + [char_padding_symbol] * (char_padding_size - char_number)
                        assert (len(char_list) == char_padding_size)
                    for char in char_list:
                        char_Id.append(char_alphabet.get_index(char))
                    chars.append(char_list)
                    char_Ids.append(char_Id)

            label_Id = label_alphabet.get_index(label)
            ## get features
            feat_list = []
            feat_Id = []
            for idx in range(feature_num):
                feat_idx = feature[idx].split(']', 1)[-1]
                feat_list.append(feat_idx)
                feat_Id.append(feature_alphabets[idx].get_index(feat_idx))
            instence_texts.append([words, feat_list, chars, label])
            instence_Ids.append([word_Ids, feat_Id, char_Ids, label_Id, words, features, chars, labels])
            words = []
            features = []
            chars = []
            char_Ids = []
            word_Ids = []
            feature_Ids = []
            label_Ids = []
    else:
        ### for sequence labeling data format i.e. CoNLL 2003
        for sent, labels, features in zip(sent_list, label_list, feature_list):
            if len(sent) > max_sent_length:
                sent = sent[:max_sent_length]
                labels = labels[:max_sent_length]
                features = features[:max_sent_length]
            for word, label, feature in zip(sent, labels, features):
                if (word not in word_count_dict) or (word_count_dict[word] > word_cutoff):
                    words.append(word)
                    if number_normalized:
                        word = normalize_word(word)
                    word_Ids.append(word_alphabet.get_index(word))
                    char_list = []
                    char_Id = []
                    for char in word:
                        char_list.append(char)
                    if char_padding_size > 0:
                        char_number = len(char_list)
                        if char_number < char_padding_size:
                            char_list = char_list + [char_padding_symbol] * (char_padding_size - char_number)
                        assert (len(char_list) == char_padding_size)
                    for char in char_list:
                        char_Id.append(char_alphabet.get_index(char))
                    chars.append(char_list)
                    char_Ids.append(char_Id)

                    labels.append(label)
                    label_Ids.append(label_alphabet.get_index(label))
                    ## get features
                    feat_list = []
                    feat_Id = []
                    for idx in range(feature_num):
                        feat_idx = feature[idx].split(']', 1)[-1]
                        feat_list.append(feat_idx)
                        feat_Id.append(feature_alphabets[idx].get_index(feat_idx))
                    features.append(feat_list)
                    feature_Ids.append(feat_Id)

            if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)):
                instence_texts.append([words, features, chars, labels])
                instence_Ids.append([word_Ids, feature_Ids, char_Ids, label_Ids, words, features, chars, labels])
            words = []
            features = []
            chars = []
            labels = []
            word_Ids = []
            feature_Ids = []
            char_Ids = []
            label_Ids = []

    return instence_texts, instence_Ids


def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    alphabet_size = word_alphabet.size()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index, :] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index, :] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
        pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / alphabet_size))
    return pretrain_emb, embedd_dim


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec / root_sum_square


def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding="utf8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            elif embedd_dim + 1 != len(tokens):
                ## ignore illegal embedding line
                continue
                # assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            first_col = tokens[0]
            first_col = word_preprocessing(first_col)
            embedd_dict[first_col] = embedd
    return embedd_dict, embedd_dim


if __name__ == '__main__':
    a = np.arange(9.0)
    print(a)
    print(norm2one(a))
