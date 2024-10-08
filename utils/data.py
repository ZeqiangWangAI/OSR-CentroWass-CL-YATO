# @Author: Jie
# @Date:   2017-06-14 17:34:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-03-25 15:39:06
from __future__ import print_function
from __future__ import absolute_import
import sys
from .alphabet import Alphabet
from .functions import *

try:
    import cPickle as pickle
except ImportError:
    import pickle as pickle

START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"


class Data:

    def __init__(self):
        self.sentence_classification = False
        self.words2sent_representation = "Attention"
        self.MAX_SENTENCE_LENGTH = 512
        self.number_normalized = False
        self.norm_word_emb = False
        self.norm_char_emb = False
        self.word_alphabet = Alphabet('word')
        self.char_alphabet = Alphabet('character')
        self.word_count_dict = {}
        self.word_cutoff = 0

        self.feature_name = []
        self.feature_alphabets = []
        self.feature_num = len(self.feature_alphabets)
        self.feat_config = None

        self.label_alphabet = Alphabet(name='label', label=True, keep_growing=True)
        self.tagScheme = "Unknown"  ## BIOES/BIO
        self.split_token = ' ||| '
        self.seg = True

        self.silence = False

        ### I/O
        self.train_dir = None
        self.dev_dir = None
        self.test_dir = None
        self.raw_dir = None

        self.decode_dir = None
        self.dset_dir = None  ## data vocabulary related file
        self.model_dir = None  ## model save  file
        self.load_model_dir = None  ## model load file

        self.word_emb_dir = None
        self.char_emb_dir = None
        self.feature_emb_dirs = []

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.raw_texts = []
        self.predict_texts = []

        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []
        self.raw_Ids = []
        self.predict_Ids = []

        self.pretrain_word_embedding = None
        self.pretrain_char_embedding = None
        self.pretrain_feature_embeddings = []

        self.label_size = 0
        self.word_alphabet_size = 0
        self.char_alphabet_size = 0
        self.label_alphabet_size = 0
        self.feature_alphabet_sizes = []
        self.feature_emb_dims = []
        self.norm_feature_embs = []
        self.word_emb_dim = 50
        self.char_emb_dim = 30
        self.sentence_tags = []

        ###Networks
        self.word_feature_extractor = "LSTM"  ## "LSTM"/"CNN"/"GRU"/
        self.use_char = True
        self.use_word_seq = True
        self.use_word_emb = True
        self.char_feature_extractor = "CNN"  ## "LSTM"/"CNN"/"GRU"/None
        self.low_level_transformer = None  ## different bert model, if enate bert output in word_input level  (bert + word embedding + char)
        self.low_level_transformer_finetune = True
        self.high_level_transformer = None  #### different bert model, if concatenate bert output in word sequence level  (bert + word LSTM/CNN)
        self.high_level_transformer_finetune = True
        self.use_crf = True
        self.nbest = 0

        ## Training
        self.average_batch_loss = False
        self.optimizer = "SGD"  ## "SGD"/"AdaGrad"/"AdaDelta"/"RMSProp"/"Adam"
        self.status = "train"
        ### Hyperparameters
        self.HP_cnn_layer = 4
        self.HP_iteration = 100
        self.HP_batch_size = 10
        self.HP_char_hidden_dim = 50
        self.HP_hidden_dim = 768
        self.HP_dropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = True

        self.HP_gpu = False
        self.device = 'GPU'
        self.HP_lr = 0.015
        self.HP_clip = None
        self.HP_lr_decay = 0
        self.HP_momentum = 0
        self.HP_l2 = 1e-8
        self.scheduler = 'liner'
        self.warmup_step_rate = 0.0
        self.customTokenizer = 'None'
        self.customModel = 'None'
        self.customCofig = 'None'
        self.classifier= False
        self.openmax = False
        self.wcl = False
        self.openmax_yaml = None
        self.classification_activation = 'tanh'
        self.classifier_dropout = 0.1

    def initial_alphabets(self, input_list=None):
        if input_list == None:
            self.initial_feature_alphabets()
            self.build_alphabet(self.train_dir)
            self.build_alphabet(self.dev_dir)
            self.build_alphabet(self.test_dir)
        else:
            '''
            input_list: [train_list, dev_list, test_list]
                  train_list/dev_list/test_list: [sent_list, label_list, feature_list]
                          sent_list: list of list [[word1, word2,...],...,[wordx, wordy]...]
                          label_list:     if sentence_classification: 
                                               list of labels [label1, label2,...labelx, labely,...]
                                          else: 
                                               list of list [[label1, label2,...],...,[labelx, labely,...]]
                          feature_list:   if sentence_classification: 
                                               list of labels [[feat1, feat2,..],...,[feat1, feat2,..]], len(feature_list)= sentence_num
                                          else: 
                                               list of list [[[feat1, feat2,..],...,[feat1, feat2,..]],...,[[feat1, feat2,..],...,[feat1, feat2,..]]], , len(feature_list)= sentence_num
            '''
            self.initial_feature_alphabets_from_list(input_list[0][2][0])
            for each_list in input_list:
                self.build_alphabet_from_list(each_list[0], each_list[1], each_list[2])
        self.fix_alphabet()

    def summary(self):
        print("++" * 50)
        if self.sentence_classification:
            print("Start Sentence Classification task...")
        else:
            print("Start   Sequence   Laebling   task...")
        print("++" * 50)
        print("DATA SUMMARY START:")
        print(" I/O:")

        print("     Train  file directory: %s" % (self.train_dir))
        print("     Dev    file directory: %s" % (self.dev_dir))
        print("     Test   file directory: %s" % (self.test_dir))
        print("     Raw    file directory: %s" % (self.raw_dir))
        print("     Dset   file directory: %s" % (self.dset_dir))
        print("     Model  file directory: %s" % (self.model_dir))
        print("     Loadmodel   directory: %s" % (self.load_model_dir))
        print("     Decode file directory: %s" % (self.decode_dir))
        print("++" * 20)
        print("Data and Settings:")
        print("     Tag          scheme: %s" % (self.tagScheme))
        print("     Split         token: %s" % (self.split_token))
        print("     MAX SENTENCE LENGTH: %s" % (self.MAX_SENTENCE_LENGTH))
        print("     Number   normalized: %s" % (self.number_normalized))
        print("     Word         cutoff: %s" % (self.word_cutoff))
        print("     Train instance number: %s" % (len(self.train_texts)))
        print("     Dev   instance number: %s" % (len(self.dev_texts)))
        print("     Test  instance number: %s" % (len(self.test_texts)))
        print("     Raw   instance number: %s" % (len(self.raw_texts)))
        print("     Word  alphabet size: %s" % (self.word_alphabet_size))
        print("     Char  alphabet size: %s" % (self.char_alphabet_size))
        print("     Label alphabet size: %s" % (self.label_alphabet_size))
        print("     Label alphabet content: %s" % (self.label_alphabet.get_content()))

        print("     FEATURE num: %s" % (self.feature_num))
        for idx in range(self.feature_num):
            print("         Fe: %s  alphabet  size: %s" % (
                self.feature_alphabets[idx].name, self.feature_alphabet_sizes[idx]))
            print(
                "         Fe: %s  embedding  dir: %s" % (self.feature_alphabets[idx].name, self.feature_emb_dirs[idx]))
            print(
                "         Fe: %s  embedding size: %s" % (self.feature_alphabets[idx].name, self.feature_emb_dims[idx]))
            print(
                "         Fe: %s  norm       emb: %s" % (self.feature_alphabets[idx].name, self.norm_feature_embs[idx]))
        print(" " + "++" * 20)
        print(" Model Network:")
        if not self.sentence_classification:
            print("     Model        use_crf: %s" % (self.use_crf))
        if self.use_word_seq:
            print("     Model   use_word_seq: %s" % (self.use_word_seq))
            print("     Model   use_word_emb: %s" % (self.use_word_emb))

            if self.use_word_emb:
                print("     Word embedding  dir: %s" % (self.word_emb_dir))
                print("     Word embedding size: %s" % (self.word_emb_dim))
                print("     Norm   word     emb: %s" % (self.norm_word_emb))
            print("     Model       use_char: %s" % (self.use_char))
            if self.use_char:
                print("     Model char extractor: %s" % (self.char_feature_extractor))
                print("     Char embedding  dir: %s" % (self.char_emb_dir))
                print("     Char embedding size: %s" % (self.char_emb_dim))
                print("     Norm   char     emb: %s" % (self.norm_char_emb))
                print("     Model char_hidden_dim: %s" % (self.HP_char_hidden_dim))

            print("     Model word extractor: %s" % (self.word_feature_extractor))

        print("     Model low level transformer: %s; Finetune: %s" % (
            self.low_level_transformer, self.low_level_transformer_finetune))
        print("     Model high level transformer: %s; Finetune: %s" % (
            self.high_level_transformer, self.high_level_transformer_finetune))
        if self.sentence_classification:
            print("     Words hidden 2 sent: %s" % (self.words2sent_representation))
        print("     Classifier: %s; Classoifier Activation: %s; OpenMax: %s; OpenMax YAML: %s; WCL: %s" % (
            self.classifier, self.classification_activation, self.openmax, self.openmax_yaml, self.wcl))
        print("     Classifier         dropout: %s" % (self.classifier_dropout))

        print(" " + "++" * 20)
        print(" Training:")
        print("     Optimizer: %s" % (self.optimizer))
        print("     Scheduler: %s" % (self.scheduler))
        print("     warmup_step_rate: %f" % (self.warmup_step_rate))
        print("     Iteration: %s" % (self.HP_iteration))
        print("     BatchSize: %s" % (self.HP_batch_size))
        print("     Average  batch   loss: %s" % (self.average_batch_loss))

        print(" " + "++" * 20)
        print(" Hyperparameters:")
        print("     Hyper              lr: %s" % (self.HP_lr))
        print("     Hyper        lr_decay: %s" % (self.HP_lr_decay))
        print("     Hyper         HP_clip: %s" % (self.HP_clip))
        print("     Hyper        momentum: %s" % (self.HP_momentum))
        print("     Hyper              l2: %s" % (self.HP_l2))
        if self.word_feature_extractor != None and self.word_feature_extractor.lower() != "none":
            print("     Hyper      hidden_dim: %s" % (self.HP_hidden_dim))
        print("     Hyper         dropout: %s" % (self.HP_dropout))
        if self.use_word_seq and (self.word_feature_extractor == "GRU" or self.word_feature_extractor == "LSTM"):
            print("     Hyper      lstm_layer: %s" % (self.HP_lstm_layer))
            print("     Hyper          bilstm: %s" % (self.HP_bilstm))
        print("     Hyper             GPU: %s" % (self.HP_gpu))
        print("     Hyper          device: %s" % (self.device))
        print("DATA SUMMARY END.")
        print("++" * 50)
        sys.stdout.flush()

    def initial_feature_alphabets(self):
        first_line = open(self.train_dir, 'r').readline().strip('\n')
        if self.sentence_classification:
            ## if sentence classification data format, splited by '\t'
            items = first_line.split(self.split_token)
        else:
            ## if sequence labeling data format i.e. CoNLL 2003, split by ' '
            items = first_line.split()
        total_column = len(items)
        if total_column > 2:
            for idx in range(1, total_column - 1):
                feature_prefix = items[idx].split(']', 1)[0] + "]"
                self.feature_alphabets.append(Alphabet(feature_prefix))
                self.feature_name.append(feature_prefix)
                print("Find feature: ", feature_prefix)
        self.feature_num = len(self.feature_alphabets)
        self.pretrain_feature_embeddings = [None] * self.feature_num
        self.feature_emb_dims = [20] * self.feature_num
        self.feature_emb_dirs = [None] * self.feature_num
        self.norm_feature_embs = [False] * self.feature_num
        self.feature_alphabet_sizes = [0] * self.feature_num
        if self.feat_config:
            for idx in range(self.feature_num):
                if self.feature_name[idx] in self.feat_config:
                    self.feature_emb_dims[idx] = self.feat_config[self.feature_name[idx]]['emb_size']
                    self.feature_emb_dirs[idx] = self.feat_config[self.feature_name[idx]]['emb_dir']
                    self.norm_feature_embs[idx] = self.feat_config[self.feature_name[idx]]['emb_norm']

    def initial_feature_alphabets_from_list(self, one_feature_list):
        ## one_feature_list: features of one instance.
        # if sentence_classification:
        #   one_feature_list = [f1, f2, f3] for one sentence
        # else;
        #   one_feature_list = [[f1, f2, f3], [f1,f2,f3],...] for words within one sentence
        #
        if self.sentence_classification:
            items = one_feature_list
        else:
            items = one_feature_list[0]
        total_column = len(items)
        if total_column > 2:
            for idx in range(1, total_column - 1):
                feature_prefix = items[idx].split(']', 1)[0] + "]"
                self.feature_alphabets.append(Alphabet(feature_prefix))
                self.feature_name.append(feature_prefix)
                print("Find feature: ", feature_prefix)
        self.feature_num = len(self.feature_alphabets)
        self.pretrain_feature_embeddings = [None] * self.feature_num
        self.feature_emb_dims = [20] * self.feature_num
        self.feature_emb_dirs = [None] * self.feature_num
        self.norm_feature_embs = [False] * self.feature_num
        self.feature_alphabet_sizes = [0] * self.feature_num
        if self.feat_config:
            for idx in range(self.feature_num):
                if self.feature_name[idx] in self.feat_config:
                    self.feature_emb_dims[idx] = self.feat_config[self.feature_name[idx]]['emb_size']
                    self.feature_emb_dirs[idx] = self.feat_config[self.feature_name[idx]]['emb_dir']
                    self.norm_feature_embs[idx] = self.feat_config[self.feature_name[idx]]['emb_norm']

    def build_alphabet(self, input_file):
        in_lines = open(input_file, 'r').readlines()
        for line in in_lines:
            if len(line) > 2:
                ## if sentence classification data format, splited by \t
                if self.sentence_classification:
                    pairs = line.strip().split(self.split_token)
                    sent = pairs[0]
                    sent = sentence_preprocessing(sent)
                    words = sent.split()
                    for word in words:
                        if self.number_normalized:
                            word = normalize_word(word)
                        self.word_alphabet.add(word)
                        for char in word:
                            self.char_alphabet.add(char)
                    label = pairs[-1]
                    self.label_alphabet.add(label)
                    self.sentence_tags.append(label)
                    ## build feature alphabet
                    for idx in range(self.feature_num):
                        feat_idx = pairs[idx + 1].split(']', 1)[-1]
                        self.feature_alphabets[idx].add(feat_idx)

                ## if sequence labeling data format i.e. CoNLL 2003
                else:
                    pairs = line.strip().split()
                    word = pairs[0]
                    word = word_preprocessing(word)
                    if self.number_normalized:
                        word = normalize_word(word)
                    label = pairs[-1]
                    self.label_alphabet.add(label)
                    self.word_alphabet.add(word)
                    ## build feature alphabet
                    for idx in range(self.feature_num):
                        feat_idx = pairs[idx + 1].split(']', 1)[-1]
                        self.feature_alphabets[idx].add(feat_idx)
                    for char in word:
                        self.char_alphabet.add(char)
        self.word_alphabet_size = self.word_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()
        for idx in range(self.feature_num):
            self.feature_alphabet_sizes[idx] = self.feature_alphabets[idx].size()
        startS = False
        startB = False
        for label, _ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BIOES"
            else:
                self.tagScheme = "BIO"
        if self.sentence_classification:
            self.tagScheme = "Not sequence labeling task"

    def build_alphabet_from_list(self, sent_list, label_list=None, feature_list=None):

        '''
        sent_list: list of list [[word1, word2,...],...,[wordx, wordy]...]
        label_list: if sentence_classification: list of labels [label1, label2,...labelx, labely,...]
                      else: list of list [[label1, label2,...],...,[labelx, labely,...]]
        feature_list: if sentence_classification: list of labels [[feat1, feat2,..],...,[feat1, feat2,..]], len(feature_list)= sentence_num
                      else: list of list [[[feat1, feat2,..],...,[feat1, feat2,..]],...,[[feat1, feat2,..],...,[feat1, feat2,..]]], , len(feature_list)= sentence_num
        '''
        ## count word
        for sent in sent_list:
            for word in sent:
                if self.number_normalized:
                    word = normalize_word(word)
                if word in self.word_count_dict:
                    self.word_count_dict[word] += 1
                else:
                    self.word_count_dict[word] = 1

        for sent in sent_list:
            for word in sent:
                if self.number_normalized:
                    word = normalize_word(word)
                if self.word_count_dict[word] > self.word_cutoff:
                    self.word_alphabet.add(word)
                    for char in word:
                        self.char_alphabet.add(char)
                ## TODO: add feature list
        if label_list != None:
            for label in label_list:
                if type(label) is list:
                    for each_label in label:
                        self.label_alphabet.add(each_label)
                else:
                    self.label_alphabet.add(label)
        self.word_alphabet_size = self.word_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()
        ## TODO: feature alphabet
        if label_list != None:
            startS = False
            startB = False
            for label, _ in self.label_alphabet.iteritems():
                if "S-" in label.upper():
                    startS = True
                elif "B-" in label.upper():
                    startB = True
            if startB:
                if startS:
                    self.tagScheme = "BIOES"
                else:
                    self.tagScheme = "BIO"
        if self.sentence_classification:
            self.tagScheme = "Not sequence labeling task"

    def fix_alphabet(self):
        self.word_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close()
        for idx in range(self.feature_num):
            self.feature_alphabets[idx].close()

    def build_pretrain_emb(self):
        if self.word_emb_dir:
            print("Load pretrained word embedding, norm: %s, dir: %s" % (self.norm_word_emb, self.word_emb_dir))
            self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(self.word_emb_dir,
                                                                                       self.word_alphabet,
                                                                                       self.word_emb_dim,
                                                                                       self.norm_word_emb)
        if self.char_emb_dir:
            print("Load pretrained char embedding, norm: %s, dir: %s" % (self.norm_char_emb, self.char_emb_dir))
            self.pretrain_char_embedding, self.char_emb_dim = build_pretrain_embedding(self.char_emb_dir,
                                                                                       self.char_alphabet,
                                                                                       self.char_emb_dim,
                                                                                       self.norm_char_emb)
        for idx in range(self.feature_num):
            if self.feature_emb_dirs[idx]:
                print("Load pretrained feature %s embedding:, norm: %s, dir: %s" % (
                    self.feature_name[idx], self.norm_feature_embs[idx], self.feature_emb_dirs[idx]))
                self.pretrain_feature_embeddings[idx], self.feature_emb_dims[idx] = build_pretrain_embedding(
                    self.feature_emb_dirs[idx], self.feature_alphabets[idx], self.feature_emb_dims[idx],
                    self.norm_feature_embs[idx])

    def generate_instance(self, name, predict_list=None):

        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_instance(self.train_dir, self.word_alphabet, self.char_alphabet,
                                                             self.feature_alphabets, self.label_alphabet,
                                                             self.number_normalized, self.MAX_SENTENCE_LENGTH,
                                                             self.sentence_classification, self.split_token,
                                                             predict_line=None)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_instance(self.dev_dir, self.word_alphabet, self.char_alphabet,
                                                         self.feature_alphabets, self.label_alphabet,
                                                         self.number_normalized, self.MAX_SENTENCE_LENGTH,
                                                         self.sentence_classification, self.split_token,
                                                         predict_line=None)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance(self.test_dir, self.word_alphabet, self.char_alphabet,
                                                           self.feature_alphabets, self.label_alphabet,
                                                           self.number_normalized, self.MAX_SENTENCE_LENGTH,
                                                           self.sentence_classification, self.split_token,
                                                           predict_line=None)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_instance(self.raw_dir, self.word_alphabet, self.char_alphabet,
                                                         self.feature_alphabets, self.label_alphabet,
                                                         self.number_normalized, self.MAX_SENTENCE_LENGTH,
                                                         self.sentence_classification, self.split_token,
                                                         predict_line=None)

        elif name == 'predict':
            self.predict_texts, self.predict_Ids = read_instance(predict_list, self.word_alphabet,
                                                                 self.char_alphabet,
                                                                 self.feature_alphabets, self.label_alphabet,
                                                                 self.number_normalized,
                                                                 self.MAX_SENTENCE_LENGTH,
                                                                 self.sentence_classification, self.split_token,
                                                                 predict_line=predict_list)
        else:
            print("Error: you can only generate train/dev/test/raw/predict instance! Illegal input:%s" % (name))

    def generate_instance_from_list(self, input_list, name):
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
        '''

        instance_texts, instance_Ids = read_instance_from_list(input_list, self.word_count_dict, self.word_cutoff,
                                                               self.word_alphabet, self.char_alphabet,
                                                               self.feature_alphabets, self.label_alphabet,
                                                               self.number_normalized, self.MAX_SENTENCE_LENGTH,
                                                               self.sentence_classification, self.split_token)
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = instance_texts, instance_Ids
        elif name == "dev":
            self.dev_texts, self.dev_Ids = instance_texts, instance_Ids
        elif name == "test":
            self.test_texts, self.test_Ids = instance_texts, instance_Ids
        elif name == "raw":
            self.raw_texts, self.raw_Ids = instance_texts, instance_Ids
        elif name == 'predict':
            self.predict_texts, self.predict_Ids = instance_texts, instance_Ids
        else:
            print("Error: you can only generate train/dev/test/raw instance! Illegal input:%s" % (name))
        return instance_Ids

    def write_decoded_results(self, predict_results, name):
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
            content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        elif name == 'predict':
            content_list = self.predict_texts
        else:
            print("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        assert (sent_num == len(content_list))
        fout = open(self.decode_dir, 'w')
        for idx in range(sent_num):
            if self.sentence_classification:
                fout.write(" ".join(content_list[idx][0]) + self.split_token + predict_results[idx] + '\n')
            else:
                sent_length = len(predict_results[idx])
                for idy in range(sent_length):
                    ## content_list[idx] is a list with [word, char, label]
                    try:
                        fout.write(content_list[idx][0][idy].encode('utf-8') + " " + predict_results[idx][idy] + '\n')
                    except:
                        fout.write(content_list[idx][0][idy] + " " + predict_results[idx][idy] + '\n')
                fout.write('\n')
        fout.close()
        print("Predict %s result has been written into file. %s" % (name, self.decode_dir))

    def return_predict_results(self, predict_results):
        return_result = []
        sent_num = len(predict_results)
        content_list = self.predict_texts
        assert (sent_num == len(content_list))
        for idx in range(sent_num):
            if self.sentence_classification:
                return_result.append(" ".join(content_list[idx][0]) + self.split_token + predict_results[idx])
            else:
                sent_length = len(predict_results[idx])
                for idy in range(sent_length):
                    ## content_list[idx] is a list with [word, char, label]
                    try:
                        return_result.append(
                            content_list[idx][0][idy].encode('utf-8') + " " + predict_results[idx][idy])
                    except:
                        return_result.append(content_list[idx][0][idy] + " " + predict_results[idx][idy])
        return return_result

    def load(self, data_file):
        f = open(data_file, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    def save(self, save_file):
        f = open(save_file, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def write_nbest_decoded_results(self, predict_results, pred_scores, name):
        ## predict_results : [whole_sent_num, nbest, each_sent_length]
        ## pred_scores: [whole_sent_num, nbest]
        fout = open(self.decode_dir, 'w')
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
            content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        elif name == 'predict':
            content_list = self.predict_texts
        else:
            print("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        print('sent_num pred_scores', sent_num, len(pred_scores))
        assert (sent_num == len(content_list))
        assert (sent_num == len(pred_scores))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx][0])
            nbest = len(predict_results[idx])
            score_string = "# "
            for idz in range(nbest):
                score_string += format(pred_scores[idx][idz], '.4f') + " "
            fout.write(score_string.strip() + "\n")

            for idy in range(sent_length):
                try:  # Will fail with python3
                    label_string = content_list[idx][0][idy].encode('utf-8') + " "
                except:
                    label_string = content_list[idx][0][idy] + " "
                for idz in range(nbest):
                    label_string += predict_results[idx][idz][idy] + " "
                label_string = label_string.strip() + "\n"
                fout.write(label_string)
            fout.write('\n')
        fout.close()
        print("Predict %s %s-best result has been written into file. %s" % (name, nbest, self.decode_dir))

    def manual_config(self, manual_dict):
        """

        :param manual_dict:
        :return:
        """
        config = manual_dict
        ## read data:
        the_item = 'train_dir'
        if the_item in config:
            self.train_dir = config[the_item]
        the_item = 'dev_dir'
        if the_item in config:
            self.dev_dir = config[the_item]
        the_item = 'test_dir'
        if the_item in config:
            self.test_dir = config[the_item]
        the_item = 'raw_dir'
        if the_item in config:
            self.raw_dir = config[the_item]
        the_item = 'decode_dir'
        if the_item in config:
            self.decode_dir = config[the_item]
        the_item = 'dset_dir'
        if the_item in config:
            self.dset_dir = config[the_item]
        the_item = 'model_dir'
        if the_item in config:
            self.model_dir = config[the_item]
        the_item = 'load_model_dir'
        if the_item in config:
            self.load_model_dir = config[the_item]

        the_item = 'word_emb_dir'
        if the_item in config:
            self.word_emb_dir = config[the_item]
        the_item = 'char_emb_dir'
        if the_item in config:
            self.char_emb_dir = config[the_item]
        the_item = 'MAX_SENTENCE_LENGTH'
        if the_item in config:
            self.MAX_SENTENCE_LENGTH = int(config[the_item])

        the_item = 'norm_word_emb'
        if the_item in config:
            self.norm_word_emb = str2bool(config[the_item])
        the_item = 'norm_char_emb'
        if the_item in config:
            self.norm_char_emb = str2bool(config[the_item])
        the_item = 'number_normalized'
        if the_item in config:
            self.number_normalized = str2bool(config[the_item])
        the_item = 'word_cutoff'
        if the_item in config:
            self.word_cutoff = int(config[the_item])

        the_item = 'sentence_classification'
        if the_item in config:
            self.sentence_classification = str2bool(config[the_item])
        the_item = 'seg'
        if the_item in config:
            self.seg = str2bool(config[the_item])
        the_item = 'word_emb_dim'
        if the_item in config:
            self.word_emb_dim = int(config[the_item])
        the_item = 'char_emb_dim'
        if the_item in config:
            self.char_emb_dim = int(config[the_item])

        ## read network:
        the_item = 'use_crf'
        if the_item in config:
            self.use_crf = str2bool(config[the_item])
        the_item = 'use_char'
        if the_item in config:
            self.use_char = str2bool(config[the_item])
        the_item = 'use_word_seq'
        if the_item in config:
            self.use_word_seq = str2bool(config[the_item])
        the_item = 'use_word_emb'
        if the_item in config:
            self.use_word_emb = str2bool(config[the_item])
        the_item = 'word_seq_feature'
        if the_item in config:
            self.word_feature_extractor = config[the_item]
        the_item = 'char_seq_feature'
        if the_item in config:
            self.char_feature_extractor = config[the_item]
        the_item = 'low_level_transformer'
        if the_item in config:
            self.low_level_transformer = config[the_item]
        the_item = 'low_level_transformer_finetune'
        if the_item in config:
            self.low_level_transformer_finetune = str2bool(config[the_item])
        the_item = 'high_level_transformer'
        if the_item in config:
            self.high_level_transformer = config[the_item]
        the_item = 'high_level_transformer_finetune'
        if the_item in config:
            self.high_level_transformer_finetune = str2bool(config[the_item])
        the_item = 'nbest'
        if the_item in config:
            self.nbest = int(config[the_item])
        the_item = 'customTokenizer'
        if the_item in config:
            self.customTokenizer = config[the_item]
        the_item = 'customModel'
        if the_item in config:
            self.customModel = config[the_item]
        the_item = 'customCofig'
        if the_item in config:
            self.customCofig = config[the_item]
        the_item = 'feature'
        if the_item in config:
            self.feat_config = config[the_item]  ## feat_config is a dict

        ## read training setting:
        the_item = 'optimizer'
        if the_item in config:
            self.optimizer = config[the_item]
        ## read training setting:
        the_item = 'scheduler'
        if the_item in config:
            self.scheduler = config[the_item]
        the_item = 'warmup_step_rate'
        if the_item in config:
            self.warmup_step_rate = float(config[the_item])
        the_item = 'ave_batch_loss'
        if the_item in config:
            self.average_batch_loss = str2bool(config[the_item])
        the_item = 'status'
        if the_item in config:
            self.status = config[the_item]

        ## read Hyperparameters:
        the_item = 'cnn_layer'
        if the_item in config:
            self.HP_cnn_layer = int(config[the_item])
        the_item = 'iteration'
        if the_item in config:
            self.HP_iteration = int(config[the_item])
        the_item = 'batch_size'
        if the_item in config:
            self.HP_batch_size = int(config[the_item])

        the_item = 'char_hidden_dim'
        if the_item in config:
            self.HP_char_hidden_dim = int(config[the_item])
        the_item = 'hidden_dim'
        if the_item in config:
            self.HP_hidden_dim = int(config[the_item])
        the_item = 'dropout'
        if the_item in config:
            self.HP_dropout = float(config[the_item])
        the_item = 'classifier_dropout'
        if the_item in config:
            self.classifier_dropout = float(config[the_item])
        the_item = 'lstm_layer'
        if the_item in config:
            self.HP_lstm_layer = int(config[the_item])
        the_item = 'bilstm'
        if the_item in config:
            self.HP_bilstm = str2bool(config[the_item])
        the_item = 'gpu'
        if the_item in config:
            self.HP_gpu = str2bool(config[the_item])
        the_item = 'device'
        if the_item in config:
            self.device = config[the_item]
        the_item = 'learning_rate'
        if the_item in config:
            self.HP_lr = float(config[the_item])
        the_item = 'lr_decay'
        if the_item in config:
            self.HP_lr_decay = float(config[the_item])
        the_item = 'clip'
        if the_item in config:
            self.HP_clip = float(config[the_item])
        the_item = 'momentum'
        if the_item in config:
            self.HP_momentum = float(config[the_item])
        the_item = 'l2'
        if the_item in config:
            self.HP_l2 = float(config[the_item])
        the_item = 'words2sent'
        if the_item in config:
            self.words2sent_representation = config[the_item]
        the_item = 'classifier'
        if the_item in config:
            self.classifier = str2bool(config[the_item])
        the_item = 'classifier_activation'
        if the_item in config:
            self.classification_activation = config[the_item]
        the_item = "openmax"
        if the_item in config:
            self.openmax = str2bool(config[the_item])
        the_item = "wcl"
        if the_item in config:
            self.wcl = str2bool(config[the_item])
        the_item = "openmax_yaml"
        if the_item in config:
            self.openmax_yaml = config[the_item]


        ## no seg for sentence classification
        if self.sentence_classification:
            self.seg = False
            self.use_crf = False

    def read_config(self, config_file, nni_dict=None):
        """

        :param config_file:
        :param nni_dict:
        :return:
        """
        config = config_file_to_dict(config_file)
        if nni_dict is not None:
            config = self.nni_change(nni_dict=nni_dict, config=config)
        self.manual_config(config)

    def nni_change(self, nni_dict, config):
        """

        :param nni_dict:
        :param config:
        :return:
        """
        search_space = list(nni_dict.keys())
        for searchKey in search_space:
            if searchKey in config:
                config[searchKey] = nni_dict.get(searchKey)
        return config


def config_file_to_dict(input_file):
    """

    :param input_file:
    :return:
    """
    config = {}
    fins = open(input_file, 'r').readlines()
    for line in fins:
        if len(line) > 0 and line[0] == "#":
            continue
        if "=" in line:
            pair = line.strip().split('#', 1)[0].split('=', 1)
            item = pair[0]
            if item == "feature":
                if item not in config:
                    feat_dict = {}
                    config[item] = feat_dict
                feat_dict = config[item]
                new_pair = pair[-1].split()
                feat_name = new_pair[0]
                one_dict = {}
                one_dict["emb_dir"] = None
                one_dict["emb_size"] = 10
                one_dict["emb_norm"] = False
                if len(new_pair) > 1:
                    for idx in range(1, len(new_pair)):
                        conf_pair = new_pair[idx].split('=')
                        if conf_pair[0] == "emb_dir":
                            one_dict["emb_dir"] = conf_pair[-1]
                        elif conf_pair[0] == "emb_size":
                            one_dict["emb_size"] = int(conf_pair[-1])
                        elif conf_pair[0] == "emb_norm":
                            one_dict["emb_norm"] = str2bool(conf_pair[-1])
                feat_dict[feat_name] = one_dict
                # print "feat",feat_dict
            else:
                if item in config:
                    print("Warning: duplicated config item found: %s, updated." % (pair[0]))
                config[item] = pair[-1]
    return config


def str2bool(string):
    """

    :param string:
    :return:
    """

    if string == "True" or string == "true" or string == "TRUE":
        return True
    else:
        return False
