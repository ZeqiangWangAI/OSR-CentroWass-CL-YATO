# -*- coding: utf-8 -*
from main import *
from utils import *
from model import *
import argparse
import re
import seqeval
import sklearn
import pandas as pd


class YATO:
    def __init__(self, config):
        self.set_seed()
        self.config = config
        self.data = Data()
        self.data.read_config(self.config)
        

    def set_config_from_dset(self, dset):
        self.data.load(dset)

    def set_config_from_data(self, custom_data):
        self.data = custom_data

    def set_config_from_custom_configuration(self, custom_configuration):
        self.data.read_config(self.config, custom_configuration)

    def get_config(self):
        return self.data

    def train(self, log='test.log', metric='F'):
        status = self.data.status.lower()
        if status == 'train':
            print("MODEL: train")
            data_initialization(self.data)
            self.data.generate_instance('train')
            self.data.generate_instance('dev')
            self.data.generate_instance('test')
            self.data.build_pretrain_emb()
            self.data.summary()
            train(self.data, log, metric)

    def decode(self, write_decode_file=True):
        print("MODEL: decode")
        predict_lines = self.convert_file_to_predict_style()
        speed, acc, p, r, f, pred_results, pred_scores = self.predict(input_text=predict_lines,
                                                                      write_decode_file=write_decode_file)

        return {"speed": speed, "accuracy": acc, "precision": p, "recall": r, "predict_result": pred_results,
                "nbest_predict_score": pred_scores, 'label': self.data.label_alphabet}

    def predict(self, input_text=None, predict_file=None, write_decode_file=True):
        self.data.read_config(self.config)
        dset = self.data.dset_dir
        self.set_config_from_dset(dset)
        self.data.read_config(self.config)
        if predict_file is not None and input_text is None:
            input_text = open(predict_file, 'r', encoding="utf8").readlines()
        elif predict_file is not None and input_text is not None:
            print("Choose Predict Source")
        self.data.generate_instance('predict', input_text)
        print("nbest: {}".format(self.data.nbest))
        speed, acc, p, r, f, pred_results, pred_scores = load_model_decode(self.data, 'predict')
        if write_decode_file and self.data.nbest > 0 and not self.data.sentence_classification:
            self.data.write_nbest_decoded_results(pred_results, pred_scores, 'predict')
        elif write_decode_file:
            self.data.write_decoded_results(pred_results, 'predict')
        return speed, acc, p, r, f, pred_results, pred_scores

    def attention(self, input_text=None):
        self.data.read_config(self.config)
        dset = self.data.dset_dir
        self.set_config_from_dset(dset)
        self.data.read_config(self.config)
        print("MODEL: Attention Weight")
        self.data.generate_instance('predict', input_text)
        probs_ls, weights_ls = extract_attention_weight(self.data)
        return probs_ls, weights_ls

    def set_seed(self, seed=42, hard=False):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        torch.backends.cudnn.deterministic = True
        if hard:
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            os.environ['PYTHONHASHSEED'] = str(seed)

    def convert_file_to_predict_style(self):
        predict_lines = open(self.data.raw_dir, 'r', encoding="utf8").readlines()
        return predict_lines

    def para2sent(self, para):
        """
        :param para:Dividing paragraphs into sentences
        """
        para = re.sub('([.。!！?？\?])([^”’])', r"\1\n\2", para)
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        para = para.rstrip()
        return para.split("\n")

    def sent2word(self, sentence):
        """
        :param sentence:Dividing sentences into words or chars
        """
        char_ls = list(sentence)
        word_ls = [char_ls[0]]
        for i in range(1, len(char_ls)):
            if 65 <= ord(char_ls[i]) <= 122 and 65 <= ord(char_ls[i - 1]) <= 122:
                word_ls[-1] = word_ls[-1] + char_ls[i]
            else:
                word_ls.append(char_ls[i])
        return word_ls

    def decode_raw(self, raw_text_path, task, out_text_path='raw.out'):
        """
        :param raw_text_path:The path of raw text file
        :param task:choose the task
        :param out_text_path:The path of decode result file
        """
        raw_text = open(raw_text_path, 'r', encoding='utf-8').read()
        out_text = open(out_text_path, 'w', encoding='utf-8')
        if task.lower() == 'ner':
            sentences = self.para2sent(raw_text)
            for sentence in sentences:
                words = self.sent2word(sentence)
                for word in words:
                    out_text.write(word + ' O\n')
                out_text.write('\n')
        elif task.lower() == 'classifier':
            sentences = self.para2sent(raw_text)
            for sentence in sentences:
                out_text.write(sentence + ' ||| 0\n')
        self.data.raw_dir = out_text_path
        self.decode()

    def get_gold_predict_seqTagging(self, golden_standard, predict_result, stoken):
        """
        :param golden_standard:golden standard file path
        :param predict_result:predict result file path
        :param stoken:split token
        """
        golden_data = open(golden_standard, 'r', encoding='utf-8').readlines()
        predict_data = open(predict_result, 'r', encoding='utf-8').readlines()
        golden_list = []
        predict_list = []
        tmp_gold = []
        tmp_predict = []
        for gold_idx, pre_idx in zip(golden_data, predict_data):
            if gold_idx != '\n':
                gentity_with_label = gold_idx.split(stoken)
                glabel = gentity_with_label[1].replace('\n', '')
                tmp_gold.append(glabel)
                pentity_with_label = pre_idx.split(stoken)
                plabel = pentity_with_label[1].replace('\n', '')
                tmp_predict.append(plabel)
            else:
                golden_list.append(tmp_gold)
                predict_list.append(tmp_predict)
                tmp_gold = []
                tmp_predict = []
        return golden_list, predict_list
        
    def get_gold_predict_seqLabeling(self, golden_standard, predict_result, stoken):
        """
        :param golden_standard:golden standard file path
        :param predict_result:predict result file path
        :param stoken:split token
        """
        golden_data = open(golden_standard, 'r', encoding='utf-8').readlines()
        predict_data = open(predict_result, 'r', encoding='utf-8').readlines()
        golden_list = []
        predict_list = []
        for gold_idx, pre_idx in zip(golden_data, predict_data):
            gold_idx = gold_idx.strip()
            gold_seq_label = gold_idx.split(stoken)
            glabel = gold_seq_label[1]
            golden_list.append(glabel)

            pre_idx = pre_idx.strip()
            pre_seq_label = pre_idx.split(stoken)
            plabel = pre_seq_label[1].replace('\n', '')
            predict_list.append(plabel)
        return golden_list, predict_list

    def report_f1(self, golden_standard, predict_result, split=" ", sentence_clf = False, output_csv='report.csv'):
        """
        :param golden_standard:golden standard file path
        :param predict_result:predict result file path
        :param split:split token
        """
        if not sentence_clf:
            golden_list, predict_list = self.get_gold_predict_seqTagging(golden_standard, predict_result, split)
            report = seqeval.metrics.classification_report(golden_list, predict_list)
        else:
            golden_list, predict_list = self.get_gold_predict_seqLabeling(golden_standard, predict_result, split)
            report = sklearn.metrics.classification_report(golden_list, predict_list, output_dict=True, zero_division=0)
        print(report)
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv(output_csv, index=True)


    def report_acc(self, golden_standard, predict_result, split=' ||| ', sentence_clf = False):
        """
        :param golden_standard:golden standard file path
        :param predict_result:predict result file path
        :param split:split token
        """
        if not sentence_clf:
            golden_list, predict_list = self.get_gold_predict_seqTagging(golden_standard, predict_result, split)
            print("Report accuracy: %0.2f" % seqeval.metrics.accuracy_score(golden_list, predict_list))
        else:
            golden_list, predict_list = self.get_gold_predict_seqLabeling(golden_standard, predict_result, split)
            print("Report accuracy: %0.2f" % sklearn.metrics.accuracy_score(golden_list, predict_list))
            



def parse_args():
    description = "YATO: Yet Another deep learning based Text analysis Open toolkit"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--status", type=str, default='train', help="model status")
    parser.add_argument("--config", type=str, help="config file path")
    parser.add_argument("--log", type=str, default="test.log", help="log name")
    parser.add_argument("--metric", type=str, default="F", help="metric")
    return parser.parse_args()

