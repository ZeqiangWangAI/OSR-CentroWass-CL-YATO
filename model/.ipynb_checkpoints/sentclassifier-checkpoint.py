# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2019-01-01 21:11:50
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-03-29 14:27:55

from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import wasserstein_distance

from .wordsequence import WordSequence
from .feature.openmax_processing import OpenMaxProcessing
from .feature.contrastive_memory_bank import MemoryBankManager
from .classifier_head import SequenceClassifier

import math
import torch
import torch.linalg as linalg


class SentClassifier(nn.Module):
    def __init__(self, data):
        super(SentClassifier, self).__init__()

        if not data.silence:
            print("build sentence classification network...")
            print("use_char: ", data.use_char)
            if data.use_char:
                print("char feature extractor: ", data.char_feature_extractor)
            print("word feature extractor: ", data.word_feature_extractor)
        self.data = data
        self.status = data.status
        self.openmax = data.openmax
        self.wcl = data.wcl
        self.gpu = data.HP_gpu
        self.average_batch = data.average_batch_loss
        self.label_size = data.label_alphabet_size
        self.classifier = data.classifier
        self.word_hidden = WordSequence(data).to(data.device)
        self.openmax_yaml = data.openmax_yaml
        self.alpha = None  # OpenMax the number of top categories
        self.epsilon = None
        self.labels_data = None

        if self.classifier:
            self.classifier = SequenceClassifier(hidden_size=self.word_hidden.output_hidden_dim,
                                                 activation_function=data.classification_activation,
                                                 num_labels=self.label_size,
                                                 openmax=self.openmax,
                                                 classifier_dropout=data.classifier_dropout,
                                                 dropout_prob=data.HP_dropout).to(data.device)
            if self.openmax and self.wcl:

                # initialized MemoryBank
                self.memory_bank = MemoryBankManager(num_classes=self.label_size, bank_size=512,
                                                     embedding_dim=self.label_size,
                                                     device=self.data.device, min_samples_required=64)
        if self.status == 'openmax_decode':
            self.weibull_models, self.mean_vectors, self.top_weibull_model, self.top_mean_vectors = self.initialize_openmax()

    def calculate_loss(self, *input):
        device = self.data.device

        word_inputs = input[0].to(device)
        batch_label = input[7].to(device)

        batch_size = word_inputs.size(0)
        outs, _ = self.word_hidden.sentence_representation(*input)

        if self.classifier:
            if self.openmax and self.status == 'train':
                outs, dense_outputs = self.classifier(outs.to(device))
            else:
                outs = self.classifier(outs.to(device))

        outs = outs.view(batch_size, -1)

        # Compute cross-entropy loss
        total_loss = F.cross_entropy(outs, batch_label.view(batch_size))

        # Optional averaging of the loss
        if self.average_batch:
            total_loss = total_loss / batch_size

        # Get predicted labels
        _, predicted_labels = torch.max(outs, 1)

        if self.openmax and self.status == 'train' :

            true_label_list = batch_label.view(-1)

            if self.wcl:

                info_nce_loss = self.compute_info_nce_loss(outs , true_label_list)

                if info_nce_loss is not None:
                    total_loss = total_loss + 0.1 * info_nce_loss

                # update memory bank
                self.memory_bank.update(dense_outputs.detach(), batch_label.view(batch_size), predicted_labels.detach())


            correct_dense_outputs = {i: [] for i in range(self.label_size)}

            correct_mask = (predicted_labels == batch_label.view(batch_size))

            correct_indices = torch.nonzero(correct_mask).squeeze()

            if correct_indices.dim() == 0:
                correct_indices = correct_indices.unsqueeze(0)

            for idx in correct_indices:
                idx = idx.item()

                label = batch_label[idx].item()
                vector = dense_outputs[idx].detach().cpu()

                correct_dense_outputs[label].append(vector)
        
            return total_loss, predicted_labels, correct_dense_outputs
        else:
            return total_loss, predicted_labels

    def compute_info_nce_loss(self, dense_output, labels, num_negatives=2):
        """
        Calculate the InfoNCE loss using batch processing and hard negative mining.

        Args:
            dense_output (torch.Tensor): Embedding vectors for the current batch, shaped as (batch_size, embedding_dim).
            labels (torch.Tensor): Labels for the current batch, shaped as (batch_size,).
            num_negatives (int): Number of negative samples per example.

        Returns:
            torch.Tensor: The computed InfoNCE loss.

        Strategy:
            - Positive samples: Prioritize class centers for the corresponding label; if not available, use the average of same-class samples within the batch; if still unavailable, skip the sample.
            - Negative samples: Use hard negative mining to select the most similar negative sample (from a different class) within the batch.
        """

        device = dense_output.device
        batch_size = dense_output.size(0)
        embedding_dim = dense_output.size(1)

        labels = labels.to(device)

        # get all available centers
        class_centers = self.memory_bank.get_class_centers()  # 返回字典 {class_label: class_center}

        # ==== positive sample ====
        positive_samples = []
        valid_indices = []  # available indices

        for i in range(batch_size):
            label = labels[i].item()
            if label in class_centers:
                positive_sample = class_centers[label]
            else:
                same_class_mask = (labels == label)
                same_class_mask[i] = False
                if same_class_mask.any():
                    positive_sample = dense_output[same_class_mask].mean(dim=0)
                else:
                    continue
            positive_samples.append(positive_sample)
            valid_indices.append(i)

        if len(valid_indices) == 0:
            return None

        anchor = dense_output[valid_indices]  # (num_valid, embedding_dim)
        positive_samples = torch.stack(positive_samples, dim=0)  # (num_valid, embedding_dim)
        valid_labels = labels[valid_indices]  # (num_valid,)

        num_valid = anchor.size(0)

        # ==== negative sample ====
        similarity_matrix = torch.zeros(num_valid, batch_size, device=device)
        for i in range(num_valid):
            similarity_matrix[i] = self.calculate_batch_wasserstein_dist(anchor[i], dense_output)

        mask = (valid_labels.unsqueeze(1) != labels.unsqueeze(0))  # (num_valid, batch_size)
        for idx_in_anchor, idx_in_batch in enumerate(valid_indices):
            mask[idx_in_anchor, idx_in_batch] = False

        all_neg_indices = []
        for i in range(num_valid):
            neg_indices_in_batch = mask[i].nonzero(as_tuple=False).view(-1)
            all_neg_indices.append(neg_indices_in_batch)

        negative_samples_list = []
        final_anchor = []
        final_positive_samples = []

        for i in range(num_valid):
            neg_indices = all_neg_indices[i]
            if neg_indices.numel() == 0:
                continue

            num_random_negatives = num_negatives // 2
            random_neg_indices = neg_indices[torch.randperm(neg_indices.size(0))[:num_random_negatives]]
            random_neg_samples = dense_output[random_neg_indices]

            num_hard_negatives = num_negatives - num_random_negatives
            neg_sim = similarity_matrix[i][neg_indices]
            if neg_sim.numel() > 0:
                num_neg = min(num_hard_negatives, neg_sim.size(0))
                topk = torch.topk(neg_sim, k=num_neg, largest=True)
                hard_neg_indices = neg_indices[topk.indices]
                hard_neg_samples = dense_output[hard_neg_indices]
            else:
                hard_neg_samples = torch.empty(0, embedding_dim).to(device)

            neg_samples = torch.cat([random_neg_samples, hard_neg_samples], dim=0)
            negative_samples_list.append(neg_samples)

            final_anchor.append(anchor[i])
            final_positive_samples.append(positive_samples[i])

        if len(final_anchor) == 0:
            return None

        anchor = torch.stack(final_anchor, dim=0)
        positive_samples = torch.stack(final_positive_samples, dim=0)

        max_negatives = max([neg.size(0) for neg in negative_samples_list])

        padded_negative_samples = []
        for neg_samples in negative_samples_list:
            num_neg = neg_samples.size(0)
            if num_neg < max_negatives:
                padding = torch.zeros(max_negatives - num_neg, embedding_dim, device=device)
                neg_samples = torch.cat([neg_samples, padding], dim=0)
            padded_negative_samples.append(neg_samples)
        negative_samples = torch.stack(padded_negative_samples, dim=0)

        self.memory_bank.update_class_centers()

        pos_sim = torch.sum(anchor * positive_samples, dim=-1, keepdim=True)

        anchor_expanded = anchor.unsqueeze(1)
        neg_sim = torch.bmm(anchor_expanded, negative_samples.transpose(1, 2)).squeeze(1)

        logits = torch.cat([pos_sim, neg_sim], dim=1)

        labels_i = torch.zeros(anchor.size(0), dtype=torch.long, device=device)

        loss = F.cross_entropy(logits, labels_i)

        return loss

    def calculate_batch_wasserstein_dist(self, X, Y):

        batch_size = Y.size(0)

        # 计算每个样本的 Wasserstein 距离
        wasserstein_distances = torch.zeros(batch_size, device=Y.device)
        for i in range(batch_size):
            wasserstein_distances[i] = self.calculate_2_wasserstein_dist(X, Y[i])

        return wasserstein_distances

    def calculate_2_wasserstein_dist(self, X, Y):
        if X.shape != Y.shape:
            raise ValueError("X and Y must be same shape")

        if X.dim() == 1:
            X, Y = X.unsqueeze(1).double(), Y.unsqueeze(1).double()
        else:
            X, Y = X.transpose(0, 1).double(), Y.transpose(0, 1).double()  # [n, b]

        mu_X, mu_Y = torch.mean(X, dim=1, keepdim=True), torch.mean(Y, dim=1, keepdim=True)  # [n, 1]
        n, b = X.shape
        fact = 1.0 if b < 2 else 1.0 / (b - 1)

        E_X = X - mu_X
        E_Y = Y - mu_Y
        cov_X = torch.matmul(E_X, E_X.t()) * fact  # [n, n]
        cov_Y = torch.matmul(E_Y, E_Y.t()) * fact

        C_X = E_X * math.sqrt(fact)
        C_Y = E_Y * math.sqrt(fact)
        M = torch.matmul(C_X.t(), C_Y)

        S = linalg.eigvals(M) + 1e-15
        sq_tr_cov = S.sqrt().abs().sum()

        trace_term = torch.trace(cov_X + cov_Y) - 2.0 * sq_tr_cov  # 标量

        diff = mu_X - mu_Y  # [n, 1]
        mean_term = torch.sum(torch.mul(diff, diff))  # 标量

        return (trace_term + mean_term).float()
    def forward(self, *input):
        def common_forward(outs):
            batch_size = outs.size(0)
            outs = outs.view(batch_size, -1)
            _, tag_seq = torch.max(outs, 1)
            return tag_seq

        outs, _ = self.word_hidden.sentence_representation(*input)

        if not self.openmax:
            if self.classifier:
                outs = self.classifier(outs)
            else:
                outs, _ = self.word_hidden.sentence_representation(*input)
            return common_forward(outs)
        else:
            if self.status != 'openmax_decode':
                outs, dense_outputs = self.classifier(outs)
                return common_forward(outs), dense_outputs
            if self.status == 'openmax_decode':
                outs, dense_outputs = self.classifier(outs)
                openmax_scores = self.compute_openmax_scores(dense_outputs)
                tag_seq = self.decode_openmax_scores(openmax_scores)
                return tag_seq, dense_outputs

    def decode_openmax_scores(self, openmax_scores):
        openmax_probs = torch.softmax(openmax_scores, dim=1)

        max_known_prob, y_star = torch.max(openmax_probs[:, 1:], dim=1)

        unknown_prob = openmax_probs[:, 0]
        #print("max_known_prob:", torch.mean(max_known_prob), torch.mean(unknown_prob), torch.max(max_known_prob))
        unknown_mask = (unknown_prob > max_known_prob)  #| (unknown_prob < float(self.epsilon))
        y_star[unknown_mask] = self.label_size

        return y_star

    def initialize_openmax(self):
        """
        Initialize OpenMax by building Weibull models and mean vectors.
        """
        openmax_processor = OpenMaxProcessing(self.data, self.openmax_yaml)
        self.labels_data = openmax_processor.labels_data
        self.alpha = openmax_processor.top_k
        self.epsilon = openmax_processor.threshold
        self.labels_data = openmax_processor.labels_data
        weibull_models, mean_vectors, top_weibull_model, top_mean_vectors = openmax_processor.build_weibull_models()
        return weibull_models, mean_vectors, top_weibull_model, top_mean_vectors

    def compute_openmax_scores(self, dense_outputs):
        batch_size = dense_outputs.size(0)
        num_labels = self.label_size
        openmax_scores = torch.zeros(batch_size, num_labels + 1).to(dense_outputs.device)
        for i in range(batch_size):
            openmax_scores[i] = self.recalibrate_scores(dense_outputs[i])

        return openmax_scores

    def euclidean_distance(self, mean_vector, dense_output):
        mean_vector_tensor = torch.tensor(mean_vector).float().to(dense_output.device)
        distance = F.pairwise_distance(dense_output, mean_vector_tensor, p=2)
        distance = torch.log1p(distance)
        return distance

    def wasserstein_distance_tensor(self, mean_vector, dense_output):
        mean_vector = mean_vector
        dense_output_np = dense_output.cpu().numpy()
        distances = wasserstein_distance(mean_vector, dense_output_np)
        distances_tensor = torch.tensor(distances).to(dense_output.device)
        log_distances = torch.log1p(distances_tensor)

        return log_distances

    def recalibrate_scores(self, dense_output):
        scores, top_alpha_idx = torch.topk(dense_output, self.alpha)

        v_hat = dense_output.clone().to(dense_output.device)

        unknown_scores = torch.zeros_like(dense_output).to(dense_output.device)

        ranked_alpha = torch.tensor([(self.alpha - i) / self.alpha for i in range(self.alpha)]).to(dense_output.device)

        # 计算修正后的激活值和未知类别评分
        for i in range(self.alpha):
            j = top_alpha_idx[i].item()
            weibull_model = self.weibull_models.get(str(j))
            mean_vector = self.mean_vectors.get(str(j))

            if weibull_model is None or mean_vector is None:
                wscore = 0.7
            else:
                distance = self.wasserstein_distance_tensor(mean_vector,
                                                            dense_output)  #self.euclidean_distance(mean_vector, dense_output)
                wscore = weibull_model.wscore(distance).item()

            v_hat[j] = dense_output[j] * (wscore * ranked_alpha[i])
            unknown_scores[j] = dense_output[j] * (1 - wscore * ranked_alpha[i])
        mother_scores = {}
        for mother_label, child_labels in self.labels_data.items():
            mother_weibull_model = self.top_weibull_model.get(mother_label)
            mother_mean_vector = self.top_mean_vectors.get(mother_label)
            if mother_weibull_model is None:
                mother_wscore = 0.7
            else:
                mother_distance = self.wasserstein_distance_tensor(mother_mean_vector,
                                                                   dense_output)  #self.euclidean_distance(mother_mean_vector, dense_output)
                mother_wscore = mother_weibull_model.wscore(mother_distance).item()
            mother_scores[mother_label] = mother_wscore

        for i in range(self.alpha):
            j = top_alpha_idx[i].item()
            mother_label = self.get_mother_label(j)
            if mother_label in mother_scores and j != mother_label:
                v_hat[j] = v_hat[j] * mother_scores[mother_label] * self.ranked_alpha[i]
                unknown_scores[j] = unknown_scores[j] * (1 - mother_scores[mother_label] * self.ranked_alpha[i])

        unknown_score = torch.sum(unknown_scores)

        openmax_scores = torch.cat([unknown_score.unsqueeze(0), v_hat])
        return openmax_scores

    def get_mother_label(self, child_label):
        for mother_label, child_labels in self.labels_data.items():
            if child_label in child_labels:
                return mother_label
        return None

    def get_target_probability(self, *input):
        # input = word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_word_text, mask
        word_inputs = input[0]
        outs, weights = self.word_hidden.sentence_representation(*input)
        batch_size = word_inputs.size(0)
        outs = outs.view(batch_size, -1)
        _, tag_seq = torch.max(outs, 1)
        outs = outs[:, 1:]
        sf = nn.Softmax(1)
        prob_outs = sf(outs)
        if self.gpu:
            prob_outs = prob_outs.cpu()
            if type(weights) != type(None):
                weights = weights.cpu()

        if type(weights) != type(None):
            weights = weights.detach().numpy()

        probs = np.insert(prob_outs.detach().numpy(), 0, 0, axis=1)

        return probs, weights
