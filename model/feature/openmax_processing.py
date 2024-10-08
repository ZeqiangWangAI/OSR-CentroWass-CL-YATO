import json
import yaml
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.stats import wasserstein_distance
from .DistributionModels.weibull import weibull
import ot

class OpenMaxProcessing:
    def __init__(self, data, openmax_config_file):

        with open(openmax_config_file, 'r') as file:
            self.openmax_config = yaml.safe_load(file)
        self.file_name = self.openmax_config['file_name']
        self.tail_percentage = self.openmax_config['tail_percent']
        self.top_k = self.openmax_config['top_k']
        self.threshold = self.openmax_config['threshold']
        self.border_size = self.openmax_config['border_size']
        self.labels_data = self.openmax_config.get('label', {})
        self.gpu = self.openmax_config.get('gpu', 0)  # Default to 0 if 'gpu' is not specified
        self.label_alphabet = data.label_alphabet
        self.weibull_models = {}
        self.mean_vectors = {}
        self.top_weibull_model = {}
        self.top_mean_vectors = {}

    def openmax_dense_vector_json_load(self):
        """Load a JSON file and return the dictionary."""
        with open(self.file_name, 'r') as json_file:
            data = json.load(json_file)
        return data

    #def compute_mean_and_euclidean_distance(self, vectors):
    #    mean_vector = np.mean(vectors, axis=0)
    #
    #    distances = euclidean_distances([mean_vector], vectors)[0]
    #
    #    log_distances = np.log1p(distances)
    #
    #    return mean_vector, log_distances
    def compute_mean_and_wasserstein_distance(self, vectors):
        """计算向量的均值和Wasserstein距离"""
        mean_vector = np.mean(vectors, axis=0)

        # 计算Wasserstein距离
        distances = np.array([
            wasserstein_distance(mean_vector, vector)
            for vector in vectors
        ])

        log_distances = np.log1p(distances)  # 使用 log1p 可避免 log(0) 的问题

        return mean_vector, log_distances

    def openmax_weibull_distribution_builder(self):
        data = self.openmax_dense_vector_json_load()

        for key, value in data.items():
            data[key] = np.array(value)

        #total_vectors = np.array([vector for vectors in data.values() for vector in vectors])

        weibull_build_data = {}
        mother_vectors = {}

        for category, vectors in data.items():
            if len(vectors) < 1:
                print(f"Skipping category {category} as it has fewer than 4 vectors.")

                self.mean_vectors[category] = None
                self.weibull_models[category] = None
                continue

            mean_vector, distance = self.compute_mean_and_wasserstein_distance(vectors) #self.compute_mean_and_euclidean_distance(vectors)
            tail_points = distance

            weibull_build_data[category] = {
                'mean_vector': mean_vector,
                'distance': distance,
                'tail_points': tail_points
            }
            category_label = self.label_alphabet.get_instance(int(category))
            mother_label = None
            for mother, children in self.labels_data.items():
                if category_label in children:
                    mother_label = mother
                    break

            if mother_label:
                if mother_label not in mother_vectors:
                    mother_vectors[mother_label] = []
                mother_vectors[mother_label].extend(vectors)

        for mother_label, vectors in mother_vectors.items():
            mean_vector, distance = self.compute_mean_and_wasserstein_distance(vectors)#self.compute_mean_and_euclidean_distance(vectors)#self.compute_mean_and_wasserstein_distance(vectors)
            #self.compute_mean_and_euclidean_distance(vectors)
            tail_points = distance

            weibull_build_data[mother_label] = {
                'mean_vector': mean_vector,
                'distance': distance,
                'tail_points': tail_points
            }

        return weibull_build_data

    def fit_weibull_distributions(self, weibull_build_data):
        for category, data in weibull_build_data.items():
            tail_points = data['tail_points']
            if int(len(tail_points) * self.tail_percentage) <= self.border_size:
                print(f"Warning: Not enough tail points for category {category}. Skipping.")
                if category in self.labels_data.keys():
                    self.top_weibull_model[category] = None
                    self.top_mean_vectors[category] = None
                else:
                    self.weibull_models[category] = None
                    self.mean_vectors[category] = None
            else:
                tail_points_tensor = torch.tensor(tail_points).unsqueeze(0).cuda(self.gpu)
                weibull_model = weibull()
                weibull_model.FitLowNormalized(tail_points_tensor, int(len(tail_points) * self.tail_percentage), isSorted=False, gpu=self.gpu)
                if category in self.labels_data.keys():
                    self.top_weibull_model[category] = weibull_model
                    self.top_mean_vectors[category] = data['mean_vector']
                else:
                    self.weibull_models[category] = weibull_model
                    self.mean_vectors[category] = data['mean_vector']

    def build_weibull_models(self):
        weibull_build_data = self.openmax_weibull_distribution_builder()
        self.fit_weibull_distributions(weibull_build_data)
        return self.weibull_models, self.mean_vectors, self.top_weibull_model, self.top_mean_vectors
