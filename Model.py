import gensim.downloader as api
from gensim.models import Word2Vec
from gensim import models
from gensim.corpora import Dictionary
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
from gensim.models import TfidfModel

import numpy as np
from tqdm import tqdm
import pickle, os, random
from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn.metrics import hamming_loss, precision_score, recall_score, f1_score

from DataLoader import XMLDataset, TXTDataset
from Preprocessing import Preprocessor


def sigmoid(x, derivative=False):
    return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))


class UnsupervisedACD:

    def __init__(self, num_clusters=12, max_iter=100):
        print("Initializing ...")
        # number of clusters
        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self.processor = Preprocessor()

    def fit(self, dataset, sample=None):
        self.save_path = f"save/{dataset.name}_{self.num_clusters}_model.pkl"
        if os.path.exists(self.save_path):
            print("Model exists, loading ...")
            self.load()
            return
        self.corpus = [sentence.split() for sentence in dataset.sentences]
        self.dictionary = Dictionary(self.corpus)
        self.tfidf = TfidfModel(dictionary=self.dictionary)
        self.categories = list(dataset.category_label_num.keys())
        self.category_seed_words = dataset.category_seed_words
        self.w2v_model = dataset.w2v_model
        self.sentences = dataset.sentences

        self.similarity_index = WordEmbeddingSimilarityIndex(self.w2v_model)
        print("Building similarity matrix ...")
        self.similarity_matrix = SparseTermSimilarityMatrix(
            self.similarity_index, self.dictionary, self.tfidf)

        print("Embedding sentences ...")
        if sample is None:
            self.embedded = [
                self.sentence_embedd_average(sentence)
                for sentence in tqdm(dataset.sentences)
            ]
        else:
            self.embedded = [
                self.sentence_embedd_average(sentence)
                for sentence in tqdm(random.sample(dataset.sentences, sample))
            ]

        # cluster sentences
        self.k_means_clustering()

        # get cluster similarity score with categories
        self.get_cluster_category_score()

        self.save()

    def sentence_embedd_average(self, sentence):
        '''
        Get the average word embedding of a sentence
        '''
        return np.mean(
            [
                self.w2v_model[word]
                for word in sentence if word in self.w2v_model
            ],
            axis=0,
        )

    def k_means_clustering(self):
        '''
        Cluster embedded sentences by k-means 
        '''
        print("Clustering ...")
        self.kmeans = KMeans(n_clusters=self.num_clusters,
                             max_iter=self.max_iter,
                             random_state=0).fit(self.embedded)
        self.cluster_indexes = self.kmeans.labels_
        self.centroids = self.kmeans.cluster_centers_

    def get_sentence_category_similarity(self, sentence, seeds):
        '''
        Get the average soft cosine similarity of a sentence with seed words
        '''
        results = [
            self.similarity_matrix.inner_product(
                self.dictionary.doc2bow(sentence.split()),
                self.dictionary.doc2bow([seed]),
            ) for seed in seeds
        ]
        return np.mean(results)

    def get_sentence_category_score(self, sentence, category):
        '''
        Get the similarity score of a sentence with a category
        '''
        if category in self.category_seed_words:
            return sigmoid(
                self.get_sentence_category_similarity(
                    sentence, self.category_seed_words[category]))
        return 0.0

    def get_sentence_category_scores(self, sentence):
        '''
        Get the similarity scores of a sentence with categories
        '''
        return np.array([
            self.get_sentence_category_score(sentence, category)
            for category in self.categories
        ])

    def get_cluster_category_score(self):
        '''
        Get the similarity scores of clusters with categories
        '''
        print("Getting cluster similarity score with categories ...")
        self.cluster_category_similarity = []
        for cluster_index in range(len(self.centroids)):
            print(f"Cluster {cluster_index} ...", end="")
            # sentences in this cluster
            cluster_sentences = [
                self.sentences[i]
                for i, centroid_idx in enumerate(self.cluster_indexes)
                if centroid_idx == cluster_index
            ]
            # similarity of this cluster to each category
            cluster_scores = np.mean([
                self.get_sentence_category_scores(sentence)
                for sentence in cluster_sentences
            ],
                                     axis=0)
            print(cluster_scores)
            self.cluster_category_similarity.append(cluster_scores)

    def get_test_sentence_scores(self,
                                 sentence,
                                 alpha=0.5,
                                 is_processed=False):
        '''
        Get the predicted scores of a sentence with categories
        '''
        if not is_processed:
            sentence = self.processor.preprocess(sentence)
        embedded_sentence = self.sentence_embedd_average(sentence)
        predict_cluster = self.kmeans.predict([embedded_sentence])[0]
        cluster_scores = self.cluster_category_similarity[predict_cluster]
        sentence_scores = self.get_sentence_category_scores(sentence)
        #nomalize cluster scores and sentence scores
        cluster_scores = cluster_scores / np.linalg.norm(cluster_scores)
        sentence_scores = sentence_scores / np.linalg.norm(sentence_scores)
        scores = (1 - alpha) * cluster_scores + alpha * sentence_scores
        return scores, sentence_scores, cluster_scores

    def predict(self, sentence, is_processed, alpha=None, threshold=None):
        '''
        Predict the categories of a sentence
        '''
        if not alpha:
            alpha = self.alpha
        if not threshold:
            threshold = self.threshold

        print(sentence)
        scores, _, _ = self.get_test_sentence_scores(sentence,
                                                     alpha=alpha,
                                                     is_processed=is_processed)
        print(scores)
        labels = []
        for idx in range(len(self.categories)):
            print(f"{self.categories[idx]}: {scores[idx]}")
            if scores[idx] > threshold:
                labels.append(self.categories[idx])
        print("Predicted labels:", labels)

    def find_best_threshold(self, predicted_scores, gtruth_labels):
        best_res = [0.0] * 4
        threshold = 0.2
        while threshold < 0.7:
            TP = 0
            FP = 0
            FN = 0
            for i in range(len(gtruth_labels)):
                pred_labels = deepcopy(predicted_scores[i])
                ground_truth = deepcopy(gtruth_labels[i])
                for idx in range(len(pred_labels)):
                    if pred_labels[idx] > threshold:
                        pred_labels[idx] = 1
                    else:
                        pred_labels[idx] = 0
                # if np.any(pred_labels) == 0:
                #     pred_labels[4] = 1
                for idx in range(len(pred_labels)):
                    if pred_labels[idx] == 1:
                        if idx in ground_truth:
                            TP += 1
                        else:
                            FP += 1
                    else:
                        if idx in ground_truth:
                            FN += 1
            try:
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                f1 = 2 * (precision * recall) / (precision + recall)
            except ZeroDivisionError:
                f1 = 0.0
            if f1 > best_res[0]:
                best_res = [f1, precision, recall, threshold]
            threshold.round(4)
            threshold += 0.0001
        return best_res

    def validate(self, dataset):
        print("Validating ...")
        alpha = 0.1
        best = [0.0] * 4
        while alpha < 1.0:
            self.results = {"predict": [], "true": []}
            for idx in range(len(dataset)):
                sentence = dataset.sentences[idx]
                scores, _, _ = self.get_test_sentence_scores(sentence,
                                                             alpha=alpha,
                                                             is_processed=True)
                labels = dataset.labels[idx]
                self.results["predict"].append(scores)
                self.results["true"].append(labels)
            self.results["predict"] = np.array(self.results["predict"])
            self.results["true"] = np.array(self.results["true"])

            best_res = self.find_best_threshold(self.results["predict"],
                                                self.results["true"])
            if best_res[0] > best[0]:
                best = best_res
                self.alpha = alpha
                self.threshold = best_res[3]
                print(
                    f"Alpha: {alpha} - Threshold: {self.threshold} - Best result:",
                    best[:3])
            alpha += 0.1

    def save(self):
        state_dict = {
            "dictionary": self.dictionary,
            "similarity_matrix": self.similarity_matrix,
            "kmeans": self.kmeans,
            "cluster_score": self.cluster_category_similarity
        }

        with open(self.save_path, 'wb') as f:
            pickle.dump(state_dict, f)

    def load(self):
        with open(self.save_path, 'rb') as f:
            state_dict = pickle.load(f)
        self.dictionary = state_dict["dictionary"]
        self.similarity_matrix = state_dict["similarity_matrix"]
        self.kmeans = state_dict["kmeans"]
        self.cluster_category_similarity = state_dict["cluster_score"]

    def evaluate(self, dataset):
        '''
        Evaluate the performance of model on testset
        '''
        print("Predicting ...")
        self.results = {"predict": [], "true": []}
        for idx in tqdm(range(len(self.dataset))):
            sentence = self.dataset.sentences[idx]
            scores, _, _ = self.get_test_sentence_scores(sentence,
                                                         alpha=self.alpha,
                                                         is_processed=False)
            labels = self.dataset.labels[idx]
            self.results["predict"].append(scores)
            self.results["true"].append(labels)
        self.results["predict"] = np.array(self.results["predict"])
        self.results["true"] = np.array(self.results["true"])
        print("Evaluating ...")
        binary_predictions = (self.results["predict"]
                              >= self.threshold).astype(int)

        # Calculate Hamming Loss
        hamming_loss_value = hamming_loss(self.results["true"],
                                          binary_predictions)
        print("Hamming Loss:", hamming_loss_value)

        # Calculate Precision
        precision_micro = precision_score(self.results["true"],
                                          binary_predictions,
                                          average='micro')
        print("Precision (Micro):", precision_micro)

        precision_macro = precision_score(self.results["true"],
                                          binary_predictions,
                                          average='macro')
        print("Precision (Macro):", precision_macro)

        # Calculate Recall
        recall_micro = recall_score(self.results["true"],
                                    binary_predictions,
                                    average='micro')
        print("Recall (Micro):", recall_micro)

        recall_macro = recall_score(self.results["true"],
                                    binary_predictions,
                                    average='macro')
        print("Recall (Macro):", recall_macro)

        # Calculate F1-Score
        f1_micro = f1_score(self.results["true"],
                            binary_predictions,
                            average='micro')
        print("F1-Score (Micro):", f1_micro)

        f1_macro = f1_score(self.results["true"],
                            binary_predictions,
                            average='macro')
        print("F1-Score (Macro):", f1_macro)