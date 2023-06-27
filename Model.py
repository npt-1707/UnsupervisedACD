from gensim.corpora import Dictionary
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
from gensim.models import TfidfModel

import numpy as np
from tqdm import tqdm
import pickle, os, random
from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from Preprocessing import Preprocessor


def sigmoid(x, derivative=False):
    return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))


class UnsupervisedACD:

    def __init__(self, corpus_dataset, num_clusters=12, max_iter=100):
        print("Initializing ...")
        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self.processor = Preprocessor()

        corpus = [sentence.split() for sentence in corpus_dataset.sentences]
        self.corpus_name = corpus_dataset.name
        self.dictionary = Dictionary(corpus)
        self.tfidf = TfidfModel(dictionary=self.dictionary)
        self.save_path = f"save/model_{corpus_dataset.name}_{self.num_clusters}"
        if not os.path.exists("save"):
            os.mkdir("save")

    def fit(self, dataset, sample=None):
        self.save_path += f"_{dataset.name}"
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
                             random_state=0,
                             n_init="auto").fit(self.embedded)
        self.cluster_indexes = self.kmeans.labels_
        self.centroids = self.kmeans.cluster_centers_

    def get_sentence_category_similarity(self, sentence, seeds):
        '''
        Get the average soft cosine similarity of a sentence with seed words
        '''
        results = [
            self.similarity_matrix.inner_product(
                self.tfidf[self.dictionary.doc2bow(sentence.split())],
                self.tfidf[self.dictionary.doc2bow([seed])], (True, True))
            for seed in seeds
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
            cluster_scores = []
            for category in self.categories:
                cluster_score = np.mean([
                    self.get_sentence_category_similarity(
                        sentence, self.category_seed_words[category])
                    for sentence in cluster_sentences
                ],
                                        axis=0)
                cluster_score = sigmoid(cluster_score)
                cluster_scores.append(cluster_score)
            cluster_scores = np.array(cluster_scores)
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

    def predict(self, sentence, is_processed=False, alpha=0.5):
        '''
        Predict the categories of a sentence
        '''
        if hasattr(self, 'alpha'):
            alpha = self.alpha
        scores = self.get_test_sentence_scores(sentence,
                                               alpha=alpha,
                                               is_processed=is_processed)[0]
        return scores

    def validate(self, dataset):
        print("Validating ...")
        best_res = [0.0] * 3
        for alpha in range(2, 9):
            self.results = {"predict": [], "true": []}
            for idx in range(len(dataset)):
                sentence = dataset.sentences[idx]
                scores, _, _ = self.get_test_sentence_scores(sentence,
                                                             alpha=alpha / 10,
                                                             is_processed=True)
                label = dataset.labels[idx]
                self.results["predict"].append(np.argmax(scores))
                self.results["true"].append(label)

            res = [
                f1_score(self.results["true"],
                         self.results["predict"],
                         average='macro'),
                precision_score(self.results["true"],
                                self.results["predict"],
                                average='macro'),
                recall_score(self.results["true"],
                             self.results["predict"],
                             average='macro')
            ]
            if res[0] > best_res[0]:
                best_res = res
                self.alpha = alpha

            print(f"Alpha: {alpha} - Result: {res}")
        self.save_path += f"_{dataset.name}.pkl"

    def save(save_path, model):
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)

    def load(save_path):
        with open(save_path, 'rb') as f:
            model = pickle.load(f)
        return model

    def evaluate(self, dataset):
        '''
        Evaluate the performance of model on testset
        '''
        print("Predicting ...")
        self.results = {"predict": [], "true": []}
        if hasattr(self, 'alpha'):
            alpha = self.alpha
        for idx in tqdm(range(len(dataset))):
            sentence = dataset.sentences[idx]
            scores, _, _ = self.get_test_sentence_scores(sentence,
                                                         alpha=alpha,
                                                         is_processed=False)
            label = dataset.labels[idx]
            label = dataset.labels[idx]
            self.results["predict"].append(np.argmax(scores))
            self.results["true"].append(label)

        print(
            classification_report(self.results["true"],
                                  self.results["predict"],
                                  target_names=self.categories))
