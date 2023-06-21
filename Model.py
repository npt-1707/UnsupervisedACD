import gensim.downloader as api
from gensim.models import Word2Vec
from gensim import models
from gensim.corpora import Dictionary
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
from gensim.models import TfidfModel

import numpy as np
from tqdm import tqdm
import pickle, os
from sklearn.cluster import KMeans
from sklearn.metrics import hamming_loss, precision_score, recall_score, f1_score

from DataLoader import XMLDataset, TXTDataset
from Preprocessing import Preprocessor


def sigmoid(x, derivative=False):
    return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))


class UnsupervisedACD:

    def __init__(self):
        print("Initializing ...")
        self.category_label_num = {
            "service": 0,
            "food": 1,
            "price": 2,
            "ambience": 3,
            "anecdotes/miscellaneous": 4,
        }
        self.category_num_label = {
            0: "service",
            1: "food",
            2: "price",
            3: "ambience",
            4: "anecdotes/miscellaneous",
        }
        self.category_seed_words = {
            "service":
            {"service", "staff", "friendly", "attentive", "manager"},
            "food": {"food", "delicious", "menu", "fresh", "tasty"},
            "price": {"price", "cheap", "expensive", "money", "affordable"},
            "ambience":
            {"ambience", "atmosphere", "decor", "romantic", "loud"},
        }
        self.categories = [
            "service",
            "food",
            "price",
            "ambience",
            "anecdotes/miscellaneous",
        ]
        # processor for preprocessing sentences
        self.processor = Preprocessor()

        # load data
        print("Loading yelp dataset ...")
        self.yelp_dataset = TXTDataset("data/yelp_restaurant_review.txt")
        print("Loading SemEval dataset ...")
        self.semeval_train_dataset = XMLDataset(
            "data/SemEval'14-ABSA-TrainData_v2/Restaurants_Train_v2.xml")
        self.semeval_test_dataset = XMLDataset(
            "data/ABSA_TestData_PhaseB/Restaurants_Test_Data_phaseB.xml")

        # load models
        print("Loading word2vec model ...")
        if not os.path.exists("save"):
            os.mkdir("save")

        self.w2v_model = models.KeyedVectors.load_word2vec_format(
            'save/yelp_W2V_300_orig.bin', binary=True)

        self.corpus = [
            sentence.split()
            for sentence in self.semeval_train_dataset.processed_sentences
        ]
        self.dictionary = Dictionary(self.corpus)
        self.tfidf = TfidfModel(dictionary=self.dictionary)
        self.similarity_index = WordEmbeddingSimilarityIndex(self.w2v_model)
        print("Building similarity matrix ...")
        self.similarity_matrix = SparseTermSimilarityMatrix(
            self.similarity_index, self.dictionary, self.tfidf)
        if os.path.exists("save/model.pkl"):
            self.load()
        else:
            # embedding yelp sentences
            print("Embedding yelp sentences ...")
            self.embedded = [
                self.sentence_embedd_average(sentence)
                for sentence in tqdm(self.yelp_dataset.processed_sentences)
            ]

            # cluster yelp sentences
            self.k_mean_clustering_yelp()

            # get cluster similarity score with categories
            self.get_cluster_category_score()

            # save clusters
            self.save()

        # test performance on test set
        self.evaluate_testset()

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

    def k_mean_clustering_yelp(self, k=12):
        '''
        Cluster embedded yelp sentences by k-means 
        '''
        print("Clustering ...")
        self.kmeans = KMeans(n_clusters=k, max_iter=100,
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
        Get the similarity scores of yelp clusters with categories
        '''
        print("Getting cluster similarity score with categories ...")
        self.cluster_category_similarity = []
        for cluster_index in len(self.centroids):
            # sentences in this cluster
            cluster_sentences = [
                self.yelp_dataset.processed_sentences[i]
                for i in range(self.cluster_indexes)
                if self.cluster_indexes[i] == cluster_index
            ]
            # similarity of this cluster to each category
            cluster_scores = [
                np.mean([
                    self.get_sentence_category_scores(sentence)
                    for sentence in cluster_sentences
                ],
                        axis=0)
            ]

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
        pred_cluster = self.kmeans.predict(embedded_sentence)
        cluster_scores = self.cluster_category_similarity[pred_cluster]
        sentence_scores = self.get_sentence_category_scores(sentence)
        #nomalize cluster scores and sentence scores by np.linagl.norm
        cluster_scores = cluster_scores / np.linalg.norm(cluster_scores)
        sentence_scores = sentence_scores / np.linalg.norm(sentence_scores)
        scores = (1 - alpha) * cluster_scores + alpha * sentence_scores
        return scores, sentence_scores, cluster_scores

    def evaluate_testset(self, alpha=0.5):
        '''
        Evaluate the performance of model on testset
        '''
        print("Predicting ...")
        self.results = {"predict": [], "true": []}
        for idx in tqdm(range(len(self.semeval_test_dataset))):
            sentence = self.semeval_test_dataset.processed_sentences[idx]
            scores, _, _ = self.get_test_sentence_scores(sentence,
                                                         alpha=alpha,
                                                         is_processed=False)
            labels = self.semeval_test_dataset.labels[idx]
            self.results["predict"].append(scores)
            self.results["true"].append(labels)

        print("Evaluating ...")
        # Calculate Hamming Loss
        hamming_loss_value = hamming_loss(self.results["true"],
                                          self.results["predict"])
        print("Hamming Loss:", hamming_loss_value)

        # Calculate Precision
        precision = precision_score(self.results["true"],
                                    self.results["predict"],
                                    average='micro')
        print("Precision:", precision)

        # Calculate Recall
        recall = recall_score(self.results["true"],
                              self.results["predict"],
                              average='micro')
        print("Recall:", recall)

        # Calculate F1-Score
        f1 = f1_score(self.results["true"],
                      self.results["predict"],
                      average='micro')
        print("F1-Score:", f1)

    def predict(self, sentence, threshold=0.5):
        '''
        Predict the categories of a sentence
        '''
        print(sentence)
        scores, _, _ = self.get_test_sentence_scores(sentence)
        labels = []
        for idx in range(len(self.categories[:-1])):
            print(f"{self.categories[idx]}: {scores[idx]}")
            if scores[idx] > threshold:
                labels.append(self.categories[idx])
        if not labels:
            labels.append(self.categories[-1])
        print("Predicted labels:", labels)

    def save(self):
        state_dict = {
            "kmeans": self.kmeans,
            "cluster_score": self.cluster_category_similarity
        }

        with open("save/model.pkl", 'wb') as f:
            pickle.dump(state_dict, f)

    def load(self):
        with open("save/model.pkl", 'rb') as f:
            state_dict = pickle.load(f)
        self.kmeans = state_dict["kmeans"]
        self.cluster_category_similarity = state_dict["cluster_score"]