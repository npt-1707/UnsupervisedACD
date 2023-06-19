import numpy as np
import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer


class Dataset():
    def __init__(self, train_path, test_path, tokenizer, stop_words):
        self.stop_words = stop_words
        self.tokenizer = tokenizer

        self.category_label_num = {
            'service': 0,
            'food': 1,
            'price': 2,
            'ambience': 3,
            'anecdotes/miscellaneous': 4
        }

        tree = ET.parse(train_path)
        root = tree.getroot()
        self.xml_train_sentences = root.findall('sentence')
        self.train_sentences = []
        self.train_labels = []
        for xml_sentence in self.xml_train_sentences:
            text, label = self.extract_xml_sentence(xml_sentence)
            self.train_sentences.append(text)
            self.train_labels.append(label)

        tree = ET.parse(test_path)
        root = tree.getroot()
        self.xml_test_sentences = root.findall('sentence')
        self.test_sentences = []
        self.test_labels = []
        for xml_sentence in self.xml_test_sentences:
            text, label = self.extract_xml_sentence(xml_sentence)
            self.test_sentences.append(text)
            self.test_labels.append(label)

        self.processed_train_sentences = self.preprocess(self.train_sentences)
        self.processed_test_sentences = self.preprocess(self.test_sentences)

    def extract_xml_sentence(self, xml_sentence):
        text = xml_sentence.find("text").text
        text = text.replace('$', ' price ')
        label = [0]*5
        xml_categories = xml_sentence.find("aspectCategories")
        if len(xml_categories) > 0:
            for xml_category in xml_categories:
                category = xml_category.attrib["category"]
                label[self.category_label_num[category]] = 1
        else:
            label[-1] = 1
        return text, label

    def __len__(self):
        return len(self.train_data)

    def preprocess(self, data):
        processed_sentences = []
        for sentence in data:
            # Tokenize the sentence
            tokens = self.tokenizer(sentence)
            # Convert tokens to lowercase
            tokens = [token.lower() for token in tokens]
            # Remove stopwords and punctuation
            tokens = [token for token in tokens if token.isalnum()
                      and token not in self.stop_words]
            # Join tokens back into a sentence
            preprocessed_sentence = ' '.join(tokens)
            # Add preprocessed sentence to the preprocessed corpus
            processed_sentences.append(preprocessed_sentence)
        return processed_sentences
