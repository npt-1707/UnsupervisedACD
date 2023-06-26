import numpy as np
import xml.etree.ElementTree as ET
from Preprocessing import Preprocessor
from tqdm import tqdm
from gensim.models import Word2Vec, KeyedVectors
import os


class Dataset():

    def __init__(self, name):
        print(f"Loading {name} dataset...")
        self.name = name
        self.preprocessor = Preprocessor()
        self.sentences = None
        self.labels = None
        self.category_label_num = {
            'staff': 0,
            'food': 1,
            'ambience': 2,
        }

        self.category_seed_words = {
            "staff": {"service", "staff", "friendly", "attentive", "manager"},
            "food": {"food", "delicious", "menu", "fresh", "tasty"},
            "ambience":
            {"ambience", "atmosphere", "decor", "romantic", "loud"},
        }

    def __len__(self):
        return len(self.sentences)

    def get_w2v_model(self, size=300):
        path = f"save/{self.name}_w2v_{size}.bin"
        if not os.path.exists("save"):
            os.mkdir("save")
        try:
            print("\tLoading word2vec model...")
            wv = KeyedVectors.load_word2vec_format(path, binary=True)
        except:
            print("\tTraining word2vec model...")
            w2v_model = Word2Vec(self.sentences, vector_size=size)
            wv = w2v_model.wv
            wv.save_word2vec_format(path, binary=True)
        return wv


class XMLDataset(Dataset):
    '''
    Extract sentences and labels from XML file. For SemEval 2014 dataset
    '''

    def __init__(self, name, path):
        super().__init__(name)

        convert = {"food": "food", "service": "staff", "ambience": "ambience"}

        tree = ET.parse(path)
        root = tree.getroot()
        self.xml_sentences = root.findall('sentence')
        self.sentences = []
        self.labels = []
        for xml_sentence in tqdm(self.xml_sentences):
            text, label = self.extract_xml_sentence(xml_sentence)
            if len(label) == 1 and label[0] in convert.keys():
                preprocessed_text = self.preprocessor.preprocess(text)
                if preprocessed_text != "":
                    self.sentences.append(preprocessed_text)
                    self.labels.append(self.category_label_num[convert[label[0]]])
                
    def extract_xml_sentence(self, xml_sentence):
        text = xml_sentence.find("text").text
        text = text.replace('$', ' price ')
        label = []
        xml_categories = xml_sentence.find("aspectCategories")
        for xml_category in xml_categories:
            category = xml_category.attrib["category"]
            label.append(category)
        return text, label


class TXTDataset(Dataset):
    '''
    Extract sentences from TXT file. For Yelp and Citysearch dataset
    '''

    def __init__(self, name, text_path, label_path=None, w2v=False):
        super().__init__(name)

        with open(text_path, 'r') as f:
            self.sentences = f.read().split("\n")[:-1]
        if label_path:
            with open(label_path, 'r') as f:
                self.labels = f.read().split("\n")[:-1]
            self.labels = [
                self.category_label_num[label.lower()]
                for label in tqdm(self.labels)
            ]

        if w2v:
            self.w2v_model = self.get_w2v_model()
