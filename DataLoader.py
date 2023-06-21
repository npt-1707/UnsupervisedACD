import numpy as np
import xml.etree.ElementTree as ET
from Preprocessing import Preprocessor
from tqdm import tqdm


class Dataset():

    def __init__(self, path):
        self.preprocessor = Preprocessor()

    def preprocess(self, data):
        processed_sentences = []
        for sentence in tqdm(data):
            processed_sentences.append(self.preprocessor.preprocess(sentence))
        return processed_sentences


class XMLDataset(Dataset):
    '''
    Extract sentences and labels from XML file
    '''

    def __init__(self, path):
        super().__init__(path)

        self.category_label_num = {
            'service': 0,
            'food': 1,
            'price': 2,
            'ambience': 3,
            'anecdotes/miscellaneous': 4
        }

        tree = ET.parse(path)
        root = tree.getroot()
        self.xml_sentences = root.findall('sentence')
        self.sentences = []
        self.labels = []
        for xml_sentence in self.xml_sentences:
            text, label = self.extract_xml_sentence(xml_sentence)
            self.sentences.append(text)
            self.labels.append(label)
        self.processed_sentences = self.preprocess(self.sentences)

    def __len__(self):
        return len(self.sentences)

    def extract_xml_sentence(self, xml_sentence):
        text = xml_sentence.find("text").text
        text = text.replace('$', ' price ')
        label = [0] * 5
        xml_categories = xml_sentence.find("aspectCategories")
        if len(xml_categories) > 0:
            for xml_category in xml_categories:
                category = xml_category.attrib["category"]
                label[self.category_label_num[category]] = 1
        else:
            label[-1] = 1
        return text, label


class TXTDataset(Dataset):
    '''
    Extract sentences from TXT file
    '''

    def __init__(self, path):
        super().__init__(path)
        with open(path, 'r') as f:
            self.processed_sentences = f.readlines()
        # self.processed_sentences = self.preprocess(self.sentences)
