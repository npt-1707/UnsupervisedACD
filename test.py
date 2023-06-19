from DataLoader import Dataset
from transformers import BertModel
import json
from transformers import BertModel, BertTokenizer
from nltk.corpus import stopwords
from Model import Unsupervised

bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

stop_words = set(stopwords.words('english'))

path = {
    "train": "data/SemEval'14-ABSA-TrainData_v2/Restaurants_Train_v2.xml",
    "test": "data/ABSA_TestData_PhaseB/Restaurants_Test_Data_phaseB.xml"
}

data = Dataset(path["train"], path["test"],
               bert_tokenizer.tokenize, stop_words)
model = Unsupervised(data, bert_model)


print(json.dumps(data.processed_train_sentences[:20], indent=4))
