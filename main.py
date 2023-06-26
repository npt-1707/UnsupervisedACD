from Model import UnsupervisedACD
from DataLoader import *

model = UnsupervisedACD(4, 1000000)
model.evaluate_testset()

city_search = {
    "train":
    TXTDataset(name="city_search_train",
               text_path="data/CitySearch/train.txt",
               w2v=True),
    "test":
    TXTDataset(name="city_search_test",
               text_path="data/CitySearch/test.txt",
               label_path="data/CitySearch/test_label.txt")
}

yelp = {
    "train":
    TXTDataset(name="yelp_train",
               text_path="data/Yelp/yelp_restaurant_review.txt",
               w2v=True)
}

sem_eval = {
    "train":
    XMLDataset(
        name="sem_eval_train",
        path="data/SemEval'14-ABSA-TrainData_v2/Restaurants_Train_v2.xml"),
    "test":
    XMLDataset(
        name="sem_eval_test",
        path="data/ABSA_TestData_PhaseB/Restaurants_Test_Data_phaseB.xml")
}

corpus = [sentence.split() for sentence in sem_eval["train"].sentences]

model = UnsupervisedACD(corpus, num_clusters=4, max_iter=100)

model.fit(city_search["train"])

model.validate(sem_eval["train"])

model.test(city_search["test"])