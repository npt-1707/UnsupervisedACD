from Model import UnsupervisedACD
from DataLoader import *

# city_search = {
#     "train":
#     TXTDataset(name="city_search_train",
#                text_path="data/CitySearch/train.txt",
#                w2v=True),
#     "test":
#     TXTDataset(name="city_search_test",
#                text_path="data/CitySearch/test.txt",
#                label_path="data/CitySearch/test_label.txt")
# }

# yelp = {
#     "train":
#     TXTDataset(name="yelp_train",
#                text_path="data/Yelp/yelp_restaurant_review.txt",
#                w2v=True)
# }

# sem_eval = {
#     "train":
#     XMLDataset(
#         name="sem_eval_train",
#         path="data/SemEval'14-ABSA-TrainData_v2/Restaurants_Train_v2.xml"),
#     "test":
#     XMLDataset(
#         name="sem_eval_test",
#         path="data/ABSA_TestData_PhaseB/Restaurants_Test_Data_phaseB.xml")
# }
# import random
# random.seed(0)
# for num in range(3, 21):
#     model = UnsupervisedACD(sem_eval["train"], num_clusters=num, max_iter=100)
#     model.fit(city_search["train"], 10000)
#     model.validate(sem_eval["train"])
#     model.evaluate(city_search["test"])
#     model.evaluate(sem_eval["test"])
#     UnsupervisedACD.save(model.save_path, model)

save_path = "save/model_sem_eval_train_11_city_search_train_sem_eval_train.pkl"
model = UnsupervisedACD.load(save_path)
sentence = "The design and atmosphere is just as good."
print(model.predict(sentence))