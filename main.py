from Model import UnsupervisedACD
import pickle

model = UnsupervisedACD(4, 1000000)
model.evaluate_testset()

# model.predict("The food is delicious. But I don't like the waiters.")