import numpy as np
import argparse
from utils import triplet_loss, extract_features
from keras.models import load_model
from sklearn.metrics import pairwise_distances

ap = argparse.ArgumentParser()
ap.add_argument("-f","--features-db", help="Path to extracted features")
ap.add_argument("-m","--model", help="Path to trained model")
args = vars(ap.parse_args())


def get_similar_image_indices(embeddings, index, num_results=4):
    query = embeddings[index]
    distances = pairwise_distances(query.reshape(1, -1), embeddings)
    indices = np.argsort(distances)[0][:num_results]
    return indices

def find_num_correct(true_indices, predicted_indices):
    num_correct = 0
    for i in true_indices:
        if i in predicted_indices:
            num_correct += 1
    return num_correct

model = load_model(args["model"],custom_objects={"triplet_loss":triplet_loss})
features, labels = extract_features(args["features_db"])
embeddings = model.predict([features, features, features])
embeddings = embeddings[:,:,2]
num_correct = 0

labels = (np.arange(1000)/4)+1

for i in range(1000):
    similar_indices = get_similar_image_indices(embeddings, i)
    true_indices = np.where(labels==(i/4)+1)[0].tolist()
    num_correct += find_num_correct(true_indices, similar_indices)/4.0

print "Accuracy", num_correct/1000.0