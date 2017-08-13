import numpy as np
import h5py
import argparse
from utils import triplet_loss, extract_features
from keras.models import load_model
from sklearn.metrics import pairwise_distances
import cv2
from imutils import build_montages

ap = argparse.ArgumentParser()
ap.add_argument("-m","--model", help="Path to trained model")
ap.add_argument("-f","--features-db", help="Path to saved features database")
ap.add_argument("-d","--dataset", help="Path to dataset")
ap.add_argument("-i","--image", help="Path to query image")

args = vars(ap.parse_args())

image_ids = h5py.File(args["features_db"], mode="r")["image_ids"][:]

def get_image_index():
    filename = args["image"].split("/")[-1]
    return np.where(image_ids == filename)[0][0]

def get_image_path(index):
    return args["dataset"].strip("/")+"/"+str(image_ids[index])

model = load_model(args["model"], custom_objects={"triplet_loss":triplet_loss})
features, labels = extract_features(args["features_db"])

embeddings = model.predict([features, features, features])
embeddings = embeddings[:,:,2]

image_id = get_image_index()
query = embeddings[image_id]

distances = pairwise_distances(query.reshape(1,-1), embeddings)
indices = np.argsort(distances)[0][:12]
images = [cv2.imread(get_image_path(index)) for index in indices]
images = [cv2.resize(image, (200,200)) for image in images]
result = build_montages(images, (200, 200), (4,3))[0]

cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()