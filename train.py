import argparse
from models import get_triplet_network
from utils import extract_features, triplet_loss, get_triplets
import numpy as np
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint


ap = argparse.ArgumentParser()
ap.add_argument("-f","--features-db", help="Path to saved features db")
ap.add_argument("-o","--output", help="Path to save the model checkpoints")
args = vars(ap.parse_args())

features, labels = extract_features(args["features_db"])
print "[+] Finished loading extracted features"

model = get_triplet_network(features.shape)

data = []
for i in range(len(features)*5):
    anchor, positive, negative = get_triplets(features, labels)
    data.append([anchor, positive, negative])

data = np.array(data)
targets = np.zeros(shape=(5000,256,3))

callback = ModelCheckpoint(args["output"], period=1, monitor="val_loss")
X_train, X_test, Y_train, Y_test = train_test_split(data, targets)

model.compile(Adam(1e-4), triplet_loss)
model.fit([X_train[:,0], X_train[:,1], X_train[:,2]], Y_train, epochs=10,
          validation_data=([X_test[:,0], X_test[:,1], X_test[:,2]], Y_test),
          callbacks=[callback], batch_size=32)
