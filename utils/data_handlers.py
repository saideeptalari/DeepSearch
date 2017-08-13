import numpy as np
import h5py
import glob
import cv2

def get_triplets(data, labels):
    pos_label, neg_label = np.random.choice(labels, 2, replace=False)

    pos_indexes = np.where(labels == pos_label)[0]
    neg_indexes = np.where(labels == neg_label)[0]

    np.random.shuffle(pos_indexes)
    np.random.shuffle(neg_indexes)

    anchor = data[pos_indexes[0]]
    positive = data[pos_indexes[-1]]
    negative = data[neg_indexes[0]]

    return (anchor, positive, negative)

def dump_features(images_dir,labels, hdf5_path, feature_extractor, image_formats=("jpg")):

    image_paths = []

    for image_format in image_formats:
        image_paths += glob.glob("{}/*.{}".format(images_dir, image_format))

    image_paths = sorted(image_paths)
    db = h5py.File(hdf5_path, mode="w")

    features_shape = ((len(labels),), feature_extractor.output_shape[1:])
    features_shape = [dim for sublist in features_shape for dim in sublist]

    imageIDDB = db.create_dataset("image_ids", shape=(len(labels),), 
        dtype=h5py.special_dtype(vlen=unicode))
    featuresDB = db.create_dataset("features", 
        shape=features_shape, dtype="float")
    labelsDB = db.create_dataset("labels", 
        shape=(len(labels),), dtype="int")

    for i in range(0, len(labels), 5):
        start,end = i, i+5
        image_ids = [path.split("/")[-1] for path in image_paths[start:end]]
        images = [cv2.imread(path,1) for path in image_paths[start:end]]
        features = feature_extractor.extract(images)

        imageIDDB[start:end] = image_ids
        featuresDB[start:end] = features
        labelsDB[start:end] = labels[start:end]
        print "Extracting {}/{}".format(i+5, len(labels))

    db.close()

def extract_features(hdf5_path):
    db = h5py.File(hdf5_path,mode="r")
    features = db["features"][:]
    labels = db["labels"][:]

    return (features, labels)

def extract_embeddings(features, model):
    embeddings = model.predict([features, features, features])
    return embeddings[:,:,0]
