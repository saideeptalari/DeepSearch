from keras.applications import VGG16, InceptionV3, VGG19, ResNet50, Xception
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
import cv2
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers import Lambda, Input, Dense, GlobalAveragePooling2D, Merge, Dropout

class ImageNetFeatureExtractor(object):
    def __init__(self, model="vgg16", resize_to=(224,224)):
        MODEL_DICT = {"vgg16": VGG16, "vgg19": VGG19, "inception": InceptionV3, "resnet": ResNet50,
                      "xception": Xception}
        network = MODEL_DICT[model.lower()]
        self.model_name = model.lower()
        self.model = network(include_top=False)
        self.preprocess_input = imagenet_utils.preprocess_input
        self.imageSize = resize_to
        if model in ["inception","xception"]:
            self.preprocess_input = preprocess_input

    def extract(self, images):
        images = self.preprocess(images)
        return self.model.predict(images)

    @property
    def output_shape(self):
        return self.model.compute_output_shape([[None, self.imageSize[0], self.imageSize[1], 3]])

    def resize_images(self, images):
        images = np.array([cv2.resize(image, (self.imageSize[0], self.imageSize[1])) for image in images])
        return images

    def preprocess(self, images):
        images = self.resize_images(images)
        images = self.preprocess_input(images.astype("float"))
        return images


def concat_tensors(tensors, axis=-1):
    return K.concatenate([K.expand_dims(t, axis=axis) for t in tensors])


def get_small_network(input_shape=(None, 5,5,2048)):
    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=input_shape[1:]))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation="relu"))
    #model.add(Dense(128, activation="relu"))
    return model

def get_triplet_network(input_shape=(None, 5,5,2048)):
    base_model = get_small_network(input_shape=input_shape)

    anchor_input = Input(input_shape[1:])
    positive_input = Input(input_shape[1:])
    negative_input = Input(input_shape[1:])

    anchor_embeddings = base_model(anchor_input)
    positive_embeddings = base_model(positive_input)
    negative_embeddings = base_model(negative_input)

    output = Lambda(concat_tensors)([anchor_embeddings, positive_embeddings, negative_embeddings])
    model = Model([anchor_input, positive_input, negative_input], output)

    return model
