import argparse
import numpy as np
from models import ImageNetFeatureExtractor
from utils import dump_features

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset", help="Path to dataset", required=True)
ap.add_argument("-f","--features-db", help="Path to save extracted features", required=True)
ap.add_argument("-m","--model", help="(VGG16, VGG19, Inceptionv3, ResNet50)", default="InceptionV3")
ap.add_argument("-r","--resize", help="resize to", default=229, type=int)

args = vars(ap.parse_args())

feature_extractor = ImageNetFeatureExtractor(model=args["model"],
                                             resize_to=(int(args["resize"]), int(args["resize"])))
print "[+] Successfully loaded pre-trained model"

dump_features(args["dataset"], labels=(np.arange(1000)/4)+1,
              hdf5_path=args["features_db"], feature_extractor=feature_extractor,
              image_formats=("jpg","png"))