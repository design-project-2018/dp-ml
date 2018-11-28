from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Flatten

import cv2
import numpy as np

class VGGModel:

    def __init__(self, verbose=True):
        # Initialize VGG16 model using pretrained ImageNet weights without last 3 fc layers
        base_model = VGG16(weights='imagenet', include_top=True, input_shape = (224, 224, 3))

        print('Initializing VGG16 model with ImageNet weights...')

        # # Initialize two fc layers
        # top_model = Sequential()
        # top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
        # top_model.add(Dense(4096, activation='relu'))
        # top_model.add(Dense(4096, activation='relu'))

        # # Copy VGG layers in order to concatenate
        # model = Sequential()
        # for layer in base_model.layers:
        #     model.add(layer)        base_model.layers.pop()
        base_model.layers.pop() # Get rid of the classification layer
        base_model.layers.pop() # Get rid of the dropout layer
        base_model.outputs = [base_model.layers[-1].output]
        base_model.layers[-1].outbound_nodes = []

        # model.add(top_model)

        # Make the layers untrainable
        # for layer in model.layers:
        #     layer.trainable = False
        if (verbose):
            base_model.summary()
        
        self.model = base_model

    ''' Extracts 4096 x 1 high level feature vector from given image '''
    def extract_feature(self, img):
        img = cv2.resize(img, (224, 224))
        img_np = (np.array(img)).astype('float64')
        img_np = np.expand_dims(img_np, axis=0)
        img_data = preprocess_input(img_np)
        
        return np.transpose(self.model.predict(img_data))

    # Create 100 x 4096 feature vector from videoclip of n frames
    def extract_from_clip(self, path, n_frames=100):
        ctr= 1
        feature_list = []
        cap = cv2.VideoCapture(path)
        while (ctr<= n_frames):
            _, frame = cap.read()
            frame_feature = self.extract_feature(frame)
            feature_list.append(frame_feature)
            ctr +=1
        
        return np.asarray(feature_list)

'''
SOME TEST CODE FOR FEATURE EXTRACTION - Remove later

def main():
    sammy = VGGModel()
    img_path = './eddY_3.jpg'
    vid_path = './dataset/videos/training/positive/000001.mp4'
    img = cv2.imread(img_path)
    test = sammy.extract_feature(img)
    test_vid = sammy.extract_from_clip(vid_path)
    print(test.shape)
    print(test_vid.shape)

if __name__ == "__main__":
    main()

'''
        
def main():
    sammy = VGGModel()
    img_path = './eddy_3.jpg'
    # vid_path = './dataset/videos/training/positive/000001.mp4'
    img = cv2.imread(img_path)
    test = sammy.extract_feature(img)
    # test_vid = sammy.extract_from_clip(vid_path)
    print(test.shape)
    # print(test_vid.shape)

if __name__ == "__main__":
    main()