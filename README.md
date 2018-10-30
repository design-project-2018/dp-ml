# dp-ml

Repository for storing machine learning microservices. From here, we can train the end-to-end model on the fullframe features, as well as object features. The initial code is based off of the following paper:
[Anticipating Accidents in Dashcam Videos (ACCV 2016)](https://github.com/smallcorgi/Anticipating-Accidents)
---- 
# Documentation

## VGG Model Class Methods

*The VGG Model class consists of the Keras implementation of VGG16 with pretrained ImageNet weights, with the addition of 2 dense layers that output a 1 x 4096 feature vector.*

----
    _init__(self, verbose=True):

*Initializes VGG Model*

**Parameters**

>verbose:boolean, default=True
>>Option to print model summary

**Returns:**
>None

----
    extract_from_clip(self, path, n_frames=100):

*Class method that takes a videoclip and extracts VGG feature from each frame.*

**Parameters**

>path:string, IP_address: 
>>Videopath that is compatible with cv2.VideoCapture, provides path to videoclip

>n_frames:int, default=100
>>Number of frames in the clip

**Returns:**
>numpy_array: 100x4096 feature vector for default clip size
----
    extract_feature(self, img):
*Class method that takes in image and outputs VGG feature*

**Parameters**

>img:numpy array, image: 
>>Image to be resized to (224,224,3)

**Returns:**
>numpy_array: 1x4096 feature vector 
----

## JSON Loader Methods

*The JSON Loader file contains methods that crop objects in videoframes, given the corresponding JSON files with locations to crop from. Follows the specific JSON schema that returns up to 10 objects per frame from previous preprocessing.*

----
    objects_from_clip(vid_path, json_path, n_frames=100):

*Method that takes in videoclip, and returns array of cropped objects from each frame, as well as an array of confidence scores for each object in every frame*

**Parameters**

>vid_path:string, IP_address: 
>>Videopath that is compatible with cv2.VideoCapture, provides path to videoclip

>json_path:stringpath
>>Path to JSON file with object coordinates corresponding to video in vid_path parameter

>n_frames:int, default=100
>>Number of frames in the clip

**Returns:** 
>numpy_array (2): Array of arrays of images(n x m x 3), array of arrays of probabilities(n_frames by 10)
----
    crop_objects(frame, coord_list):

*Method that takes in frame, and returns array of cropped objects, as well as an array of confidence scores for each object*

**Parameters**

>frame:numpy array, image: 
>>Input image

>coord_list: list
>>List of dictionaries with specific schema of object id, probability, coordinates

>n_frames:int, default=100
>>Number of frames in the clip

**Returns:** 
>numpy_array (2): Array of of images, array of probabilities
----