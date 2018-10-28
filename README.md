# dp-ml

Repository for storing machine learning microservices. From here, we can train the end-to-end model on the fullframe features, as well as object features. The initial code is based off of the following paper:
[Anticipating Accidents in Dashcam Videos (ACCV 2016)](https://github.com/smallcorgi/Anticipating-Accidents)

## Feature Extractor Tool

The feature extractor tool is used to run feature extraction tasks from the [dp-jetson-alg](https://github.com/design-project-2018/dp-jetseon-alg) across the full
dataset. Currently there are two tasks in the repository: one for extracting the location of objects and one for extracting the optical flow. After cloning and 
building the dp-jetson-alg repository, the follow commands can be used to run the feature extraction tool.

```
python feature_extractor.py --extractor <PATH TO DP-JETSON-ALG>/object_extractor --data <PATH TO DATA>/videos/ --output <PATH TO OUTPUT DIRECTORY>
python feature_extractor.py --extractor <PATH TO DP-JETSON-ALG>/optical_flow_extractor --data <PATH TO DATA>/videos/ --output <PATH TO OUTPUT DIRECTORY>

```
