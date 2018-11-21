import os
import json
import numpy as np
import pandas as pd
import cv2

from VGGModel import VGGModel

# Script that creates training data for RNN with dynamic spatial attention model, stored as .npz files

embeddings_path_train = './dataset/custom_features/training'
embeddings_path_test = './dataset/custom_features/testing'

''' Given a frame and list of obj coordinates, crop objects in frame, then extracts object features ''' 
def crop_object_features(frame, coord_list, network):
    obj_list = []

    # Append full frame first
    full_frame_feat = network.extract_feature(frame)
    obj_list.append(full_frame_feat)

    if (len(coord_list) < 9):
        for i in range(0, len(coord_list)):
            cropped = frame[coord_list[i]['y']:coord_list[i]['y']+coord_list[i]['height'],coord_list[i]['x']:coord_list[i]['x']+coord_list[i]['width']]
            obj_feat = network.extract_feature(cropped)
            print(obj_feat.shape)
            obj_list.append(obj_feat)
        
        # If less than 10 objects detected
        for j in range(len(coord_list), 9):
            obj_feat = np.zeros_like(obj_list[len(coord_list)])
            obj_list.append(obj_feat)
            print(obj_feat.shape)
        print('Length of obj list: {} '.format(len(obj_list)))
    else:
        for i in range(0, 10):
            cropped = frame[coord_list[i]['y']:coord_list[i]['y']+coord_list[i]['height'],coord_list[i]['x']:coord_list[i]['x']+coord_list[i]['width']]
            obj_feat = network.extract_feature(cropped)
            obj_list.append(obj_feat)
        print('Length of obj list: {} '.format(len(obj_list)))

    return np.asarray(obj_list)



''' Output cropped images for each frame in clip given videopath, path to corresponding JSON  '''
def objects_from_clip(vid_path, json_path, network, n_frames=100):
    # Read JSON file with
    with open(json_path) as f:
        json_clip = json.load(f)
    cap = cv2.VideoCapture(vid_path)
    
    object_frames = []
    ctr= 1
    while (ctr<= n_frames):
        _, frame = cap.read()
        obj_coords = json_clip["frames"][ctr-1]["objects"]
        print("Cropping feature from frame : {}".format(ctr))
        frame_objects = crop_object_features(frame, obj_coords, network)
        print("Cropped : {} !".format(ctr))
        object_frames.append(frame_objects)
        ctr +=1

    return np.asarray(object_frames)


def main():
    # VGG_network = VGGModel()
    # json_path = './dataset/object_extraction/testing/negative/000830.json'
    # vid_path = './dataset/videos/testing/negative/000830.mp4'
    # print("Extracting high level features from {}".format(vid_path))
    # objects_830 = objects_from_clip(vid_path, json_path, VGG_network)
    # print(objects_830.shape)
    # objects_830, alphas_830 = objects_from_clip(vid_path, json_path)
    # print(objects_830.shape)
    # test_list = [objects_830, alphas_830]
    # np.savez_compressed(embeddings_path+ 'test.npz', objects_830, alphas_830) 
    # opened_arrays = np.load(embeddings_path+'test.npz')
    # print (opened_arrays.files)
    # print(opened_arrays['arr_0'].shape)
    print("hello")



if __name__ == "__main__":
    main()
