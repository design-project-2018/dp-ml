import os
import json
import numpy as np
import pandas as pd
import cv2

from VGGModel import VGGModel

# Script that creates training data for RNN with dynamic spatial attention model, stored as .npz files



''' Given a frame and list of obj coordinates, crop objects in frame, then extracts object features ''' 
def crop_object_features(frame, coord_list, network):
    obj_list = []

    # Append full frame first
    frame = cv2.resize(frame, (224, 224))
    full_frame_feat = network.extract_feature(frame)
    obj_list.append(full_frame_feat)

    if (len(coord_list) <= 9):
        for i in range(0, len(coord_list)):
            cropped = frame[coord_list[i]['y']:coord_list[i]['y']+coord_list[i]['height'],coord_list[i]['x']:coord_list[i]['x']+coord_list[i]['width']]
            obj_feat = network.extract_feature(cropped)
            obj_list.append(obj_feat)
        
        # If less than 10 objects detected
        for j in range(len(coord_list), 9):
            obj_feat = np.zeros_like(obj_list[len(coord_list)])
            obj_list.append(obj_feat)
    else:
        for i in range(0, 9):
            cropped = frame[coord_list[i]['y']:coord_list[i]['y']+coord_list[i]['height'],coord_list[i]['x']:coord_list[i]['x']+coord_list[i]['width']]
            obj_feat = network.extract_feature(cropped)
            obj_list.append(obj_feat)
            # print("Object list length: {}".format(len(obj_list)))

    return np.asarray(obj_list)



''' Output cropped images for each frame in clip given videopath, path to corresponding JSON  '''
def objects_from_clip(vid_path, json_path, network, n_frames=100):
    # Read JSON file with
    with open(json_path) as f:
        json_clip = json.load(f)
    cap = cv2.VideoCapture(vid_path)
    
    label = json_clip["label"]
    object_frames = []
    ctr= 1
    while (ctr<= n_frames):
        # print("Frame: {}".format(ctr))
        _, frame = cap.read()
        obj_coords = json_clip["frames"][ctr-1]["objects"]
        frame_objects = crop_object_features(frame, obj_coords, network)
        object_frames.append(frame_objects)
        ctr +=1

    return np.asarray(object_frames), label



''' Given path of JSON files with extracted features and labels, loops through files and writes as .npz '''
def write_batch_directory(inputPath, writePath, extraction_network):

    dTypes = ['training/', 'testing/'] # dataset folder types
    vTypes = [ 'negative/', 'positive/'] # accident/no accident folders
    idx_batch = 1

    for dType in dTypes:
        for vType in vTypes:
            jsonFolder = inputPath + dType + vType
            vidFolder = './dataset/videos/{}{}'.format(dType, vType)
            fileNames = sorted(os.listdir(jsonFolder))
            
            numFiles = len(fileNames)
            idx_file = 0
            
            # Need 10 files per batch
            while (numFiles >= 10):
                batch_features = []
                batch_labels = []
                batchCount = 1
                # Iterate through files until batchsize of 10 is reached
                while (batchCount <= 10): 
                    fileName = fileNames[idx_file]
                    jsonPath = jsonFolder + fileName
                    videoName = fileName.split('.json')[0] + '.mp4' # Corresponding video to JSON
                    vidPath = vidFolder + videoName
                    
                    print("Batch Number {} - ".format(idx_batch))
                    print("Sample Number: {}".format(batchCount))
                    print("Extracting high level features from video: {}...".format(vidPath))
                    
                    feature_array, label = objects_from_clip(vidPath, jsonPath, extraction_network)
                    
                    batch_features.append(feature_array)
                    batch_labels.append(label)
                    batchCount += 1 # Update sample size of batch
                    numFiles -= 1 # Update number of videos left to parse
                    print("Files left to parse: {} ...".format(numFiles))
                    idx_file += 1 # Update index for parsing through files in folder
                    
                # Write to batch file
                batch_features = np.asarray(batch_features)
                batch_labels = np.asarray(batch_labels)
                print("Writing to batch file: {}".format(str(idx_batch).zfill(3)))
                np.savez_compressed(writePath + 'batch_'+ str(idx_batch).zfill(3) + '.npz', data=batch_features, labels=batch_labels)
                idx_batch += 1
                
    return
                    

def restart_idx(data_path):
    print("Re-indexing files in {}...".format(data_path))
    file_names = sorted(os.listdir(data_path))

    num_files = len(file_names)
    idx_batch = 1
    for file in file_names:
        os.rename(data_path + file, data_path + 'batch_'+ str(idx_batch).zfill(3) + '.npz')
        idx_batch += 1

    print("Renamed {} files in directory {}".format(num_files, data_path))
    return 



def move_files(save_path, train_num=126, test_num=46):

    if os.path.isdir(save_path + 'training/') == False:
        os.mkdir(save_path + 'training/')
    if os.path.isdir(save_path + 'testing/') == False:
        os.mkdir(save_path + 'testing/')
        
    print("Moving files to training and testing folders")
    for i in range(1, train_num+1):
        os.rename(save_path + 'batch_'+ str(i).zfill(3) + '.npz', save_path + 'training/'+'batch_'+ str(i).zfill(3) + '.npz')
    for i in range(train_num+1, train_num+1+test_num+1):
        os.rename(save_path + 'batch_'+ str(i).zfill(3) + '.npz', save_path + 'testing/'+'batch_'+ str(i).zfill(3) + '.npz')
    return


def main():
    VGG_network = VGGModel()
    json_path = './dataset/object_extraction/'
    embeddings_path = './dataset/custom_features/'
    rewrite_path = './dataset/custom_features/testing/'
    
    write_batch_directory(json_path, embeddings_path, VGG_network)
    move_files(embeddings_path, train_num=126, test_num=46)
    restart_idx(rewrite_path)


if __name__ == "__main__":
    main()
