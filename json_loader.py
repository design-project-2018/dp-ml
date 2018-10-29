import json
import numpy as np
import cv2



# Given a frame and list of obj coordinates, crop objects in frame

def crop_objects(frame, coord_list):
    obj_list = []
    alpha_list = []
    for i in range(0, len(coord_list)):
        cropped = frame[coord_list[i]['y']:coord_list[i]['y']+coord_list[i]['height'],coord_list[i]['x']:coord_list[i]['x']+coord_list[i]['width']]
        obj_list.append(cropped)
        alpha_list.append(coord_list[i]['probability']/100)
    
    # If less than 10 objects detected
    for j in range(len(coord_list), 10):
        cropped = np.zeros_like((3, 3, 3)) # No need for 224 x 224, resized later when searching for embeddings
        obj_list.append(cropped)
        alpha_list.append(0)

    return np.asarray(obj_list), np.asarray(alpha_list)



# Given videopath, path to corresponding JSON, output cropped images for each frame in clip, and corresponding probabilities

def objects_from_clip(vid_path, json_path, n_frames=100):
    # Read JSON file with
    with open(json_path) as f:
        json_clip = json.load(f)
    cap = cv2.VideoCapture(vid_path)
    
    object_frames = []
    alpha_frames = []
    ctr= 1
    while (ctr<= n_frames):
        _, frame = cap.read()
        obj_coords = json_clip["frames"][ctr-1]["objects"]
        frame_objects, frame_alphas = crop_objects(frame, obj_coords)
        
        object_frames.append(frame_objects)
        alpha_frames.append(frame_alphas)
        ctr +=1

    return np.asarray(object_frames), np.asarray(alpha_frames)


'''
TEST CODE FOR RUNNING BOTH METHODS

def main():
    json_path = './000830.json'
    vid_path = './dataset/videos/testing/negative/000830.mp4'
    objects_830, alphas_830 = objects_from_clip(vid_path, json_path)

if __name__ == "__main__":
    main()

'''