import json 

test_path = './000830.json'

with open(test_path) as f:
    data = json.load(f)

input_list_frame_1 = data["frames"][0]["objects"][0]
print(input_list_frame_1)
def crop_frame(frame, coord_list):
    obj_list

    return obj_list