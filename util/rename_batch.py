### Script to fix small naming error in the test batch for visualization

import os

data_path = './../dataset/custom_features/testing/'
file_names = os.listdir(data_path)

num_files = len(file_names)
idx_batch = 1
for file in file_names:
    os.rename(data_path + file, data_path + 'batch_'+ str(idx_batch).zfill(3) + '.npz')
    idx_batch += 1 

print("Renamed {} files in directory {}".format(num_files, data_path))
