from torch.utils.data import Dataset
import numpy as np
import torch
import os

class DHP19NetDataset(Dataset):
    """
    Dataset class for the DHP19 dataset
    Returns image and ground truth value
    when the object is indexed.

    Parameters
    ----------
    path : str
        Path of the dataset folder.
    joint_idx : list
        Joint indices the model was trained on
    cam_id : int
        Camera index the model was trained on and which should be used for inference
    num_time_steps : int
        Number of event frames per sequence per file (usually the number of frame the model was trained on)
    Usage
    -----

    >>> dataset = DHP19NetDataset(path='./../data/dhp19/', joint_idx=[7,8], cam_id=1, num_time_steps=8)
    >>> image, target_coords_abs = dataeset[0]
    >>> num_samples = len(dataset)
    """
    def __init__(self, path, joint_idxs=[7,8], cam_id=1, num_time_steps=8):
        self.path = path
        self.joint_idxs = joint_idxs
        self.cam_id = cam_id
        self.num_time_steps = num_time_steps
        self.files = os.listdir(path)
        self.files = [f for f in self.files if f.endswith('pt')]
        self.input_tensors = []
        self.target_cords = []
        self.session = []
        self.subject = []
        self.mov = []

        # loop over all files in the folder
        for i, file in enumerate(self.files):
            session = int(file.split('_')[1][7:])
            subject = int(file.split('_')[0][1:])
            mov = int(file.split('_')[2][3:])    
            data_dict = torch.load(self.path + file) # load file

            # loop over all time steps in the file and select input frame and target joints for cam N
            for t in range(self.num_time_steps):
                self.input_tensors.append(data_dict['input_tensor'][0].to_dense()[:,:,self.cam_id,t].numpy().astype(np.float32)) # take frame t of cam N
                self.target_cords.append(data_dict['target_coords_abs'][self.joint_idxs,:,self.cam_id,t].numpy().astype(np.int32)) # take target joints for frame t of cam N
                self.session.append(session)
                self.subject.append(subject)
                self.mov.append(mov)

    def __len__(self):
        return len(self.input_tensors)

    def __getitem__(self, idx):
        return self.input_tensors[idx], self.target_cords[idx]
