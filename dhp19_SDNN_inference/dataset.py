# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import os
from typing import Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob

class dhp19(Dataset):
    """
    Dataset class for the dhp19 dataset
    """
    def __init__(self, path, camera_index=1):
        self.path = path
        self.files = os.listdir(path)
        self.files = [f for f in self.files if f.endswith('pt')]
        self.camera_index = camera_index

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        session = int(file.split('_')[1][7:])
        subject = int(file.split('_')[0][1:])
        mov = int(file.split('_')[2][3:])
        data_dict = torch.load(os.path.join(self.path, file))
        event_frame = data_dict['input_tensor'].to_dense()

        return event_frame[..., self.camera_index, :].permute(2, 1, 0, 3), data_dict['target_coords_abs'] #, data_dict['target_coords_rel'], session, subject, mov