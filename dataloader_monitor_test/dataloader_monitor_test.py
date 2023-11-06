# run infercence our traine DHP19 model on Loihi?
from dataset import DHP19NetDataset
from lava.proc import io
from plot import plot_input_sample
import numpy as np
import matplotlib.pyplot as plt
from lava.magma.core.run_conditions import RunSteps
from lava.utils.system import Loihi2

# from icecream import ic

from utils import (
    CustomHwRunConfig, CustomSimRunConfig, DHP19NetMonitor
)


# Check if Loihi2 compiker is available and import related modules.
Loihi2.preferred_partition = 'oheogulch'
loihi2_is_available = Loihi2.is_loihi2_available

# loihi2_is_available = False # Force CPU execution

if loihi2_is_available:
    print(f'Running on {Loihi2.partition}')
    compression = io.encoder.Compression.DELTA_SPARSE_8
else:
    print("Loihi2 compiler is not available in this system. "
          "This tutorial will execute on CPU backend.")
    compression = io.encoder.Compression.DENSE


# Set paths to model and data
project_path = './'
event_data_path = project_path + '../data/dhp19/'

# paramters of the traininf data
img_width = 344
img_height = 260
downsample_factor = 2
xy_coord_vec_length = int((img_width + img_height)/downsample_factor)

# Create Dataset instance
seq_length = 8 # number of event frames per data file
cam_idxs = [1,2] # camera index to train on 1,2 are frontal views
joint_idxs = [7, 8] # joint indices to train on
num_joints = len(joint_idxs)
complete_dataset = DHP19NetDataset(path=event_data_path, joint_idxs=joint_idxs, cam_id=cam_idxs[0], num_time_steps=seq_length)
print('Dataset loaded: ' + str(len(complete_dataset)) + ' samples found')

# # # show sample from the dataset
# input, target = complete_dataset[0]
# input.shape # (344, 260, 1) W x H x C x T
# target.shape # ((img_widht + img_height)/downsample_factor) * joints (604, )

# plot input sample
plot_input_sample(np.swapaxes(input[:,:,0,0],0,1), target_coords=None, title='Input frame', path=None)

# run dataloder + logger for N steps
num_steps = 100

# Create Dataloader
# The dataloader process reads data from the dataset objects and sends out the input frame and ground truth as spikes.
dataloader = io.dataloader.SpikeDataloader(dataset=complete_dataset)

# Create Logger for targets
gt_logger = io.sink.RingBuffer(shape=(1,), buffer=num_steps)

# Create Monitor
dhp19_monitor = DHP19NetMonitor(in_shape=input.shape[:-1],
                                output_offset=0,
                                num_joints=2)

# connect processes
dataloader.ground_truth.connect(gt_logger.a_in)
dataloader.s_out.connect(dhp19_monitor.frame_in)
run_condition = RunSteps(num_steps=num_steps)

if loihi2_is_available:
    run_config = CustomHwRunConfig()
else:
    run_config = CustomSimRunConfig()

# if loihi2_is_available:
#     run_config = Loihi2HwCfg()
# else:
#     run_config = Loihi2SimCfg()


if __name__ == '__main__':       
    dhp19_monitor.run(condition=run_condition, run_cfg=run_config)
    gts = gt_logger.data.get()
    dhp19_monitor.stop()
    print('DONE')
    print(gts)
