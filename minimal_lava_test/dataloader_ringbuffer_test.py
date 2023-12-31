# run infercence our traine DHP19 model on Loihi?
from dataset import DHP19NetDataset
from lava.proc import io
from plot import plot_input_sample
import numpy as np
import matplotlib.pyplot as plt
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg
from lava.utils.system import Loihi2
# from icecream import ic

# Check if Loihi2 compiker is available and import related modules.
Loihi2.preferred_partition = 'oheogulch'
loihi2_is_available = Loihi2.is_loihi2_available

loihi2_is_available = False # Force CPU execution

if loihi2_is_available:
    print(f'Running on {Loihi2.partition}')
    compression = io.encoder.Compression.DELTA_SPARSE_8
else:
    print("Loihi2 compiler is not available in this system. "
          "This tutorial will execute on CPU backend.")
    compression = io.encoder.Compression.DENSE


class CustomHwRunConfig(Loihi2HwCfg):
    """Custom Loihi2 hardware run config."""
    def __init__(self):
        super().__init__(select_sub_proc_model=True)

    def select(self, proc, proc_models):
        # customize run config
        if isinstance(proc, io.encoder.DeltaEncoder):
            return io.encoder.PyDeltaEncoderModelSparse
        return super().select(proc, proc_models)


class CustomSimRunConfig(Loihi2SimCfg):
    """Custom Loihi2 simulation run config."""
    def __init__(self):
        super().__init__(select_tag='fixed_pt')

    def select(self, proc, proc_models):
        # customize run config
        if isinstance(proc, io.sink.RingBuffer):
            return io.sink.PyReceiveModelFloat
        if isinstance(proc, io.encoder.DeltaEncoder):
            return io.encoder.PyDeltaEncoderModelDense
        return super().select(proc, proc_models)

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

# # show sample from the dataset
input, target = complete_dataset[0]
input.shape # (344, 260, 1) W x H x C x T
target.shape # ((img_widht + img_height)/downsample_factor) * joints (604, )

for joint in range(num_joints):
    coords_1hot = target[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
    coords_one_hot_y = coords_1hot[0:int(img_height / downsample_factor)]
    coords_one_hot_x = coords_1hot[int(img_height / downsample_factor):]
    act_target_y = np.where(coords_one_hot_y == coords_one_hot_y.max())[0][0]
    act_target_x = np.where(coords_one_hot_x == coords_one_hot_x.max())[0][0]
    act_target = np.array([act_target_y, act_target_x]) * downsample_factor
    plot_input_sample(np.swapaxes(input[:,:,0,0],0,1), target_coords=act_target, title='Input frame, Traget joint ' + str(joint_idxs[joint]), path=None)

# show target
plt.figure(figsize=(20, 10))
for joint in range(num_joints):
    coords_1hot = target[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
    coords_one_hot_y = coords_1hot[0:int(img_height / downsample_factor)]
    coords_one_hot_x = coords_1hot[int(img_height / downsample_factor):]
    act_target_y = np.where(coords_one_hot_y == coords_one_hot_y.max())[0][0]
    act_target_x = np.where(coords_one_hot_x == coords_one_hot_x.max())[0][0]

    plt.subplot(num_joints, 2, (2*joint)+1)
    plt.plot(coords_one_hot_y)
    plt.vlines(act_target_y, ymin=0, ymax=1, colors='g', linestyles='dashed', label='target')
    plt.ylabel('y target')
    plt.title('Joint ' + str(joint_idxs[joint]))
    
    plt.subplot(num_joints, 2, (2*joint)+2)
    plt.plot(coords_one_hot_x)
    plt.vlines(act_target_x, ymin=0, ymax=1, colors='g', linestyles='dashed', label='target')
    plt.ylabel('x target')
plt.show()    

# run dataloder + logger for N steps
num_steps = 10

# Create Dataloader
# The dataloader process reads data from the dataset objects and sends out the input frame and ground truth as spikes.
dataloader = io.dataloader.SpikeDataloader(dataset=complete_dataset)

# Create Logger for targets
gt_logger = io.sink.RingBuffer(shape=target.shape, buffer=num_steps)

# connect processes
dataloader.ground_truth.connect(gt_logger.a_in)
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
    gt_logger.run(condition=run_condition, run_cfg=run_config)
    gts = gt_logger.data.get()
    gt_logger.stop()
    print('DONE')
    print(gts)
