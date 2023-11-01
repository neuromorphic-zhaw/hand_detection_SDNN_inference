# run infercence our traine DHP19 model on Loihi?
from dataset import DHP19NetDataset
from lava.proc import io
from plot import plot_input_sample
import numpy as np
import matplotlib.pyplot as plt
from lava.magma.core.run_conditions import RunSteps

from utils import (
    DHP19NetEncoder, DHP19NetDecoder, DHP19NetMonitor, CustomHwRunConfig, CustomSimRunConfig
)
#     get_input_transform


# Check if Loihi2 compiker is available and import related modules.
from lava.utils.system import Loihi2
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

# Set paths to model and data
project_path = './'
model_path = project_path + 'model/train/'
event_data_path = project_path + '../data/dhp19/'

# paramters of the traininf data
img_width = 344
img_height = 260
downsample_factor = 2
xy_coord_vec_length = int((img_width + img_height)/downsample_factor)


# Create a network block from the trained model.
batch_size = 8  # batch size
learning_rate = 0.00005 # leaerning rate
lam = 0.001 # lagrangian for event rate loss
num_epochs = 30  # training epochs
# steps  = [60, 120, 160] # learning rate reduction milestones
cam_idxs = [1,2] # camera index to train on 1,2 are frontal views
seq_length = 8 # number of event frames per sequence to be shown to the SDNN
joint_idxs = [7, 8] # joint indices to train on
num_joints = len(joint_idxs)

# print(f'There are {len(net)} layers in the network:')
# for l in net.layers:
#     print(f'{l.__class__.__name__:5s} : {l.name:10s}, shape : {l.shape}')

# Create Dataset instance
complete_dataset = DHP19NetDataset(path=event_data_path, joint_idxs=joint_idxs, cam_id=cam_idxs[0], num_time_steps=seq_length)
print('Dataset loaded: ' + str(len(complete_dataset)) + ' samples found')
input, target = complete_dataset[0]
input.shape # (344, 260, 1) W x H x C x T
target.shape # ((img_widht + img_height)/downsample_factor) * joints (604)

# show sample from the dataset
for joint in range(num_joints):
    coords_1hot = target[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
    coords_one_hot_y = coords_1hot[0:int(img_height / downsample_factor)]
    coords_one_hot_x = coords_1hot[int(img_height / downsample_factor):]
    act_target_y = np.where(coords_one_hot_y == coords_one_hot_y.max())[0][0]
    act_target_x = np.where(coords_one_hot_x == coords_one_hot_x.max())[0][0]
    act_target = np.array([act_target_y, act_target_x]) * downsample_factor
    plot_input_sample(np.swapaxes(input[:,:,0,0],0,1), target_coords=act_target, title='Input frame, Traget joint ' + str(joint_idxs[i]), path=None)

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

model_name = 'sdnn_1hot_smoothed_scaled_lowres_rmsprop'
# create experiment name
experiment_name = model_name + \
                '_epochs' + str(num_epochs) + \
                '_lr' + str(learning_rate) + \
                '_batchsize' + str(batch_size) + \
                '_seq' + str(seq_length) + \
                '_cam' + str(cam_idxs).replace(' ', '') + \
                '_lam' + str(lam)

act_model_path = model_path + experiment_name + '/'
# net = netx.hdf5.Network(net_config=act_model_path + 'model.net', skip_layers=1)
# print('Loading net ' + experiment_name)
# print(net)
# print('Done')

# # Set execution parameters... taken from PilotNet example ... needs to be adaped to our model
# num_samples = 200
# steps_per_sample = 1
# num_steps = num_samples + len(net.layers)
# # Output appears delayed due to encoding and spike communication
# out_offset = len(net.layers) + 3


# Create Dataloader
# The dataloader process reads data from the dataset objects and sends out the input frame and ground truth as spikes.
# data, ground_truth = dataset[0]
# data.shape
# gt_shape = (1,) if np.isscalar(ground_truth) else ground_truth.shape
# interval = 1
# data_shape = data.shape[:-1] + (interval,)

dataloader = io.dataloader.SpikeDataloader(dataset=complete_dataset)
gt_logger = io.sink.RingBuffer(shape=target.shape, buffer=10)

# Create Input Encoder
# The input encoder process does frame difference of subsequent frames to sparsify the input to the network.
# For Loihi execution, it additionally compresses and sends the input data to the Loihi 2 chip.
# input_encoder = DHP19NetEncoder(shape=net.inp.shape,
#                                 net_config=net.net_config,
#                                 compression=compression)

# # Create Output Decoder
# # The output decoder process receives the output from the network and applies proper scaling to decode the hand position.
# # For Loihi execution, it additionally communicates the network's output spikes from the Loihi 2 chip.
# output_decoder = DHP19NetDecoder(shape=net.out.shape)

# monitor = DHP19NetMonitor(in_shape=net.inp.shape,  # (344, 260, 1)
#                           out_shape=net.out.shape, # (604,)
#                           target_shape=target.shape, # (2, 2)
#                           output_offset=out_offset,
#                           num_joints=2)

gt_logger = io.sink.RingBuffer(shape=target.shape, buffer=10)
# output_logger = io.sink.RingBuffer(shape=net.out_layer.shape, buffer=num_steps)

# connect processes
dataloader.ground_truth.connect(gt_logger.a_in)

from lava.magma.core.run_conditions import RunSteps
run_condition = RunSteps(num_steps=10)

if loihi2_is_available:
    run_config = CustomHwRunConfig()
else:
    run_config = CustomSimRunConfig()

gt_logger.run(condition=run_condition, run_cfg=run_config)

# # gt_logger.a_in.shape
# # dataloader.ground_truth.shape

# dataloader.s_out.connect(input_encoder.inp)
# # input_encoder.inp.shape
# # dataloader.s_out.shape # (344, 260, 1)

# input_encoder.out.connect(net.inp)
# # net.inp.shape
# # input_encoder.out.shape # (344, 260, 1)
# net.out.connect(output_decoder.inp)
# # output_decoder.inp.shape
# # net.out.shape # (604,)

# output_decoder.out.connect(output_logger.a_in)
# # output_logger.a_in.shape
# # output_decoder.out.shape # (604,)

# dataloader.s_out.connect(monitor.frame_in)
# # monitor.frame_in.shape # (344, 260, 1)
# # dataloader.s_out.shape # (344, 260, 1)

# dataloader.ground_truth.connect(monitor.target_in)
# # dataloader.ground_truth.shape
# # monitor.target_in.shape

# output_decoder.out.connect(monitor.output_in)
# # monitor.output_in.shape
# # output_decoder.out.shape

# if loihi2_is_available:
#     run_config = CustomHwRunConfig()
# else:
#     run_config = CustomSimRunConfig()


# if __name__ == '__main__':
# # # Run the network
#     from lava.magma.compiler.subcompilers.nc.ncproc_compiler import CompilerOptions
#     CompilerOptions.verbose = True
#     CompilerOptions.log_memory_info = True
#     CompilerOptions.show_resource_count = True

#     # Compile but do not run the network
#     exec = net.compile(run_config, None)
#     print('WE ARE DONE')
#     net.create_runtime(run_cfg=run_config, executable=exec)

# net.run(condition=RunSteps(num_steps=2))

# net._log_config.level = logging.INFO
# net.run(condition=RunSteps(num_steps=1), run_cfg=run_config)
# net.stop()
# print('hello')
# # net.run(condition=RunSteps(num_steps=1), run_cfg=run_config)
# # output = output_logger.data.get().flatten()
# # net.stop()

# if __name__ == '__main__':
#     net.run(condition=RunSteps(num_steps=1), run_cfg=run_config)
#     output = output_logger.data.get().flatten()
#     gts = gt_logger.data.get().flatten()
#     net.stop()