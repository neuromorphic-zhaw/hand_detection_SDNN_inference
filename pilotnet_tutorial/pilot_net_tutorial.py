import numpy as np
import matplotlib.pyplot as plt

from lava.magma.core.run_conditions import RunSteps
from lava.proc import io

from lava.lib.dl import netx
from dataset import PilotNetDataset
from utils import (
    PilotNetEncoder, PilotNetDecoder, PilotNetMonitor,
    CustomHwRunConfig, CustomSimRunConfig,
    get_input_transform
)

# Check if Loihi2 compiler is available and import related modules.
from lava.utils.system import Loihi2
Loihi2.preferred_partition = 'oheogulch'
loihi2_is_available = Loihi2.is_loihi2_available

if loihi2_is_available:
    print(f'Running on {Loihi2.partition}')
    compression = io.encoder.Compression.DELTA_SPARSE_8
else:
print("Loihi2 compiler is not available in this system. "
        "This tutorial will execute on CPU backend.")
compression = io.encoder.Compression.DENSE

# create network block
net = netx.hdf5.Network(net_config='network.net', skip_layers=1)
print(net)

print(f'There are {len(net)} layers in the network:')
for l in net.layers:
    print(f'{l.__class__.__name__:5s} : {l.name:10s}, shape : {l.shape}')

# Set execution parameters
num_samples = 200
steps_per_sample = 1
num_steps = num_samples + len(net.layers)
# Output appears delayed due to encoding and spike communication
out_offset = len(net.layers) + 3

# Create Dataset instance
transform = get_input_transform(net.net_config)
full_set = PilotNetDataset(
    path='../data',
    size=net.inp.shape[:2],
    transform=transform,  # input transform
    visualize=True,  # visualize ensures the images are returned in sequence
    sample_offset=10550,
)
train_set = PilotNetDataset(
    path='../data',
    size=net.inp.shape[:2],
    transform=transform,  # input transform
    train=True,
)
test_set = PilotNetDataset(
    path='../data',
    size=net.inp.shape[:2],
    transform=transform,  # input transform
    train=False,
)

# Create Dataloader
# The dataloader process reads data from the dataset objects and sends out the input frame and ground truth as spikes.
dataloader = io.dataloader.SpikeDataloader(dataset=full_set)


# Create Input Encoder
# The input encoder process does frame difference of subsequent frames to sparsify the input to the network.
# For Loihi execution, it additionally compresses and sends the input data to the Loihi 2 chip.
input_encoder = PilotNetEncoder(shape=net.inp.shape,
                                net_config=net.net_config,
                                compression=compression)

# Create Output Decoder
# The output decoder process receives the output from the network and applies proper scaling to decode the steering angle prediction.
# For Loihi execution, it additionally communicates the network's output spikes from the Loihi 2 chip.
output_decoder = PilotNetDecoder(shape=net.out.shape)

# Create Monitor and Dataloggers
# Monitor is a lava process that visualizes the PilotNet network prediction in real-time. In addition, datalogger processes store the network predictions and ground truths.
monitor = PilotNetMonitor(shape=net.inp.shape,
                          transform=transform,
                          output_offset=out_offset)
gt_logger = io.sink.RingBuffer(shape=(1,), buffer=num_steps)
output_logger = io.sink.RingBuffer(shape=net.out_layer.shape, buffer=num_steps)

# Connect the processes
dataloader.ground_truth.connect(gt_logger.a_in)
dataloader.s_out.connect(input_encoder.inp)

input_encoder.out.connect(net.inp)
net.out.connect(output_decoder.inp)
output_decoder.out.connect(output_logger.a_in)

dataloader.s_out.connect(monitor.frame_in)
dataloader.ground_truth.connect(monitor.gt_in)
output_decoder.out.connect(monitor.output_in)

# Run the network
loihi2_is_available = False # force simulation
if loihi2_is_available:
    run_config = CustomHwRunConfig()
else:
    run_config = CustomSimRunConfig()

net.run(condition=RunSteps(num_steps=num_steps), run_cfg=run_config)
output = output_logger.data.get().flatten()
gts = gt_logger.data.get().flatten()
net.stop()

plt.figure(figsize=(7, 5))
plt.plot(np.array(gts), label='Ground Truth')
plt.plot(np.array(output[out_offset:]).flatten(), label='Lava output')
plt.xlabel(f'Sample frames (+10550)')
plt.ylabel('Steering angle (radians)')
plt.legend()
