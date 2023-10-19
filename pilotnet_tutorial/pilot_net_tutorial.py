import numpy as np
import matplotlib.pyplot as plt

from lava.magma.core.run_configs import Loihi2SimCfg, Loihi2HwCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc import io
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import RefPort

from lava.lib.dl import netx
from dataset import PilotNetDataset
from utils import (
    PilotNetEncoder, PilotNetDecoder, VoltageReader, PilotNetMonitor,
    loihi2hw_exception_map, loihi2sim_exception_map
)

# Check if Loihi2 compiler is available and import related modules.
from lava.utils.system import Loihi2
Loihi2.preferred_partition = 'oheogulch'
# loihi2_is_available = Loihi2.is_loihi2_available
loihi2_is_available = False # force to run on CPU backend

if loihi2_is_available:
    print(f'Running on {Loihi2.partition}')
    compression = io.encoder.Compression.DELTA_SPARSE_8
    from lava.proc import embedded_io as eio
    from lava.proc.embedded_io.state import Read as VoltageReader
else:
    print("Loihi2 compiler is not available in this system. "
          "This tutorial will execute on CPU backend.")
    compression = io.encoder.Compression.DENSE
# create network block
# The input spike loads to dataloader at t=0
# Gets transmitted through the embedded processor at t=1
# and appears at the input layer at t=2
net = netx.hdf5.Network(net_config='network.net',
                        reset_interval=16,
                        reset_offset=3)
print(net)

# print(net)
# print(f'There are {len(net)} layers in the network:')
# for l in net.layers:
#     print(f'{l.__class__.__name__:5s} : {l.name:10s}, shape : {l.shape}')

# Set execution parameters
num_samples = 201
steps_per_sample = net.reset_interval
readout_offset = len(net) + 2
num_steps = num_samples * steps_per_sample + 1

# Output appears delayed due to encoding and spike communication
out_offset = len(net.layers) + 3

# Create Dataset instance
full_set = PilotNetDataset(
    path='../data',
    transform=net.in_layer.transform,  # input transform
    visualize=True,  # visualize ensures the images are returned in sequence
    sample_offset=10550,
)
train_set = PilotNetDataset(
    path='../data',
    transform=net.in_layer.transform,  # input transform
    train=True,
)
test_set = PilotNetDataset(
    path='../data',
    transform=net.in_layer.transform,  # input transform
    train=False,
)
# Create Dataloader
# The dataloader process reads data from the dataset objects and sends out the input frame and ground truth as spikes. The dataloader injects new input sample every steps_per_sample.
dataloader = io.dataloader.SpikeDataloader(dataset=full_set, interval=steps_per_sample)

# Configure the input layer for graded spike input
net.in_layer.neuron.du.init = -1  # Make current state persistent

# Create Input Encoder
# The input encoder process does frame difference of subsequent frames to sparsify the input to the network.
# For Loihi execution, it additionally compresses and sends the input data to the Loihi 2 chip.
input_encoder = PilotNetEncoder(shape=net.in_layer.shape,
                                interval=steps_per_sample,
                                offset=1,
                                compression=compression)
# Create Output Decoder
# The output of PilotNet LIF network is the output layer neuron's voltage. We use a VoltageReader to read the neuron voltage and scale the input appropriately using AffineTransformer.
# For Loihi execution, VoltageReader additionally communicates the read values from the Loihi 2 chip.
output_adapter = VoltageReader(shape=net.out.shape,
                               interval=steps_per_sample,
                               offset=len(net) + 1)
output_decoder = PilotNetDecoder(shape=net.out.shape,
                                 weight=1 / steps_per_sample / 32 / 64,
                                 interval=steps_per_sample,
                                 offset=len(net) + 2)

# Create Monitor and Dataloggers
# Monitor is a lava process that visualizes the PilotNet network prediction in real-time. In addition, datalogger processes store the network predictions and ground truths.
monitor = PilotNetMonitor(shape=net.inp.shape,
                          transform=net.in_layer.transform,
                          interval=steps_per_sample)
gt_logger = io.sink.RingBuffer(shape=(1,), buffer=num_steps)
output_logger = io.sink.RingBuffer(shape=net.out.shape, buffer=num_steps)

# Connect the processes
dataloader.ground_truth.connect(gt_logger.a_in)
dataloader.s_out.connect(input_encoder.inp)
input_encoder.out.connect(net.in_layer.neuron.a_in)

output_adapter.connect_var(net.out_layer.neuron.v)
output_adapter.out.connect(output_decoder.inp)
output_decoder.out.connect(output_logger.a_in)

dataloader.s_out.connect(monitor.frame_in)
dataloader.ground_truth.connect(monitor.gt_in)
output_decoder.out.connect(monitor.output_in)

# Run the network
if loihi2_is_available:
    run_config = Loihi2HwCfg(exception_proc_model_map=loihi2hw_exception_map)
else:
    run_config = Loihi2SimCfg(select_tag='fixed_pt',
                              exception_proc_model_map=loihi2sim_exception_map)
net.run(condition=RunSteps(num_steps=num_steps), run_cfg=run_config)
output = output_logger.data.get().flatten()
gts = gt_logger.data.get().flatten()[::steps_per_sample]
net.stop()
result = output[readout_offset::steps_per_sample]

plt.figure(figsize=(7, 5))
plt.plot(np.array(gts), label='Ground Truth')
plt.plot(result[1:].flatten(), label='Lava output')
plt.xlabel(f'Sample frames (+10550)')
plt.ylabel('Steering angle (radians)')
plt.legend()