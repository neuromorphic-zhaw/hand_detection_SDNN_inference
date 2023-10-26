import numpy as np
import matplotlib.pyplot as plt
from lava.magma.core.run_conditions import RunSteps
from lava.proc import io
from lava.lib.dl import netx
from dataset import dhp19
import os
from utils import (
    OasisNetEncoder, OasisNetDecoder, OasisNetMonitor,
    CustomHwRunConfig, CustomSimRunConfig,
    get_input_transform
)

# Import modules for Loihi2 execution
from lava.utils.system import Loihi2
Loihi2.preferred_partition = 'oheogulch'
loihi2_is_available = Loihi2.is_loihi2_available

# loihi2_is_available = False # Force CPU execution

if loihi2_is_available:
    print(f'Running on {Loihi2.partition}')
    # compression = io.encoder.Compression.DELTA_SPARSE_8
    compression = io.encoder.Compression.DENSE
else:
    print("Loihi2 compiler is not available in this system. "
          "This tutorial will execute on CPU backend.")
    compression = io.encoder.Compression.DENSE

# Paths
parent_path = os.path.dirname(os.path.abspath(__file__))
act_result_path = "sdnn_1hot_smoothed_rmsprop2_epochs50_lr0.0001_batchsize8_seq8_cam1_lam0.001"
model_path = os.path.join(parent_path, act_result_path, "model.net")
event_data_path = os.path.join(os.path.dirname(parent_path), "data", "data")

# Create network block
net = netx.hdf5.Network(net_config=model_path, skip_layers=1)
# print(net)

# Set execution parameters
num_samples = 1
num_steps = num_samples + len(net.layers)

# Create Dataset instance
transform = get_input_transform(net.net_config)
full_set = dhp19(event_data_path, camera_index=1)
num_test_samples = full_set.__len__()

# Create Dataloader
dataloader = io.dataloader.SpikeDataloader(dataset=full_set)

# # Create Input Encoder
input_encoder = OasisNetEncoder(shape=net.inp.shape,
                                net_config=net.net_config,
                                compression=compression)

# Create Output Decoder
output_decoder = OasisNetDecoder(shape=net.out.shape)

# Connect the processes
dataloader.s_out.connect(input_encoder.inp)
dataloader.s_out.connect(net.inp)
input_encoder.out.connect(net.inp)
net.out.connect(output_decoder.inp)

# Run the network
if __name__ == '__main__':
    if loihi2_is_available:
        run_config = CustomHwRunConfig()
    else:
        run_config = CustomSimRunConfig()
    net.run(condition=RunSteps(num_steps=num_steps), run_cfg=run_config)
    net.stop()