import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from lava.utils.system import Loihi2
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg, Loihi2HwCfg
from lava.magma.compiler.subcompilers.nc.ncproc_compiler import CompilerOptions
from lava.proc import io
from lava.lib.dl import netx
import h5py


# path to model
parent_path = os.path.dirname(os.path.abspath(__file__))
act_result_path = "sdnn_1hot_smoothed_rmsprop2_epochs50_lr0.0001_batchsize8_seq8_cam1_lam0.001"
model_path = os.path.join(parent_path, act_result_path, "model.net")

if __name__ == "__main__":
    CompilerOptions.verbose = True
    CompilerOptions.log_memory_info = True
    CompilerOptions.show_resource_count = True

    net = netx.hdf5.Network(
        net_config=model_path, skip_layers=1,
                    )
    print('net loaded')
    print(net)
    run_config = Loihi2HwCfg()
    print('compiling network')

    compile_only = True

    if compile_only:
        # Compile but do not run the network
        exec = net.compile(run_config, None)
        print('WE ARE DONE')
    else:
        # This will try to execute the dummy network
        net._log_config.level = logging.INFO
        net.run(condition=RunSteps(num_steps=10), run_cfg=run_config)
        net.stop()
