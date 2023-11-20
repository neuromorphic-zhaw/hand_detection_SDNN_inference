# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from typing import Iterable, Tuple, Union
import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import OutPort, RefPort
from lava.magma.core.process.variable import Var
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyOutPort, PyRefPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import HostCPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.resources import CPU, Loihi2NeuroCore
from lava.magma.core.model.sub.model import AbstractSubProcessModel

# dhp19 Dataloader ############################################################
class DHP19Dataloader(AbstractProcess):
    """Dataloader object that sends spike for a input sample at a
    set interval and offset (phase).

    Parameters
    ----------
    dataset : Iterable
        The actual dataset object. Dataset is expected to return
        ``(frame, label/ground_truth)`` when indexed.
    """
    
    def __init__(
        self,
        dataset: Iterable,
        ) -> None:
        super().__init__()
        
        data, ground_truth = dataset[0]
        gt_shape = ground_truth.shape
        data_shape = data.shape
        self.proc_params['dataset'] = dataset
        self.frame_out = OutPort(shape=data_shape)
        self.gt_out = OutPort(shape=gt_shape)

@implements(proc=DHP19Dataloader, protocol=LoihiProtocol)
@requires(CPU)
class DHP19DataloaderModel(PyLoihiProcessModel):
    frame_out = LavaPyType(PyOutPort.VEC_DENSE, float)
    gt_out = LavaPyType(PyOutPort.VEC_DENSE, float)
    
    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)
        self.dataset = proc_params['dataset']

    def run_spk(self) -> None:
        self.frame_out.send(self.dataset[self.time_step][0])
        self.gt_out.send(self.dataset[self.time_step][1])
