from typing import Dict, Tuple
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.io.encoder import Compression
from lava.lib.dl.netx.utils import NetDict
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.resources import CPU, Loihi2NeuroCore
# from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg

# from lava.proc import io
# from lava.proc.io.encoder import Compression
# from lava.lib.dl.netx.utils import NetDict

# from lava.utils.system import Loihi2
# if Loihi2.is_loihi2_available:
#     from lava.proc import embedded_io as eio


# Input Encoder ###############################################################
class DHP19NetEncoder(AbstractProcess):
    """Input encoder process for DHP19 SDNN.

    Parameters
    ----------
    shape : tuple of ints
        Shape of input.
    net_config : NetDict
        Network configuration.
    compression : Compression mode enum
        Compression mode of encoded data.
    """
    def __init__(self,
                 shape: Tuple[int, ...],
                 net_config: NetDict,
                 compression: Compression = Compression.DELTA_SPARSE_8) -> None:
        super().__init__(shape=shape,
                         vth=net_config['layer'][0]['neuron']['vThMant'],
                         compression=compression)
        self.inp = InPort(shape=shape)
        self.out = OutPort(shape=shape)

@implements(proc=DHP19NetEncoder, protocol=LoihiProtocol)
@requires(CPU)
class DHP19NetPyEncoderModel(AbstractSubProcessModel):
    """DHP19Net encoder model for CPU."""
    def __init__(self, proc: AbstractProcess) -> None:
        self.inp: PyInPort = LavaPyType(np.ndarray, np.int32)
        self.out: PyOutPort = LavaPyType(np.ndarray, np.int32)
        shape = proc.proc_params.get('shape')
        vth = proc.proc_params.get('vth')
        self.encoder = io.encoder.DeltaEncoder(shape=shape,
                                               vth=vth,
                                               spike_exp=6)
        proc.inp.connect(self.encoder.a_in)
        self.encoder.s_out.connect(proc.out)


@implements(proc=DHP19NetEncoder, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class PilotNetNxEncoderModel(AbstractSubProcessModel):
    """DHP19Net encoder model for Loihi 2."""
    def __init__(self, proc: AbstractProcess) -> None:
        self.inp: PyInPort = LavaPyType(np.ndarray, np.int32)
        self.out: PyOutPort = LavaPyType(np.ndarray, np.int32)
        shape = proc.proc_params.get('shape')
        vth = proc.proc_params.get('vth')
        compression = proc.proc_params.get('compression')
        self.encoder = io.encoder.DeltaEncoder(shape=shape, vth=vth,
                                               compression=compression)
        self.adapter = eio.spike.PyToN3ConvAdapter(shape=shape,
                                                   num_message_bits=16,
                                                   spike_exp=6,
                                                   compression=compression)
        proc.inp.connect(self.encoder.a_in)
        self.encoder.s_out.connect(self.adapter.inp)
        self.adapter.out.connect(proc.out)



# Output Decoder ##############################################################
class DHP19NetDecoder(AbstractProcess):
    """Output decoder process for DHP19Net SDNN.

    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of output.
    """
    def __init__(self,
                 shape: Tuple[int, ...]) -> None:
        super().__init__(shape=shape)
        self.inp = InPort(shape=shape)
        self.out = OutPort(shape=shape)

@implements(proc=DHP19NetDecoder, protocol=LoihiProtocol)
@requires(CPU)
class DHP19NetPyDecoderModel(PyLoihiProcessModel):
    """DHP19Net decoder model for CPU."""
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self):
        raw_data = self.inp.recv()
        data = raw_data / (1 << 18)
        self.out.send(data)


class DHP19NetFixedPtDecoder(AbstractProcess):
    """DHP19Net fixed point output decoder."""
    def __init__(self,
                 shape: Tuple[int, ...]) -> None:
        super().__init__(shape=shape)
        self.inp = InPort(shape=shape)
        self.out = OutPort(shape=shape)


@implements(proc=DHP19NetFixedPtDecoder, protocol=LoihiProtocol)
@requires(CPU)
class DHP19NetFixedPtDecoderModel(PyLoihiProcessModel):
    """DHP19Net decoder model for fixed point output."""
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self):
        raw_data = self.inp.recv()
        # interpret it as a 24 bit signed integer
        raw_data = (raw_data.astype(np.int32) << 8) >> 8
        data = raw_data / (1 << 18)
        self.out.send(data)

@implements(proc=DHP19NetDecoder, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class DHP19NetNxDecoderModel(AbstractSubProcessModel):
    """DHP19Net decoder model for Loihi."""
    def __init__(self, proc: AbstractProcess) -> None:
        self.inp: PyInPort = LavaPyType(np.ndarray, np.int32)
        self.out: PyOutPort = LavaPyType(np.ndarray, np.int32)
        shape = proc.proc_params.get('shape')
        self.decoder = PilotNetFixedPtDecoder(shape=shape)
        self.adapter = eio.spike.NxToPyAdapter(shape=shape,
                                               num_message_bits=32)
        proc.inp.connect(self.adapter.inp)
        self.adapter.out.connect(self.decoder.inp)
        self.decoder.out.connect(proc.out)

# Monitor #####################################################################
class DHP19NetMonitor(AbstractProcess):
    def __init__(self,
                 in_shape: Tuple[int, ...],
                 out_shape: Tuple[int, ...],
                 output_offset=0,
                 num_joints=2) -> None:
        """DHP19Net monitor process.

        Parameters
        ----------
        in_shape : Tuple[int, ...]
            Shape of input.
        out_shape : Tuple[int, ...],
            Shape of output.
        output_offset : int, optional
            Latency of output, by default 0.
        num_joints : int
            Number of predicted joints, by default 2.
        """
        super().__init__(in_shape=in_shape, out_shape=out_shape) #, transform=transform)
        self.frame_in = InPort(shape=in_shape)
        self.output_in = InPort(shape=out_shape)
        self.proc_params['output_offset'] = output_offset
        self.proc_params['num_joints'] = num_joints


@implements(proc=DHP19NetMonitor, protocol=LoihiProtocol)
@requires(CPU)
class DHP19NetMonitorModel(PyLoihiProcessModel):
    """PilotNet monitor model."""
    frame_in = LavaPyType(PyInPort.VEC_DENSE, float)
    output_in = LavaPyType(PyInPort.VEC_DENSE, float)

    def __init__(self, proc_params=None) -> None:
        super().__init__(proc_params=proc_params)
        self.fig = plt.figure(figsize=(15, 5))
        
    def run_spk(self) -> None:
        frame_data = self.frame_in.recv()
        output_data = self.output_in.recv()
        print(output_data)
        self.fig.imshow(frame_data)
        display(self.fig)