
from typing import Dict, Tuple
import numpy as np
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

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
from lava.proc import io

from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg
from lava.utils.system import Loihi2
if Loihi2.is_loihi2_available:
    from lava.proc import embedded_io as eio


class CustomHwRunConfig(Loihi2HwCfg):
    """Custom Loihi2 hardware run config."""
    def __init__(self):
        super().__init__(select_sub_proc_model=True)

    def select(self, proc, proc_models):
        # customize run config
        if isinstance(proc, io.encoder.DeltaEncoder):
            return io.encoder.PyDeltaEncoderModelSparse
        if isinstance(proc, DHP19NetEncoder):
             return DHP19NetNxEncoderModel
        if isinstance(proc, DHP19NetDecoder):
            return DHP19NetNxDecoderModel
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
        if isinstance(proc, DHP19NetEncoder):
            return DHP19NetPyEncoderModel
        if isinstance(proc, DHP19NetDecoder):
            return DHP19NetPyDecoderModel
        return super().select(proc, proc_models)



# # Input Encoder ###############################################################
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
class DHP19NetNxEncoderModel(AbstractSubProcessModel):
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
        self.decoder = DHP19NetFixedPtDecoder(shape=shape)
        self.adapter = eio.spike.NxToPyAdapter(shape=shape,
                                               num_message_bits=32)
        proc.inp.connect(self.adapter.inp)
        self.adapter.out.connect(self.decoder.inp)
        self.decoder.out.connect(proc.out)

# Monitor #####################################################################
class DHP19NetMonitor(AbstractProcess):
    def __init__(self,
                in_shape: Tuple[int, ...],
                in_enc_shape: Tuple[int, ...],
                out_enc_shape: Tuple[int, ...],
                out_dec_shape: Tuple[int, ...],
                output_offset=0,
                num_joints=2,
                ) -> None:
        """DHP19Net monitor process.
        """
        super().__init__()
        self.frame_in = InPort(shape=in_shape)
        self.frame_in_enc = InPort(shape=in_enc_shape)
        self.output_in_enc = InPort(shape=out_enc_shape)
        self.output_in_dec = InPort(shape=out_dec_shape)
        self.proc_params['output_offset'] = output_offset
        self.proc_params['num_joints'] = num_joints
        

@implements(proc=DHP19NetMonitor, protocol=LoihiProtocol)
@requires(CPU)
class DHP19NetMonitorModel(PyLoihiProcessModel):
    """DHP19Net monitor model."""
    frame_in = LavaPyType(PyInPort.VEC_DENSE, float)
    frame_in_enc = LavaPyType(PyInPort.VEC_DENSE, np.int32)
    output_in_enc = LavaPyType(PyInPort.VEC_DENSE, np.int32)
    output_in_dec = LavaPyType(PyInPort.VEC_DENSE, float)

    def __init__(self, proc_params=None) -> None:
        super().__init__(proc_params=proc_params)
        self.fig = plt.figure(figsize=(10, 10)) # create figure

        # self.input_frame = self.fig.add_subplot(111)
        self.input_frame = plt.subplot2grid((4, 2), (0, 0), rowspan=2, colspan=2)
        self.y1 = plt.subplot2grid((4, 2), (2, 0))
        self.x1 = plt.subplot2grid((4, 2), (2, 1))
        self.y2 = plt.subplot2grid((4, 2), (3, 0))
        self.x2 = plt.subplot2grid((4, 2), (3, 1))
        
        self.output_offset = self.proc_params['output_offset']
        self.num_joints = self.proc_params['num_joints']
        self.img_height = 260
        self.img_width = 344
        self.downsample_factor = 2
        self.xy_coord_vec_length = int((self.img_width + self.img_height)/self.downsample_factor)

    def run_spk(self) -> None:
        # print('run_spk')
        frame_data = self.frame_in.recv()
        print('got frame data')
        frame_data_enc = self.frame_in_enc.recv()
        # print('got frame encoded data')
        output_data_enc = self.output_in_enc.recv()
        # print('got output data enc')
        output_data_dec = self.output_in_dec.recv()
        print('got output data dec')
        # print(output_data_enc)
        #print(output_data_dec)

        # get parts of the output data by coordinate and joint
        joint = 0    
        coords_1hot = output_data_dec[joint*self.xy_coord_vec_length:(joint+1)*self.xy_coord_vec_length]
        coords_one_hot_y1 = coords_1hot[0:int(self.img_height / self.downsample_factor)]
        coords_one_hot_x1 = coords_1hot[int(self.img_height / self.downsample_factor):]
    
        joint = 1    
        coords_1hot = output_data_dec[joint*self.xy_coord_vec_length:(joint+1)*self.xy_coord_vec_length]
        coords_one_hot_y2 = coords_1hot[0:int(self.img_height / self.downsample_factor)]
        coords_one_hot_x2 = coords_1hot[int(self.img_height / self.downsample_factor):]

        self.input_frame.clear()
        self.y1.clear()
        self.x1.clear()
        self.y2.clear()
        self.x2.clear()

        self.input_frame.set_title('Input Frame ' + str(self.time_step)) 
        self.input_frame.imshow(np.swapaxes(frame_data[:,:,0],0,1), cmap='gray')
        self.y1.plot(coords_one_hot_y1)
        self.y1.set_ylabel('y')
        self.y1.set_title('Joint 1')
        self.x1.plot(coords_one_hot_x1)
        self.x1.set_ylabel('x')
        self.y2.plot(coords_one_hot_y2)
        self.y2.set_ylabel('y') 
        self.y2.set_title('Joint 2')
        self.x2.plot(coords_one_hot_x2)
        self.x2.set_ylabel('y')
        plt.tight_layout()
        clear_output(wait=True)
        # plt.show()
        display(self.fig)
