# run infercence our traine DHP19 model on Loihi?
from lava.lib.dl import netx
import logging
from dataset import DHP19NetDataset

from lava.proc import io
# from lava.proc import embedded_io as eio
import dataloader 
from plot import plot_input_sample
import numpy as np
import matplotlib.pyplot as plt
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg
from lava.utils.system import Loihi2
# from icecream import ic
from dataset import DHP19NetDataset
# from utils import (
#     CustomHwRunConfig, CustomSimRunConfig, DHP19NetMonitor, DHP19NetEncoder, DHP19NetDecoder
# )

if __name__ == '__main__':      
    # Check if Loihi2 compiker is available and import related modules.
    Loihi2.preferred_partition = 'oheogulch'
    loihi2_is_available = Loihi2.is_loihi2_available
    # loihi2_is_available = False # Force CPU execution

    if loihi2_is_available:
        print(f'Running on {Loihi2.partition}')
        # from lava.magma.compiler.subcompilers.nc.ncproc_compiler import CompilerOptions
        # CompilerOptions.verbose = True
        compression = io.encoder.Compression.DELTA_SPARSE_8
    else:
        print("Loihi2 compiler is not available in this system. "
            "This tutorial will execute on CPU backend.")
        compression = io.encoder.Compression.DENSE

    # Set paths to model and data
    project_path = './'
    # project_path = '/home/hand_detection_SDNN_inference/dataloader_monitor_encoder_test/'
    model_path = project_path + '../model/train/'
    event_data_path = project_path + '../data/dhp19/'
    # paramters of the traininf data
    img_width = 344
    img_height = 260
    downsample_factor = 2
    xy_coord_vec_length = int((img_width + img_height)/downsample_factor)

    # Create Dataset instance
    batch_size = 8  # batch size
    learning_rate = 0.00005 # leaerning rate
    lam = 0.001 # lagrangian for event rate loss
    num_epochs = 30  # training epochs
    # steps  = [60, 120, 160] # learning rate reduction milestones
    cam_idxs = [1,2] # camera index to train on 1,2 are frontal views
    seq_length = 8 # number of event frames per sequence to be shown to the SDNN
    joint_idxs = [7, 8] # joint indices to train on
    num_joints = len(joint_idxs)

    # Load model
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
    net = netx.hdf5.Network(net_config=act_model_path + 'model.net', skip_layers=1)
    print(net)
    len(net)
    # print('Loading net ' + experiment_name    )
    complete_dataset = DHP19NetDataset(path=event_data_path, joint_idxs=joint_idxs, cam_id=cam_idxs[0], num_time_steps=seq_length)
    print('Dataset loaded: ' + str(len(complete_dataset)) + ' samples found')

    # show sample from the dataset
    # input, target = complete_dataset[0]
    # input.shape # (344, 260) W x H x 1
    # # input.max()
    # # input.min()
    # # target.shape # ((img_widht + img_height)/downsample_factor) * joints (604, )
    # # plot input sample
    # plot_input_sample(np.swapaxes(input[:,:,0],0,1), target_coords=None, title='Input frame', path=None)

    # y_ind = np.where(input[:,:,0] > 0)[0]
    # x_ind = np.where(input[:,:,0] > 0)[1]
    # input[y_ind,x_ind,0]
    # Create Dataloader
    # The dataloader process reads data from the dataset objects and sends out the input frame and ground truth as spikes.
    dataloader = dataloader.DHP19Dataloader(dataset=complete_dataset)
    
    # Input encoding
    quantize = netx.modules.Quantize(exp=6)  # convert to fixed point representation with 6 bit of fraction
    sender = io.injector.Injector(shape=net.inp.shape, buffer_size=128)
    sender.out_port.shape
    if loihi2_is_available:
        # This is needed for the time being until high speed IO is enabled
        inp_adapter = eio.spike.PyToN3ConvAdapter(shape=sender.out_port.shape,
                                                num_message_bits=16,
                                                spike_exp=net.spike_exp,
                                                compression=compression)
    # Output decoding 
    if loihi2_is_available:
        # This is needed for the time being until high speed IO is enabled
        state_adapter = eio.state.ReadConv(shape=net.out.shape)

    receiver = io.extractor.Extractor(shape=net.out.shape, buffer_size=128)
    dequantize = netx.modules.Dequantize(exp=net.spike_exp + 12, num_raw_bits=24)


    # Data buffers / delays
    # There is a latency in the prediction equal to the number of layers the network has and the encoding step.
    # Two FIFO buffers are used to synchronize the input frame and target annotation with the predicted output.
    frame_buffer = netx.modules.FIFO(depth=len(net) + 1)
    annotation_buffer = netx.modules.FIFO(depth=len(net) + 1)
    
    # connect the proceses    
    if loihi2_is_available:
        sender.out_port.connect(inp_adapter.inp)
        inp_adapter.out.connect(net.inp)
        state_adapter.connect_var(net.out_layer.neuron.sigma)
        state_adapter.out.connect(receiver.in_port)
    else:
        sender.out_port.connect(net.inp)
        net.out.connect(receiver.in_port)

    # setup run conditions
    num_steps = 256
    run_condition = RunSteps(num_steps=num_steps, blocking=False)
    
    if loihi2_is_available:
        exception_proc_model_map = {io.encoder.DeltaEncoder: io.encoder.PyDeltaEncoderModelSparse}
        run_config = Loihi2HwCfg(exception_proc_model_map=exception_proc_model_map)
    else:
        exception_proc_model_map = {io.encoder.DeltaEncoder: io.encoder.PyDeltaEncoderModelDense}
        run_config = Loihi2SimCfg(select_tag='fixed_pt',
                              exception_proc_model_map=exception_proc_model_map)

    sender._log_config.level = logging.WARN
    sender.run(condition=run_condition, run_cfg=run_config)
    
    t = 0
    for t in range(num_steps):
        input, target = complete_dataset[t]
        input.shape
        frame_quantized = quantize(input)
        frame_quantized.max()
        frame_quantized.min()
        
        sender.send(frame_quantized)        # This sends the input frame to the Lava network
        model_out = receiver.receive()  # This receives the output from the Lava network
        out_dequantized = dequantize(model_out)
        
        input_frame = frame_buffer(input)
        gt = annotation_buffer(target)

        print('Input frame: ' + str(t))
        print(out_dequantized)

    sender.wait()
    sender.stop()
    # Create Logger for targets
    gt_logger = io.sink.RingBuffer(shape=target.shape, buffer=num_steps)
    output_logger = io.sink.RingBuffer(shape=net.out.shape, buffer=num_steps)
    input_logger = io.sink.RingBuffer(shape=net.inp.shape, buffer=num_steps)
    
    # Create Monitor
    dhp19_monitor = DHP19NetMonitor(in_shape=net.inp.shape,
                                    model_out_shape=net.out.shape)
    # connect processes
    dataloader.gt_out.connect(gt_logger.a_in)
    dataloader.frame_out.connect(dhp19_monitor.frame_in)
    dataloader.frame_out.connect(quantize.a_in)
    quantize.out.connect(input_logger.a_in)
    # net.out.connect(dhp19_monitor.model_output_in)
    # net.out.connect(output_logger.a_in)

    run_condition = RunSteps(num_steps=num_steps)

    if loihi2_is_available:
        run_config = CustomHwRunConfig()
    else:
        run_config = CustomSimRunConfig()

    input_logger.run(condition=run_condition, run_cfg=run_config)
    input_logger.stop()
    print('DONE')
    gts = gt_logger.data.get()
    quant_inputs = input_logger.data.get()
    print(gts)
    print(quant_inputs)
    
    
    # net.run(condition=run_condition, run_cfg=run_config)
    # # input_encoder.run(condition=run_condition, run_cfg=run_config)
    # gts = gt_logger.data.get()
    # outputs = output_logger.data.get()
    # net.stop()
    # print('DONE')
    # print(gts)
    # print(outputs)
    
    # gts.shape
    # plt.plot(gts[:,0], label='Ground Truth')
    # plt.plot(outputs[:,0], label='Output')
    # plt.plot(outputs[:,1], label='Output')
    # plt.plot(outputs[:,2], label='Output')
    # plt.plot(outputs[:,3], label='Output')
    # plt.plot(outputs[:,4], label='Output')
    # plt.plot(outputs[:,5], label='Output')
    # plt.plot(outputs[:,6], label='Output')
    # plt.plot(outputs[:,7], label='Output')
    # plt.plot(outputs[:,8], label='Output')
    # plt.plot(outputs[:,9], label='Output')
    
    # import torch.nn.functional as F
    # import torch
    
    # raw_data = outputs[:,6]
    # raw_data = (raw_data.astype(np.int32) << 8) >> 8
    # o_scaled =  raw_data / (1 << 18)
    # plt.plot(o_scaled)
    # o_sig = F.sigmoid(torch.tensor(o_scaled)) 
    # plt.plot(o_sig)