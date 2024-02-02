# run infercence our traine DHP19 model on Loihi?
import logging

import numpy as np
import matplotlib.pyplot as plt

from lava.magma.core.run_conditions import RunSteps, RunContinuous
from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg
from lava.proc import embedded_io as eio
from lava.proc import io

from lava.lib.dl import netx
import lava.lib.peripherals.dvs.transformation as tf
from metavision_sdk_core import RoiFilterAlgorithm
from metavision_sdk_cv import AntiFlickerAlgorithm, SpatioTemporalContrastAlgorithm
from evk_utils import (
    PropheseeCamera,
    PyPropheseeCameraEventsIteratorModel,
    PyPropheseeCameraRawReaderModel,
    EventFrameDisplay)

# import torch
from dataset import DHP19NetDataset
from lava.utils.system import Loihi2


def plot_output_vs_target(model_output, target, downsample_factor=2, img_height=260, img_width=344, time_step=None, filename=None):
    xy_coord_vec_length = int((img_width + img_height)/downsample_factor)
    
    # get parts of the output data by coordinate and joint
    joint = 0    
    coords_1hot = model_output[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
    coords_one_hot_y1 = coords_1hot[0:int(img_height / downsample_factor)]
    coords_one_hot_x1 = coords_1hot[int(img_height / downsample_factor):]
    
    coords_one_hot_y1.shape
    # get prediction from model output
    predicted_y1 = np.where(coords_one_hot_y1 == coords_one_hot_y1.max())[0][0]
    predicted_x1 = np.where(coords_one_hot_x1 == coords_one_hot_x1.max())[0][0]

    # target_1hot = target[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
    # target_one_hot_y1 = target_1hot[0:int(img_height / downsample_factor)]
    # target_one_hot_x1 = target_1hot[int(img_height / downsample_factor):]
    
    # target_one_hot_y1.shape
    # # get prediction from target
    # target_y1 = np.where(target_one_hot_y1 == target_one_hot_y1.max())[0][0]
    # target_x1 = np.where(target_one_hot_x1 == target_one_hot_x1.max())[0][0]    
    
    joint = 1    
    coords_1hot = model_output[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
    coords_one_hot_y2 = coords_1hot[0:int(img_height / downsample_factor)]
    coords_one_hot_x2 = coords_1hot[int(img_height / downsample_factor):]
    # get prediction from model output
    predicted_y2 = np.where(coords_one_hot_y2 == coords_one_hot_y2.max())[0][0]
    predicted_x2 = np.where(coords_one_hot_x2 == coords_one_hot_x2.max())[0][0]
        
    # target_1hot = target[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
    # target_one_hot_y2 = target_1hot[0:int(img_height / downsample_factor)]
    # target_one_hot_x2 = target_1hot[int(img_height / downsample_factor):]
    # # get prediction from target
    # target_y2 = np.where(target_one_hot_y2 == target_one_hot_y2.max())[0][0]
    # target_x2 = np.where(target_one_hot_x2 == target_one_hot_x2.max())[0][0]
    
    fig = plt.figure(figsize=(10, 10)) # create figure

    # self.input_frame = self.fig.add_subplot(111)
    y1 = plt.subplot2grid((2, 2), (0, 0))
    x1 = plt.subplot2grid((2, 2), (0, 1))
    y2 = plt.subplot2grid((2, 2), (1, 0))
    x2 = plt.subplot2grid((2, 2), (1, 1))

    y1.plot(coords_one_hot_y1)
    # y1.plot(target_one_hot_y1)
    # y1.vlines(target_y1, ymin=0, ymax=1, colors='g', linestyles='dashed', label='target')
    y1.vlines(predicted_y1, ymin=0, ymax=1, colors='r', linestyles='dashed', label='predicted')
    y1.set_ylabel('y')
    y1.set_title('Joint 1')

    x1.plot(coords_one_hot_x1)
    # x1.plot(target_one_hot_x1)
    # x1.vlines(target_x1, ymin=0, ymax=1, colors='g', linestyles='dashed', label='target')
    x1.vlines(predicted_x1, ymin=0, ymax=1, colors='r', linestyles='dashed', label='predicted')
    x1.set_ylabel('x')
    
    y2.plot(coords_one_hot_y2)
    # y2.plot(target_one_hot_y2)
    # y2.vlines(target_y2, ymin=0, ymax=1, colors='g', linestyles='dashed', label='target')
    y2.vlines(predicted_y2, ymin=0, ymax=1, colors='r', linestyles='dashed', label='predicted')
    y2.set_ylabel('y') 
    y2.set_title('Joint 2')
    
    x2.plot(coords_one_hot_x2)
    # x2.plot(target_one_hot_x2)
    # x2.vlines(target_x2, ymin=0, ymax=1, colors='g', linestyles='dashed', label='target')
    x2.vlines(predicted_x2, ymin=0, ymax=1, colors='r', linestyles='dashed', label='predicted')
    x2.set_ylabel('y')
    fig.tight_layout()
    if time_step is not None:
        fig.suptitle('Model output at time step ' + str(time_step))
    else:
        fig.suptitle('Model output')

    if filename is not None:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_input_vs_prediction_vs_target(input, model_output, target, downsample_factor=2, img_height=260, img_width=344, time_step=None, filename=None):
    xy_coord_vec_length = int((img_width + img_height)/downsample_factor)
    
    # get parts of the output data by coordinate and joint
    joint = 0    
    coords_1hot = model_output[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
    coords_one_hot_y1 = coords_1hot[0:int(img_height / downsample_factor)]
    coords_one_hot_x1 = coords_1hot[int(img_height / downsample_factor):]
    
    coords_one_hot_y1.shape
    # get prediction from model output
    predicted_y1 = np.where(coords_one_hot_y1 == coords_one_hot_y1.max())[0][0]
    predicted_x1 = np.where(coords_one_hot_x1 == coords_one_hot_x1.max())[0][0]
    predicted_y1 = predicted_y1 * downsample_factor
    predicted_x1 = predicted_x1 * downsample_factor

    # target_1hot = target[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
    # target_one_hot_y1 = target_1hot[0:int(img_height / downsample_factor)]
    # target_one_hot_x1 = target_1hot[int(img_height / downsample_factor):]
    
    # target_one_hot_y1.shape
    # # get prediction from target
    # target_y1 = np.where(target_one_hot_y1 == target_one_hot_y1.max())[0][0]
    # target_x1 = np.where(target_one_hot_x1 == target_one_hot_x1.max())[0][0]    
    # target_y1 = target_y1 * downsample_factor
    # target_x1 = target_x1 * downsample_factor


    joint = 1    
    coords_1hot = model_output[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
    coords_one_hot_y2 = coords_1hot[0:int(img_height / downsample_factor)]
    coords_one_hot_x2 = coords_1hot[int(img_height / downsample_factor):]
    # get prediction from model output
    predicted_y2 = np.where(coords_one_hot_y2 == coords_one_hot_y2.max())[0][0]
    predicted_x2 = np.where(coords_one_hot_x2 == coords_one_hot_x2.max())[0][0]
    predicted_y2 = predicted_y2 * downsample_factor
    predicted_x2 = predicted_x2 * downsample_factor
        
    # target_1hot = target[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
    # target_one_hot_y2 = target_1hot[0:int(img_height / downsample_factor)]
    # target_one_hot_x2 = target_1hot[int(img_height / downsample_factor):]
    # # get prediction from target
    # target_y2 = np.where(target_one_hot_y2 == target_one_hot_y2.max())[0][0]
    # target_x2 = np.where(target_one_hot_x2 == target_one_hot_x2.max())[0][0]
    # target_y2 = target_y2 * downsample_factor
    # target_x2 = target_x2 * downsample_factor
        
    fig = plt.figure(figsize=(10,5)) # create figure
    joint1 = plt.subplot2grid((1, 2), (0, 0))
    joint2 = plt.subplot2grid((1, 2), (0, 1))

    # joint1.imshow(np.swapaxes(input[:,:,0],0,1), cmap='gray')
    joint1.imshow(input, cmap='gray')
    # joint1.plot(target_x1, target_y1, 'go', label='target')
    joint1.plot(predicted_x1, predicted_y1, 'ro', label='predicted')

    # joint2.imshow(np.swapaxes(input[:,:,0],0,1), cmap='gray')
    joint2.imshow(input, cmap='gray')
    # joint2.plot(target_x2, target_y2, 'go', label='target')
    joint2.plot(predicted_x2, predicted_y2, 'ro', label='predicted')

    fig.tight_layout()
    if time_step is not None:
        fig.suptitle('Model output at time step ' + str(time_step))
    else:
        fig.suptitle('Model output')

    if filename is not None:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':      
    # Check if Loihi2 compiker is available and import related modules.
    # Loihi2.preferred_partition = 'oheogulch'
    loihi2_is_available = False # force Loihi2
    recording = True
    exception_proc_model_map = {}
    
    if loihi2_is_available:
        print(f'Running on loihi2')
        from lava.magma.compiler.subcompilers.nc.ncproc_compiler import CompilerOptions
        CompilerOptions.verbose = True
        compression = io.encoder.Compression.DELTA_SPARSE_8
        # compression = io.encoder.Compression.DENSE
        system = 'loihi2_delta_sparse_8'
    else:
        print("Loihi2 compiler is not available in this system. "
            "This tutorial will execute on CPU backend.")
        compression = io.encoder.Compression.DENSE
        system = 'cpu' 

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
    model_name = 'sdnn_1hot_smoothed_scaled_lowres_rmsprop_relu_v2'
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
    # net.out_layer.neuron.sigma
    # len(net)
    # net.inp.shape
    # net.out.shape
    # net.input_message_bits
    # net.output_message_bits
    # net.spike_exp
    # net.in_layer.output_message_bits
    # net.out_layer.neuron

    # # print('Loading net ' + experiment_name    )
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

    # Use EVK3 as input
    # Network input shape
    input_shape = (img_height, img_width)

    # EVK settings
    sensor_shape = (260, 344)
    filename = ""
    n_events = 5000
    biases = {
        'bias_diff': 0,
        'bias_diff_off': 0,
        'bias_diff_on': 0,
        'bias_fo': 0,
        'bias_hpf': 0,
        'bias_refr': 0,
        }
    
    if recording:
        filters = [
            RoiFilterAlgorithm(x0=325,
                            y0=120,
                            x1=325 + sensor_shape[1],
                            y1=120 + sensor_shape[0],
                            output_relative_coordinates=True
            ),
            AntiFlickerAlgorithm(
                width=sensor_shape[1],
                height=sensor_shape[0],
                min_freq=15,
                max_freq=1000,
                filter_length=2,
                diff_thresh_us=100000
            ),
            SpatioTemporalContrastAlgorithm(
                width=sensor_shape[1],
                height=sensor_shape[0],
                threshold=100000
            )
        ]
        biases = None
        filename = "../data/lja_test_1.raw"
        n_events = 100000000

    # Transformations
    # - downsample: 1280x720 -> 344x260
    # - ignore event polarity
    transformations = tf.Compose([
        tf.Downsample(factor={"x": input_shape[1] / sensor_shape[1],
                            "y": input_shape[0] / sensor_shape[0]}),
        tf.MergePolarities()
    ])

    # (Only for EventsIterator) mode can be set to "delta_t" or "n_events" which will load
    # events by timeslice or number of events;
    # if mode is set to "mixed", events will be loaded by the first met criterion
    evk_input_proc = PropheseeCamera(sensor_shape=sensor_shape,
                                    filename=filename,
                                    mode="n_events",
                                    n_events=n_events,
                                    transformations=transformations,
                                    filters=filters,
                                    biases=biases)
    # exception_proc_model_map[PropheseeCamera] = PyPropheseeCameraEventsIteratorModel
    exception_proc_model_map[PropheseeCamera] = PyPropheseeCameraRawReaderModel

    # Visualization
    # display = EventFrameDisplay(event_frame_shape=(img_height, img_width))
    display = io.extractor.Extractor(shape=(img_height, img_width), buffer_size=128)

    encoder = io.encoder.DeltaEncoder(shape=net.inp.shape,
                                  vth=net.net_config['layer'][0]['neuron']['vThMant'],
                                  spike_exp=0,
                                  num_bits=8,
                                  compression=compression)
    
    inp_adapter = eio.spike.PyToN3ConvAdapter(shape=encoder.s_out.shape,
                                              num_message_bits=16,
                                              spike_exp=net.spike_exp,
                                              compression=compression)
    
    #state_adapter = eio.state.ReadConv(shape=net.out.shape)
    state_adapter = eio.state.Read(shape=net.out.shape)

    receiver = io.extractor.Extractor(shape=net.out.shape, buffer_size=128)
    
    # Connect the processes
    # Input: event frame aggregation
    evk_input_proc.s_out.reshape(new_shape=(img_width, img_height, 1)).connect(encoder.a_in)
    # Output: display event-frame in OpenCV window
    # evk_input_proc.s_out.reshape(new_shape=(img_height, img_width)).connect(display.event_frame_port)
    # Output: grab event-frame from Extractor
    evk_input_proc.s_out.reshape(new_shape=(img_height, img_width)).connect(display.in_port)

    if loihi2_is_available:
        encoder.s_out.connect(inp_adapter.inp)
        inp_adapter.out.connect(net.inp)
        state_adapter.connect_var(net.out_layer.neuron.sigma)
        state_adapter.out.connect(receiver.in_port)
    else:
        encoder.s_out.connect(net.inp)
        net.out.connect(receiver.in_port)
    
    # setup run conditions
    num_steps = 100
    run_condition = RunSteps(num_steps=num_steps, blocking=False)

    if loihi2_is_available:
        exception_proc_model_map[io.encoder.DeltaEncoder] = io.encoder.PyDeltaEncoderModelSparse
        run_config = Loihi2HwCfg(exception_proc_model_map=exception_proc_model_map)
    else:
        exception_proc_model_map[io.encoder.DeltaEncoder] = io.encoder.PyDeltaEncoderModelDense
        run_config = Loihi2SimCfg(select_tag="fixed_pt", exception_proc_model_map=exception_proc_model_map)

    evk_input_proc.run(condition=run_condition, run_cfg=run_config)

    # t = 1
    for t in range(num_steps):
        
        input = display.receive()
        target = None
        model_out = receiver.receive()


        # show_model_output(out_dequantized, downsample_factor=2, img_height=260, img_width=344, time_step=t)   
        plot_output_vs_target(model_out, target, downsample_factor=2, img_height=260, img_width=344, time_step=t, filename='plots/output_vs_target' + str(t) + '_' + system + '.png')
        plot_input_vs_prediction_vs_target(input, model_out, target, downsample_factor=2, img_height=260, img_width=344, time_step=t, filename='plots/input_vs_prediction_vs_target_' + str(t) + '_' + system + '.png')
        print('t = ' + str(t))
        
    evk_input_proc.wait()
    evk_input_proc.stop()
    print('Done')
