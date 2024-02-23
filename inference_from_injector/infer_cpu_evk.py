from lava.lib.dl import netx
import logging
import numpy as np

from lava.proc import io
from lava.magma.core.run_conditions import RunSteps, RunContinuous
from lava.magma.core.run_configs import Loihi2SimCfg

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import lava.lib.peripherals.dvs.transformation as tf
from metavision_sdk_core import RoiFilterAlgorithm
from metavision_sdk_cv import AntiFlickerAlgorithm, SpatioTemporalContrastAlgorithm
from evk_utils import (
    PropheseeCamera,
    PyPropheseeCameraEventsIteratorModel,
    EventFrameDisplay)
from dataset import DHP19NetDataset


def get_prediction_from_1hot_vector(vector_1hot, downsample_factor=2, img_height=260, img_width=344):

    # plot model output  as annimation
    xy_coord_vec_length = int((img_width + img_height)/downsample_factor)
    
    # get parts of the output data by coordinate and joint
    joint = 0    
    coords_1hot = vector_1hot[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
    coords_one_hot_y1 = coords_1hot[0:int(img_height / downsample_factor)]
    coords_one_hot_x1 = coords_1hot[int(img_height / downsample_factor):]
    
    coords_one_hot_y1.shape
    # get max 
    max_y1 = np.where(coords_one_hot_y1 == coords_one_hot_y1.max())[0][0]
    max_x1 = np.where(coords_one_hot_x1 == coords_one_hot_x1.max())[0][0]
    
    joint = 1    
    coords_1hot = vector_1hot[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
    coords_one_hot_y2 = coords_1hot[0:int(img_height / downsample_factor)]
    coords_one_hot_x2 = coords_1hot[int(img_height / downsample_factor):]
    # get max
    max_y2 = np.where(coords_one_hot_y2 == coords_one_hot_y2.max())[0][0]
    max_x2 = np.where(coords_one_hot_x2 == coords_one_hot_x2.max())[0][0]
    
    return max_y1, max_x1, max_y2, max_x2, coords_one_hot_y1, coords_one_hot_x1, coords_one_hot_y2, coords_one_hot_x2


if __name__ == '__main__':      
    # Check if Loihi2 compiker is available and import related modules.
    # Loihi2.preferred_partition = 'oheogulch'
    # loihi2_is_available = Loihi2.is_loihi2_available

    loihi2_is_available = False # Force CPU execution
    exception_proc_model_map = {}
    print("Loihi2 compiler is not available in this system. "
        "This tutorial will execute on CPU backend.")
    compression = io.encoder.Compression.DENSE
    system = 'cpu_dense'
    

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

    # Use EVK3 as input
    # Network input shape
    input_shape = (img_height, img_width)

    # EVK settings
    sensor_shape = (260, 344) #if no cropping: (720, 1280)
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

    # Transformations
    # - downsample: 1280x720 -> 344x260
    # - ignore event polarity
    transformations = tf.Compose([
        tf.Downsample(factor={"x": input_shape[1] / sensor_shape[1],
                            "y": input_shape[0] / sensor_shape[0]}),
        tf.MergePolarities()
    ])
    filters = []
    # (Only for EventsIterator) mode can be set to "delta_t" or "n_events" which will load
    # events by timeslice or number of events;
    # if mode is set to "mixed", events will be loaded by the first met criterion
    evk_input_proc = PropheseeCamera(sensor_shape=sensor_shape,
                                    filename=filename,
                                    mode="mixed",
                                    n_events=int(1e4),
                                    delta_t=10000,
                                    transformations=transformations,
                                    filters=filters,
                                    biases=biases)
    exception_proc_model_map[PropheseeCamera] = PyPropheseeCameraEventsIteratorModel
    display = io.extractor.Extractor(shape=(img_height, img_width), buffer_size=128)

    encoder = io.encoder.DeltaEncoder(shape=net.inp.shape,
                                  vth=net.net_config['layer'][0]['neuron']['vThMant'],
                                  spike_exp=12,
                                  num_bits=8,
                                  compression=compression)
    
    receiver = io.extractor.Extractor(shape=net.out.shape, buffer_size=128)
    dequantize = netx.modules.Dequantize(exp=net.spike_exp + 12, num_raw_bits=24)

    # connect modules
    evk_input_proc.s_out.reshape(new_shape=(img_width, img_height, 1)).connect(encoder.a_in)
    encoder.s_out.reshape(new_shape=(img_height, img_width)).connect(display.in_port)
    encoder.s_out.connect(net.inp)
    net.out.connect(receiver.in_port)

    # setup run conditions
    num_steps = 100
    run_condition = RunSteps(num_steps=num_steps, blocking=False)
    
    exception_proc_model_map = {io.encoder.DeltaEncoder: io.encoder.PyDeltaEncoderModelDense}
    run_config = Loihi2SimCfg(select_tag="fixed_pt", exception_proc_model_map=exception_proc_model_map)

    evk_input_proc.run(condition=run_condition, run_cfg=run_config)
    
    input_list = []
    output_dequand_list = []
    output_raw_list = []
    encoder_output_list = []
    
    # t = 1
    for t in range(num_steps):
        
        input = display.receive()        
        model_out = receiver.receive()
        out_dequantized = dequantize(model_out)


        input_list.append(input)
        output_dequand_list.append(out_dequantized)
        output_raw_list.append(model_out)
        print('t = ' + str(t))

    evk_input_proc.wait()
    evk_input_proc.stop()
    print('Done')

    # create annimation of model outputs vs targets
    # inital data
    prediction_y1, prediction_x1, prediction_y2, prediction_x2, output_one_hot_y1, output_one_hot_x1, output_one_hot_y2, output_one_hot_x2 = get_prediction_from_1hot_vector(output_dequand_list[78], downsample_factor=2, img_height=260, img_width=344)

    # initial plot
    ylim = (-1, 1)
    fig, axs = plt.subplots(2, 2)
    y1_plot = axs[0, 0].plot(output_one_hot_y1)
    y1_predict = axs[0,0].axvline(prediction_y1, color='r', ls='--')
    axs[0, 0].set_ylabel('y1')
    axs[0, 0].set_title('Joint 1')
    axs[0, 0].set_ylim(ylim)

    x1_plot = axs[0, 1].plot(output_one_hot_x1)
    x1_predict = axs[0,1].axvline(prediction_x1, color='r', ls='--')
    axs[0, 1].set_ylabel('x1')
    axs[0, 1].set_ylim(ylim)

    y2_plot = axs[1, 0].plot(output_one_hot_y2)
    y2_predict = axs[1,0].axvline(prediction_y2, color='r', ls='--')
    axs[1, 0].set_ylabel('y2')
    axs[1, 0].set_title('Joint 2')
    axs[1, 0].set_ylim(ylim)

    x2_plot = axs[1, 1].plot(output_one_hot_x2)
    x2_predict = axs[1,1].axvline(prediction_x2, color='r', ls='--')
    axs[1, 1].set_ylabel('x2')
    axs[1, 1].set_ylim(ylim)

    title = axs[0, 0].text(0.5,0.85, "frame 0", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, ha="center")
    fig.tight_layout()

    # update function for the animation
    def update(frame):
            # for each frame, update the data stored on each artist.
        prediction_y1, prediction_x1, prediction_y2, prediction_x2, output_one_hot_y1, output_one_hot_x1, output_one_hot_y2, output_one_hot_x2 = \
            get_prediction_from_1hot_vector(output_dequand_list[frame], downsample_factor=2, img_height=260, img_width=344)

        title.set_text('frame ' + str (frame))
        y1_plot[0].set_ydata(output_one_hot_y1)
        y1_predict.set_xdata(prediction_y1)
        
        x1_plot[0].set_ydata(output_one_hot_x1)
        x1_predict.set_xdata(prediction_x1)
        
        y2_plot[0].set_ydata(output_one_hot_y2)
        y2_predict.set_xdata(prediction_y2)

        x2_plot[0].set_ydata(output_one_hot_x2)
        x2_predict.set_xdata(prediction_x2)
        
        return y1_plot, x1_plot, y2_plot, x2_plot, y1_predict

    # create animation
    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(output_dequand_list), interval=200)
    ani.save(filename='model_outputs_' + system + '.mp4', writer='ffmpeg')
        