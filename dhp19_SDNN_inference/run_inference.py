import numpy as np
import matplotlib.pyplot as plt

from lava.magma.core.run_conditions import RunSteps
from lava.proc import io

from lava.lib.dl import netx
from utils import (
    PilotNetEncoder, PilotNetDecoder, PilotNetMonitor,
    CustomHwRunConfig, CustomSimRunConfig,
    get_input_transform
)





import pickle
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import lava.lib.dl.slayer as slayer
from scipy.ndimage import gaussian_filter1d
from lava.lib.dl import netx


class dhp19(Dataset):
    """
    Dataset class for the dhp19 dataset
    """
    def __init__(self, path):
        self.path = path
        self.files = os.listdir(path)
        self.files = [f for f in self.files if f.endswith('pt')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        session = int(file.split('_')[1][7:])
        subject = int(file.split('_')[0][1:])
        mov = int(file.split('_')[2][3:])
        data_dict = torch.load(self.path + file)

        return data_dict['input_tensor'].to_dense(), data_dict['target_coords_abs'], data_dict['target_coords_rel'], session, subject, mov


def plot_output_sample(input_frame, target_coords, predicted_coords, title=None, path=None):
    plt.imshow(input_frame, cmap='gray')
    plt.plot(target_coords[1], target_coords[0], 'go', label='target')
    plt.plot(predicted_coords[1], predicted_coords[0], 'ro', label='predicted')
    plt.legend()
    if title:
        plt.title(title)
    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()


def event_rate_loss(x, max_rate=0.01):
    """
    Sparsity loss to penalize the network for high event-rate.  
    """
    mean_event_rate = torch.mean(torch.abs(x))
    return F.mse_loss(F.relu(mean_event_rate - max_rate), torch.zeros_like(mean_event_rate))

# target_coords_by_cam = target_coords_abs_by_cam 
def target_coords_to_onehot(target_coords_by_cam, img_height = 260, img_width = 344):
    
    # target_coords_by_cam.shape
    batch_size, num_joints, num_cords, time = target_coords_by_cam.shape
    
    target_coords_by_cam_onehot = torch.zeros(batch_size, num_joints, img_height + img_width, time)
    # target_coords_by_cam_onehot.shape
    
    # set the target coordinates to 1 from the target coordinates
    for batch_idx in range(batch_size):
        for joint_idx in range(num_joints):
            for time_idx in range(time):
                y_coord = int(target_coords_by_cam[batch_idx, joint_idx, 0, time_idx])
                x_coord = int(target_coords_by_cam[batch_idx, joint_idx, 1, time_idx])

                y_coord_one_hot = torch.zeros(img_height)
                y_coord_one_hot[y_coord] = 1

                x_coord_one_hot = torch.zeros(img_width)
                x_coord_one_hot[x_coord] = 1
                xy_coords_one_hot = torch.concat([y_coord_one_hot,x_coord_one_hot])
                # xy_coords_one_hot.shape

                target_coords_by_cam_onehot[batch_idx, joint_idx, : , time_idx] = xy_coords_one_hot
    
    return target_coords_by_cam_onehot

# target_coords_by_cam = target_coords_abs_by_cam 
def target_coords_to_onehot_smoothed(target_coords_by_cam, img_height = 260, img_width = 344, sigma=2):
    
    # target_coords_by_cam.shape
    batch_size, num_joints, num_cords, time = target_coords_by_cam.shape
    target_coords_by_cam_onehot = torch.zeros(batch_size, num_joints, img_height + img_width, time)
    # target_coords_by_cam_onehot.shape
    
    # batch_idx = 0
    # joint_idx = 0
    # time_idx = 0
    # set the target coordinates to 1 from the target coordinates
    for batch_idx in range(batch_size):
        for joint_idx in range(num_joints):
            for time_idx in range(time):
                y_coord = int(target_coords_by_cam[batch_idx, joint_idx, 0, time_idx])
                x_coord = int(target_coords_by_cam[batch_idx, joint_idx, 1, time_idx])

                y_coord_one_hot = torch.zeros(img_height)
                y_coord_one_hot[y_coord] = 1

                x_coord_one_hot = torch.zeros(img_width)
                x_coord_one_hot[x_coord] = 1

                # smooth the one hot encoding
                y_coord_one_hot_smoothed = gaussian_filter1d(y_coord_one_hot, sigma=sigma)
                # plt.plot(y_coord_one_hot, label='original', color='b', linestyle='', marker='.')
                # plt.plot(y_coord_one_hot_smoothed, label='smoothed', color='r')
                # plt.show()
                x_coord_one_hot_smoothed = gaussian_filter1d(x_coord_one_hot, sigma=sigma)

                xy_coords_one_hot_smoothed = torch.concat([torch.tensor(y_coord_one_hot_smoothed), torch.tensor(x_coord_one_hot_smoothed)])
                # xy_coords_one_hot.shape

                target_coords_by_cam_onehot[batch_idx, joint_idx, : , time_idx] = xy_coords_one_hot_smoothed
    
    return target_coords_by_cam_onehot


def reshape_target_coords_onehot_to_flatt_layout(target_coords_by_cam_onehot):
    batch_size, num_joints, xy_onehot_length , time = target_coords_by_cam_onehot.shape
    
    target_coords_by_cam_onehot_flatt = torch.zeros(batch_size, num_joints * xy_onehot_length, time)  
    target_coords_by_cam_onehot_flatt.shape
    
    for batch_idx in range(batch_size):
        for joint_idx in range(num_joints):
            for time_idx in range(time):
                target_coords_by_cam_onehot_flatt[batch_idx, joint_idx*xy_onehot_length:(joint_idx+1)*xy_onehot_length, time_idx] = target_coords_by_cam_onehot[batch_idx, joint_idx, :, time_idx]
    return target_coords_by_cam_onehot_flatt


def compute_batch_mean_euclidean_dist(target_coords_abs_by_cam, output, joint_idxs, img_height=260, img_width=344):
    # target_coords_abs_by_cam.shape
    num_samples = target_coords_abs_by_cam.shape[0]
    num_time_steps = target_coords_abs_by_cam.shape[-1]
    num_joints = len(joint_idxs)
    
    xy_onehot_length = img_height + img_width

    act_device = target_coords_abs_by_cam.device  
    img_height = torch.tensor(img_height).to(act_device)
    img_width = torch.tensor(img_width).to(act_device)
    
    
    # creae tensor to store euclidean distances
    dist_tensor = torch.zeros(num_joints * num_time_steps * num_samples).to(act_device)
    
    # target_coords_abs_by_cam.shape
    dist_count = 0

    # sample_idx = 0
    # time_idx = 1
    # joint_idx = 0

    for sample_idx in range(num_samples):
        for time_idx in range(num_time_steps):
            for joint_idx in range(len(joint_idxs)):
                # get target coords
                target_coords = target_coords_abs_by_cam[sample_idx, joint_idx, :, time_idx]
                # get predicted coords
                
                predicted_coords = output[sample_idx, joint_idx*xy_onehot_length:(joint_idx+1)*xy_onehot_length, time_idx]
                # predicted_coords.shape
                predicted_coords_one_hot_y = predicted_coords[0:img_height]
                predicted_coords_one_hot_x = predicted_coords[img_height:]
                # predicted_coords_one_hot_x.shape
                predicted_y = torch.argmax(predicted_coords_one_hot_y)
                predicted_x = torch.argmax(predicted_coords_one_hot_x)
                                
                predicted_coords = torch.tensor([predicted_y, predicted_x]).to(act_device)
                # compute euclidean distance
                # dist = torch.sqrt(torch.sum((target_coords - predicted_coords)**2))
                distance = torch.norm(target_coords - predicted_coords, p=2)
                dist_tensor[dist_count] = distance
                dist_count += 1
    
    return torch.mean(dist_tensor), dist_tensor


# Define the network
class Network(torch.nn.Module):
    """
    Define the CNN model as SDNN with slayer blocks
    """

    def __init__(self, num_output):
        super(Network, self).__init__()
        self.num_output = num_output

        sdnn_params = { # sigma-delta neuron parameters (taken from PilotNet tutorial)
                'threshold'     : 0.001,    # delta unit threshold
                'tau_grad'      : 0.5,    # delta unit surrogate gradient relaxation parameter
                'scale_grad'    : 1,      # delta unit surrogate gradient scale parameter
                'requires_grad' : True,   # trainable threshold
                'shared_param'  : True,   # layer wise threshold
                'activation'    : F.relu, # activation function
            }
        
        sdnn_cnn_params = { # conv layer has additional mean only batch norm
                **sdnn_params,                                 # copy all sdnn_params
                'norm' : slayer.neuron.norm.MeanOnlyBatchNorm, # mean only quantized batch normalizaton
            }
        
        sdnn_dense_params = { # dense layers have additional dropout units enabled
                **sdnn_cnn_params,                        # copy all sdnn_cnn_params
                'dropout' : slayer.neuron.Dropout(p=0.2), # neuron dropout
            }
        
        sdnn_output_params = { # output layer has additional softmax activation
                'threshold'     : 0.001,    # delta unit threshold
                'tau_grad'      : 0.5,    # delta unit surrogate gradient relaxation parameter
                'scale_grad'    : 1,      # delta unit surrogate gradient scale parameter
                'requires_grad' : True,   # trainable threshold
                'shared_param'  : True,   # layer wise threshold
                'activation'    : torch.sigmoid, # activation function
            }
        
        self.blocks = torch.nn.ModuleList([# sequential network blocks 
                # delta encoding of the input
                slayer.block.sigma_delta.Input(sdnn_params, bias=None), 
                
                # 1. convolution layer 1 stride 1, dilitation 1
                slayer.block.sigma_delta.Conv(sdnn_cnn_params, in_features=1, out_features=4, kernel_size=3, stride=2, padding=1, dilation=1, weight_scale=2, weight_norm=True),
                # torch.Size([B, 8, 130, 172, T])
                
                # 2. convolution layer  stride 1, dilitation 1,
                slayer.block.sigma_delta.Conv(sdnn_cnn_params, in_features=4, out_features=8, kernel_size=3, stride=2, padding=2, dilation=2, weight_scale=2, weight_norm=True),
                # # torch.Size([B, 8, 65, 86, T])
     
                # 5. convolution layer  stride 1, dilitation 2,
                slayer.block.sigma_delta.Conv(sdnn_cnn_params, in_features=8, out_features=8, kernel_size=3, stride=2, padding=2, dilation=2, weight_scale=2, weight_norm=True),
                # # torch.Size([B, 8, 33, 43, T])
                slayer.block.sigma_delta.Conv(sdnn_cnn_params, in_features=8, out_features=8, kernel_size=3, stride=2, padding=2, dilation=2, weight_scale=2, weight_norm=True),
                # torch.Size([B, 8, 17, 22, 8])
                
                # # flatten layer
                slayer.block.sigma_delta.Flatten(),
                # dense layers
                slayer.block.sigma_delta.Dense(sdnn_dense_params, 8*17*22, 2500, weight_scale=2, weight_norm=True),
                # slayer.block.sigma_delta.Dense(sdnn_dense_params, 1000, 250, weight_scale=2, weight_norm=True),
                slayer.block.sigma_delta.Output(sdnn_output_params, 2500, self.num_output, weight_scale=2, weight_norm=True),
        ])  
        

    def forward(self, x):
        count = []
        event_cost = 0

        for block in self.blocks: 
            # forward computation is as simple as calling the blocks in a loop
            x = block(x)
            if hasattr(block, 'neuron'):
                event_cost += event_rate_loss(x)
                count.append(torch.sum(torch.abs((x[..., 1:]) > 0).to(x.dtype)).item())

        return x, event_cost, torch.FloatTensor(count).reshape((1, -1)).to(x.device)


    def grad_flow(self, path=None, show=False):
        # helps monitor the gradient flow
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')]

        # plot grads
        if path:
            plt.figure()
            plt.semilogy(grad)
            if show:
                plt.show()
            else:
                plt.savefig(path + 'gradFlow.png')
                plt.close()

        return grad


    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))


project_path = './'
model_path = project_path + 'results/train/'
event_data_path = project_path + 'data/'

batch_size = 8  # batch size
learning_rate = 0.0001 # leaerning rate
lam = 0.001 # lagrangian for event rate loss
num_epochs = 50  # training epochs
cam_index = 1 # camera index to train on
seq_length = 8 # number of event frames per sequence to be shown to the SDNN
# joint_idxs = [0, 7, 8] # joint indices to train on

model_name = 'sdnn_1hot_smoothed_rmsprop2'
# create experiment name

experiment_name = model_name + \
                '_epochs' + str(num_epochs) + \
                '_lr' + str(learning_rate) + \
                '_batchsize' + str(batch_size) + \
                '_seq' + str(seq_length) + \
                '_cam' + str(cam_index) + \
                '_lam' + str(lam)

act_result_path = model_path + experiment_name + '/'

# load results
with open(act_result_path + 'results.pkl', 'rb') as f:
    results = pickle.load(f)

# print results from original test set
# results.keys()
print('Results for experiment: ' + experiment_name)
print('Best epoch: ' + str(results['best_epoch']))
print('Best Val loss: ' + str(results['best_val_loss']))
print('Test loss: ' + str(results['test_loss']))
print('Test_dist: ' + str(results['test_dist'].detach().cpu()))

joint_idxs = results['joint_idxs']# joint indices [0, 7, 8] (head, right hand, left hand)


#load/define currentz dataset
complete_dataset = dhp19(event_data_path)
num_test_samples = complete_dataset.__len__()

test_loader = DataLoader(dataset=complete_dataset,
                        persistent_workers=True,
                        batch_size=8,
                        shuffle=False,
                        num_workers=12,
                        pin_memory=True)


num_test_batches = len(test_loader)
print('Got ' + str(num_test_samples) + ' samples in the dataset. Each constisting of ' + str(seq_length) + ' consecutive event frames.')
print('Samples are show as batches of size ' + str(batch_size) + ' for ' + str(num_test_batches) + ' batches.')

img_height = 260
img_width = 344
xy_onehot_length = img_height + img_width

# check if cuda is available
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# define the network (needs to match the model that was trained)
model = Network(num_output=len(joint_idxs)*(xy_onehot_length)).to(device)

# forawars a sample of data to get the input dimensions
input, target_coords_abs, target_coords_rel, session, subject, mov = next(iter(test_loader))
input = input[:,:,:,:,cam_index,:].to(device) # N C H W T
output, event_cost, count = model(input)
# load model weights from the trained network
model.load_state_dict(torch.load(act_result_path + 'model.pt'))
print(model)

# test one epoch
running_loss = 0.0
running_dist = 0.0
model.eval()

print('Running model evaluation...', end=' ')
for batch_idx, (input, target_coords_abs, target_coords_rel, session, subject, mov) in enumerate(test_loader):
    # get dims of the input images
    if batch_idx == 0:
        y_img = input.shape[2]
        x_img = input.shape[3]

    input = input[:,:,:,:,cam_index,:].to(device) # N C H W T
    # input.shape        
    target_coords_abs_by_cam = target_coords_abs[:,joint_idxs,:,cam_index,:].to(device) # N J C T
    target_coords_abs_by_cam_onehot = target_coords_to_onehot_smoothed(target_coords_abs_by_cam)
    target_coords_abs_by_cam_onehot_flatt = reshape_target_coords_onehot_to_flatt_layout(target_coords_abs_by_cam_onehot)
    # target_coords_abs_by_cam_onehot_flatt.shape # N J num_joints*(H+W) T
    target_coords_abs_by_cam_onehot_flatt = target_coords_abs_by_cam_onehot_flatt.to(device)
    
    # forward
    with torch.no_grad():
        output, event_cost, count = model(input)
        loss = F.mse_loss(output, target_coords_abs_by_cam_onehot_flatt)

    running_loss += loss.item() * input.size(0)

    # compute batch mean euclidean distance (per joint)
    batch_mean_dist, dists = compute_batch_mean_euclidean_dist(target_coords_abs_by_cam, output, joint_idxs, img_height=y_img, img_width=x_img)
    running_dist += batch_mean_dist

    # plot sequence of event frames
    sample_idx = 0
    joint_idx = 2
    
    input_frames = input[sample_idx, 0,:, :, :].detach().cpu()
    target_coords = target_coords_abs[sample_idx, joint_idxs[joint_idx], :, cam_index, :]
    # output.shape
    predicted_coords = output[sample_idx, joint_idx*xy_onehot_length:(joint_idx+1)*xy_onehot_length, :].detach().cpu()
    # predicted_coords.shape
    predicted_coords_one_hot_y = predicted_coords[0:img_height]
    predicted_coords_one_hot_x = predicted_coords[img_height:]
    
    plt_filename = act_result_path + f'plots/eval/outputs_joint{joint_idxs[joint_idx]}_b{batch_idx}.png'
    plt.figure(figsize=(10, 20))
    num_time_steps = input_frames.shape[-1]
    for time_idx in range(num_time_steps):
        predicted_y = np.where(predicted_coords_one_hot_y[:,time_idx] == predicted_coords_one_hot_y[:,time_idx].max())[0][0]
        predicted_x = np.where(predicted_coords_one_hot_x[:,time_idx] == predicted_coords_one_hot_x[:,time_idx].max())[0][0]

        plt.subplot(num_time_steps, 2, (2*time_idx)+1)
        plt.plot(predicted_coords_one_hot_y[:,time_idx])
        plt.vlines(target_coords[0, time_idx], ymin=0, ymax=1, colors='g', linestyles='dashed', label='target')
        plt.vlines(predicted_y, ymin=0, ymax=1, colors='r', linestyles='dashed', label='predicted')
        plt.ylabel('y output')
        plt.subplot(num_time_steps, 2, (2*time_idx)+2)
        plt.plot(predicted_coords_one_hot_x[:,time_idx])
        plt.vlines(target_coords[1, time_idx], ymin=0, ymax=1, colors='g', linestyles='dashed', label='target')
        plt.vlines(predicted_x, ymin=0, ymax=1, colors='r', linestyles='dashed', label='predicted')
        plt.ylabel('x output')
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.savefig(plt_filename)
    plt.close()

    plt_filename = act_result_path + f'plots/eval/input_joint{joint_idxs[joint_idx]}_b{batch_idx}.png'
    plt.figure(figsize=(20, 10))
    num_time_steps = input_frames.shape[-1]
    for time_idx in range(num_time_steps):                 
        predicted_y = np.where(predicted_coords_one_hot_y[:,time_idx] == predicted_coords_one_hot_y[:,time_idx].max())[0][0]
        predicted_x = np.where(predicted_coords_one_hot_x[:,time_idx] == predicted_coords_one_hot_x[:,time_idx].max())[0][0]
        predicted_coords = np.array([predicted_y, predicted_x])

        plt.subplot(2, 4, time_idx+1)
        plt.imshow(input_frames[:, :, time_idx], cmap='gray')
        plt.plot(target_coords[1, time_idx], target_coords[0, time_idx], 'go', label='target')
        plt.plot(predicted_coords[1], predicted_coords[0], 'ro', label='predicted')
        plt.axis('off')
        plt.title(f'Time {time_idx}')
    plt.savefig(plt_filename)
    plt.close()

test_loss = running_loss / num_test_samples
test_mean_dist = running_dist / num_test_batches
print('Done.')
print('Test Loss: {:.4f} | Test Dist: {:.4f} '.format(test_loss, test_mean_dist))
print('See plots in: ' + act_result_path + 'plots/eval/ for the output of the network on the dataset.')