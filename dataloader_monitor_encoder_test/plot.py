from matplotlib import pyplot as plt
import numpy as np

def plot_input_sample(input_frame, target_coords=None, title=None, path=None):
    """
    Plot a single sample from the dataset.
    Parameters
    ----------
    input_frame : torch.Tensor
        Input frame. (H x W) e.g (260 x 344)
    target_coords : torch.Tensor, optional
        Target coordinates. The default is None.
    title : str, optional
        Title of the plot. The default is None.
    path : str, optional
        Path to save the plot. The default is None.
    Returns
    -------
    None.
    
    """
    plt.imshow(input_frame, cmap='gray')
    if target_coords is not None:
        plt.plot(target_coords[1], target_coords[0], 'go', label='target')
        plt.legend()
    if title:
        plt.title(title)
    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()


def plot_target_coords_1hot(target, joint_idxs, img_height, img_width, downsample_factor, path=None):
    num_joints = len(joint_idxs)
    xy_coord_vec_length = int((img_width + img_height)/downsample_factor)

    # show target
    plt.figure(figsize=(20, 10))
    for joint in range(num_joints):
        coords_1hot = target[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
        coords_one_hot_y = coords_1hot[0:int(img_height / downsample_factor)]
        coords_one_hot_x = coords_1hot[int(img_height / downsample_factor):]
        act_target_y = np.where(coords_one_hot_y == coords_one_hot_y.max())[0][0]
        act_target_x = np.where(coords_one_hot_x == coords_one_hot_x.max())[0][0]

        plt.subplot(num_joints, 2, (2*joint)+1)
        plt.plot(coords_one_hot_y)
        plt.vlines(act_target_y, ymin=0, ymax=1, colors='g', linestyles='dashed', label='target')
        plt.ylabel('y target')
        plt.title('Joint ' + str(joint_idxs[joint]))
        
        plt.subplot(num_joints, 2, (2*joint)+2)
        plt.plot(coords_one_hot_x)
        plt.vlines(act_target_x, ymin=0, ymax=1, colors='g', linestyles='dashed', label='target')
        plt.ylabel('x target')
    
    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()
