from matplotlib import pyplot as plt

def plot_input_sample(input_frame, target_coords=None, title=None, path=None):
    """
    Plot a single sample from the dataset.
    Parameters
    ----------
    input_frame : torch.Tensor
        Input frame.
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