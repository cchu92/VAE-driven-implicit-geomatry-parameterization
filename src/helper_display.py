from matplotlib import pyplot as plt
import numpy as np
def make_bitmap2(image):
    """
    Converts digital image data into a bitmap with a specific color map.

    Args:
        image (numpy.ndarray): An image represented as a 3D numpy array of shape (n, n, p), where n is the
                               dimension of the image, and p is the number of color channels.

    Returns:
        numpy.ndarray: The color-mapped image data as a numpy array.
    """
    nx = image.shape[0]
    ny = nx
    image = image.reshape(nx * ny, 3)
    color = np.array([[1, 0, 0], [0, 0, 1],[1, 1, 1], [0, 1, 0], [0, 0, 0]])
    # '1,0,0 red' '0,0,1 blue' '1,1,1 white ' '0,1,0' green,'0,0,0' black
    I = np.zeros((nx * ny, 3))
    for j in range(3):
        I[:, :3] += image[:, j, None] * color[j, :3]
    I = I.reshape(ny, nx, 3, order='F')
    return I 
def normalize(data):
    ''' data is the W*H*C numpy data
    '''
    min_vals = np.min(data, axis=(0, 1), keepdims=True)
    max_vals = np.max(data, axis=(0, 1), keepdims=True)
    data_scaled = (data - min_vals) / (max_vals - min_vals)

    return data_scaled


def imshow_compare(in_,
                    out, 
                    N, 
                    label_in=None,
                    label_out = None, 
                    count=False,
                    save_path=None,
                    epoch=None):
    """
    Displays a comparison between original and reconstructed images side by side.

    Args:
        in_ (torch.Tensor): The input images as a PyTorch tensor of shape [B, C, H, W].
        out (torch.Tensor): The reconstructed images as a PyTorch tensor of shape [B, C, H, W].
        N (int): The number of images to display.
        label_in (str, optional): The label for the input images. Defaults to None.
        label_out (str, optional): The label for the output images. Defaults to None.
        count (bool, optional): If True, displays a count on the images. Defaults to False.
        save_path (str, optional): The path to save the comparison images. Defaults to None.
        epoch (int, optional): The current epoch, used in naming the saved file. Defaults to None.

    Returns:
        None
    """

    n = N//4
    if epoch is None:
        epoch = 0
    if save_path is not None:
        # create a folder to save the images
        save_path = './'
    for N in range(n):
        if in_ is not None:
            # Adjust for the number of channels for color images
            in_pic = in_.data.cpu().permute(0, 2, 3, 1)  # Change shape from [B, C, H, W] to [B, H, W, C]
            
            plt.figure(figsize=(18, 4))
            # plt.suptitle(label + ' – real test data / reconstructions', color='w', fontsize=16)
            
            for i in range(4):
                plt.subplot(1,4,i+1,)
                # Squeeze to remove single-dimensional entries from the shape of an array.
                # image= in_pic[i+4*N].squeeze().numpy()# .squeeze() in case in_pic is a batch with B=1, to numpy array
                # image = normalize(image)
                # plt.imshow(make_bitmap2(image))  # 
                plt.imshow(in_pic[i+4*N].squeeze())  # 
                plt.axis('off')
            if label_in is not None:
                plt.title(label_in)
            plt.axis('off')
        
        if out is not None:
            # Similar handling for the output pictures
            out_pic = out.data.cpu().permute(0, 2, 3, 1)  # Change shape from [B, C, H, W] to [B, H, W, C]
            
            plt.figure(figsize=(18, 6))
            for i in range(4):
                plt.subplot(1,4,i+1)
                plt.imshow((out_pic[i+4*N].squeeze()))  # .squeeze() in case out_pic is a batch with B=1
                plt.axis('off')
                if count: plt.title(str(4 * N + i), color='w')
        if label_out is not None:
            plt.title(label_out)
        plt.show()
        plt.savefig(save_path+'compare_epoch_'+str(epoch)+'.png')
        # plt.close('all')



def display_one_row(inputs_tensor):
    """
    Displays a row of images from the provided input tensor.

    Args:
        inputs_tensor (torch.Tensor): The input images as a PyTorch tensor of shape [N, C, H, W].

    Returns:
        None
    """
    len = inputs_tensor.shape[0]
    n_pic = inputs_tensor.data.cpu().permute(0, 2, 3, 1)  # Change shape from [B, C, H, W] to [B, H, W, C]
    plt.clf()
    plt.figure(figsize=(4*len, 4))
    for N in range(len):
        plt.subplot(1,len,N+1)
        image= n_pic[N].squeeze().numpy()# .squeeze() in case in_pic is a batch with B=1, to numpy array
        image = normalize(image)
        plt.imshow(make_bitmap2(image))
        plt.axis('off')
    plt.tight_layout
