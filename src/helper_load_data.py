'''
custom organize data set for pytorch 
'''
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

def custom_transform(sample):
    '''
    define the transform of the data from [0,1] 
    '''
    # double float data
    tensor_sample = torch.from_numpy(sample).float()  # Convert numpy array to tensor
    # tensor_sample = torch.from_numpy(sample).double()  # Convert numpy array to tensor
    
    # Normalize to [0,1] 
    normalized_sample = tensor_sample.sub_(0.).mul_(1.0)  # In-place subtraction and multiplication
    return normalized_sample



class custom_datasets(Dataset):
    """Custom dataset for organizing and transforming datasets for PyTorch.

    This dataset class is designed to load data from a specified file path, optionally apply a transformation to the data,
    and support flattening the data for use with fully connected neural network layers.

    Attributes:
        data (numpy.ndarray): The dataset loaded from the specified path, limited to the first two channels.
        channels (int): The number of channels in the dataset.
        dim (int): The dimension of the images (assumed square).
        transform (callable, optional): A function/transform that takes in a sample and returns a transformed version.
        flatten (bool): Whether to flatten the data for use with fully connected layers.

    Args:
        data_path (str): The file path to the dataset, expected to be a .npy file.
        transform (callable, optional): Optional transform to apply to each sample.
        flatten (bool): If True, flattens the data for use with fully connected layers. Default is False.
    """

    print(f'\n please make sure dataset is the row major order....')
    def __init__(self, data_path, transform=None,flatten = False):
        '''
        Args: loading the dataset 
            
            data_path: path to the data, and  size should be [N*C*W*L]
                N: number of images
                C: number of channels (gray scale = 1, RGB = 3)
                W: width of the image
                L: length of the image
            transform: transform the data, default is None
            flatten:  flatten is used  only for a flatten NN layer, default is False
        '''
        if not data_path:
            raise ValueError("Please provide a valid data_path to your dataset.")
        
        self.data = np.load(data_path)[:,:2,:,:]# only first two channel is expected
        # print('data size',self.data.shape)
        self.channels = self.data.shape[1] #number of channels
        # size of each image, square image
        self.dim = self.data.shape[2] 
        self.transform = transform
        self.flatten = flatten
    def __len__(self):
        '''
            Args: return the size of data
            this function will  used for torch.utils.data.DataLoader
        '''
        return len(self.data)
    
    def __getitem__(self, idx):
        '''
        Thix function will  used for torch.utils.data.DataLoader
        Args: 
            idx: a list, size of batch_size, random choose the index of the data
        '''
        # note, the first dimension is the channel, then height*width
        sample = self.data[idx]
        # label = ... # no label for this dataset


        if self.transform: # if transform is not None, normlize the data
            sample = self.transform(sample)
        if self.flatten:
             sample = sample.view(-1)# when flatten is used for a flatten NN layer
        return sample,idx



# # test the data set
# def test_data_load():
#     import torch 
#     from torchvision import transforms
#     from torch.utils.data import DataLoader    
#     from helper_dataset import custom_datasets
#     psedo_data = np.random.rand(359,100)
#     # for i in range(359):
#     #     psedo_data[i,:] = np.repeat(i,100)
#     # np.save('./data/GEOMATRY/psedo_data.npy',psedo_data)
#     data_path = './data/GEOMATRY/pseudo_rbg.npy'
#     data = np.load(data_path)
#     print(data.shape)
#     # dataset = custom_datasets(data_path,transform=custom_transform)

#     # print(dataset[0].shape)
#     # dataset.shape()
#     return data

# dataset = test_data_load()
# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # for x in dataset:
# #     x = x.to(device)
#     print(x.shape)
