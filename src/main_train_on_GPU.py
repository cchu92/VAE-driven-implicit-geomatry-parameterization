### Import public pkgs
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import json


###  Import inhouse pkgs
from helper_load_data import custom_datasets
from helper_load_data import custom_transform
# from helper_display import imshow_compare



# Load configuration file
with open('./config_cluster.json') as f:
    config = json.load(f)

# Extract configuration parameters
batch_size = config['model_params']['batch_size']
latent_dim = config['model_params']['latent_dim']
beta = config['model_params']['beta']
learning_rate = config['train_params']['learning_rate']
epochs = config['train_params']['epochs']
manual_seed = config['random_seed']['manual_seed']
cuda_manual_seed = config['random_seed']['cuda_manual_seed']
loading_checkpoint = config['train_params']['loading_checkpoint']
# Paths from configuration
data_path_train = config['Path']['train_data_path']
data_path_test = config['Path']['test_data_path']
save_path = config['Path']['save_path']
checkpoint_path = config['Path']['log_path']




# Set random seeds for reproducibility
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(cuda_manual_seed)




# DataLoader parameters
kwargs = {'num_workers': 0, 'pin_memory': True} 
# Initialize DataLoaders for training and testing
train_loader = custom_datasets(data_path_train,transform=custom_transform,flatten=False)
train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True, **kwargs)


test_loader = custom_datasets(data_path_test,transform=custom_transform,flatten=False)
test_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=True, **kwargs)


# Initialize the model and optimizer
from helper_VAEstruc import VAE,CNN_VAE
model = CNN_VAE(channel_in=2,latent_dim=latent_dim)

# Setup device (GPU/CPU)
if torch.cuda.is_available(): # GPU is available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    device = torch.device("cuda:0")
    model.to(device)
else:  # only cpu is available
    device = torch.device("cpu")
    model.to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)


# Load checkpoint if specified
if loading_checkpoint:
    checkpoint = torch.load(checkpoint_path+'checkpoint.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    current_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
else: # use the initial model and optimiser
    current_epoch = 0
   

    
# Define the loss function
def lossfunc(x,x_hat,mu,logvar,beta):
    """
    Computes the Variational Autoencoder (VAE) loss function, combining reconstruction loss and KL divergence.

    Args:
        x (torch.Tensor): Original input images.
        x_hat (torch.Tensor): Reconstructed images.
        mu (torch.Tensor): Mean of the latent variables.
        logvar (torch.Tensor): Log variance of the latent variables.
        beta (float): Weight for the KL divergence part of the loss.

    Returns:
        torch.Tensor: The computed loss value.
    """
    recons_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1), dim = 0)
    return recons_loss + beta* kl_loss






# Training loop
for epoch in range(current_epoch, epochs+1):
    mu_list = list()
    x_test_list = list()
    model.train()
    train_loss = 0
    for x,y in train_loader:
        x = x.to(device)
        x_hat,mu,logvar = model(x)
        mu_list.append(mu.detach())
        #==== forwad pass
        loss = lossfunc(x,x_hat,mu,logvar,beta=beta)
        train_loss += loss.item()
        #==== backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 40 == 0: # save model every 40 steps
        save_model = 'VAEmodel_'+str(epoch) + '.pt'
        torch.save(model.module.state_dict(), save_path+save_model)
        # imshow_compare(x,x_hat,n=4,epoch=epoch)
        mu_list = torch.cat(mu_list,dim=0)
        save_mu = 'mu_list_'+str(epoch) + '.pt'
        torch.save(mu_list, save_path+save_mu)
        with torch.no_grad():
            model.eval()
            mu_list_test = list()
            test_loss = 0
            for x,y in test_loader:
                x = x.to(device)
                x_hat,mu,logvar = model(x)
                loss = lossfunc(x,x_hat,mu,logvar,beta=beta)
                test_loss += loss.item()
                mu_list_test.append(mu.detach())
                x_test_list.append(x.detach())

            # Save the 'mu' the latent space
            mu_list_test = torch.cat(mu_list_test,dim=0)
            save_mu = 'mu_list_test_'+str(epoch) + '.pt'
            torch.save(mu_list_test, save_path+save_mu)
            
            x_test_list = torch.cat(x_test_list,dim=0)
            save_x_test = 'x_test_'+str(epoch) + '.pt'
            torch.save(x_test_list, save_path+save_x_test)


        
    
    print(f'====> Epoch: {epoch} Average loss:{train_loss/len(train_loader.dataset):.4f}')

    # torch.save(model.state_dict(), save_path+save_model)  
# only save last on 
# 
save_model = 'VAEmodel_'+'last'+ '.pt'
save_mu = 'mu_list_'+'last'+ '.pt'
save_mu_test = 'mu_list_test_'+'last'+ '.pt'
save_x_test = 'x_test_'+'last'+ '.pt'
# torch.save(model.state_dict(), save_path+save_model)  


# Saving torch.nn.DataParallel Model for multi-GPU training
torch.save(model.module.state_dict(), save_path+save_model)
torch.save(mu_list, save_path+save_mu)
torch.save(mu_list_test, save_path+save_mu_test)
torch.save(x_test_list, save_path+save_x_test)


# Save the final model and other states
torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, checkpoint_path+'checkpoint.tar')




