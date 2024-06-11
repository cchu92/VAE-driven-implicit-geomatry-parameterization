# define the VAE model
import torch
from torch import nn



# %% DEMO TO DESIGN SIMPLE VAE MODEL
class VAE(nn.Module):
    '''
    Variational Autoencoder (VAE) implemented using linear layers.
    It encodes input data to a latent space representation and then decodes it back to the original input space.
    The VAE consists of two main parts: an encoder and a decoder. Use dense layers to connect them.
    '''
    def __init__(self):
        super().__init__()
        d = 20 # Dimension of latent space vector
        # Encoder network: Maps inputs to the latent space representation.
        # The encoder outputs two things: mean (mu) and log variance (logvar) of the latent space distribution.
        self.encoder = nn.Sequential(
            nn.Linear(784, d ** 2),
            nn.ReLU(),
            nn.Linear(d ** 2, d * 2)
        )
        # Decoder network: Maps latent space representations back to the input space.
        self.decoder = nn.Sequential(
            nn.Linear(d, d ** 2),
            nn.ReLU(),
            nn.Linear(d ** 2, 784),
            nn.Sigmoid(),
        )

    def reparameterise(self, mu, logvar):
        '''
        Reparameterizes the latent space vector to allow backpropagation through random operations.
        This is done by sampling from the distribution defined by mu and logvar using the reparameterization trick.
        
        Parameters:
        - mu: Mean of the latent space distribution.
        - logvar: Log variance of the latent space distribution.
        
        Returns:
        - A sampled latent vector (z).
        '''
        if self.training: # Only add randomness during training.
            std = logvar.mul(0.5).exp_()# Calculates the standard deviation.
            eps = std.new_empty(std.size()).normal_() # Samples epsilon from a standard normal distribution.
            return eps.mul_(std).add_(mu)  # Returns z = mu + epsilon*std.
        else:
            return mu # During evaluation, just use the mean.

    def forward(self, x):
        '''
        Defines the forward pass of the VAE.
        
        Parameters:
        - x: Input tensor.
        
        Returns:
        - Reconstructed input, mean and log variance of the latent space distribution.
        '''
        mu_logvar = self.encoder(x.view(-1, 784)).view(-1, 2, d)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar



        

                     
# %% DEMO TO DESIGN CNN VAE MODEL
class CNN_VAE(nn.Module):
    '''
    Convolutional Variational Autoencoder (CNN-VAE) for encoding input data to a latent space representation
    and decoding it back to the original input space. This class uses convolutional layers for the encoder
    and convolutional transpose layers for the decoder, with dense layers to connect the latent space representation.
    '''

    def __init__(self, channel_in, latent_dim, hidden_dims=None):
        super(CNN_VAE, self).__init__()
        self.latent_dim = latent_dim # latent space vector 
        self.channel_in = channel_in # number of colors

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        # Encoder: Convolutional layers to encode input into a latent representation
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(channel_in, h_dim, kernel_size=7, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            channel_in = h_dim
        self.encoder = nn.Sequential(*modules)

        # Transition between encoder and decoder: Dense layers for mu and log variance of latent space
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, self.latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1] * 4, self.latent_dim)


        # Decoder Input: Prepares latent dimension for decoding process
        self.decoder_input = nn.Linear(self.latent_dim, hidden_dims[-1]*4)
        modules = []
        hidden_dims=[256,128,64,64,64]
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)


        # Final layer: Convolutional transpose layer to match the original input's shape and channel
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=31, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=self.channel_in, kernel_size=2   , stride=1,padding=0),
            nn.Sigmoid()
            # nn.Tanh()
        )
    def encode(self, x):
        ''' Encodes input tensor into a latent space representation (mu and log_var).
        Args: 
            x: input tensor [B*C*W*L] 
        Return:
            mu: mean of latent space vector [B*latent_dim]
            log_var: log variance of latent space vector [B*latent_dim]
        '''
        x = self.encoder(x) #[B*C*W*L]
        x = torch.flatten(x, start_dim=1) #[B*hidden_dims[-1]*4]
        mu = self.fc_mu(x) 
        log_var = self.fc_logvar(x)
        return mu, log_var
        
    def decode(self,z):
        '''  Decodes latent space vectors back into reconstructed images.
        Args:
            z: latent space vector [B*latent_dim]
        Return:
            x_hat: reconstructed images [B*C*W*L]
        '''
        result = self.decoder_input(z)
        result = result.view(-1, 256, 2, 2)  # Here, 256 is the last dimension from `hidden_dims`, adjust accordingly.
        result = self.decoder(result)
        x_hat = self.final_layer(result)
        return x_hat
        
    def rereparameterize(self, mu,log_var):
        '''Reparameterization trick to sample from N(mu, var) from N(0,1),
        for training purpose, add a random noise log_var to the latent space vector,
        for evaluation purpose, only use the mean of latent space vector

        Args: 
            mu: mean of latent space vector [B*latent_dim]
            log_var: log variance of latent space vector [B*latent_dim]
        Return:
            a latent space tesnor [B*latent_dim]

        '''
        if  self.training: # only for training purpose
            std = torch.exp(0.5*log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def forward(self,inputs):
        """  connection of encoder and decoder inputs x and output x_hat
        Args:
            inputs: input tensor [B*C*W*L]
        Return:
            x_hat: reconstructed images [B*C*W*L]
            mu: mean of latent space vector [B*latent_dim]
            log_var: log variance of latent space vector [B*latent_dim]
        """
        mu, log_var = self.encode(inputs)
        z = self.rereparameterize(mu, log_var)
        x_hat = self.decode(z)

        return x_hat,mu,log_var
