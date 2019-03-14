#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from myutils import load_ecg_data, create_dataset, make_train_test, plot_ecg
#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
#%%
import platform
ops = platform.system()
if ops == 'Linux':
    path = '/home/boss/Projects/Data/ecgiddb'
else:
    path = '/Users/tsygal/Desktop/Data/ecgiddb'
#%%
df_signals, _ = load_ecg_data(path)
sample_size = 100
data_inp, data_out = create_dataset(df_signals, n_samples=22000, sample_size=sample_size)
data_inp = np.swapaxes(data_inp, 2, 1)
data_out = np.swapaxes(data_out, 2, 1)
X_train, X_test, Y_train, Y_test = make_train_test(data_inp, data_out)# print(X_train.shape); print(X_test.shape)
#%%
enc =  {'conv_1': nn.Conv1d(1, 128, 1,),
        'conv_2': nn.Conv1d(128, 128, 1,),
        'fc_3': nn.Linear(sample_size, sample_size),
        }
dec =  {'deconv_1' : nn.ConvTranspose1d(128, 128, 1,),
        'deconv_2': nn.ConvTranspose1d(128, 1, 1,),        
        'fc_3': nn.Linear(sample_size, sample_size),
        }
dis =  {'conv_1': nn.ConvTranspose1d(128, 1, 1),
        'fc_2': nn.Linear(sample_size, 1),
        }
# Map shortcuts from a first net layers' outputs
# to the next net layers' inputs.
map_1 = {'enc': [3],
         'dec': [1],
        }
map_2 = {'enc': [3],
         'dis': [1],
        }            
#%%
class Net(nn.Module):
    def __init__(self, net, shortcuts=None, dropout=False, last_fn=None):
        super().__init__()
        self.net = net
        self.shortcuts = shortcuts
        self.model = nn.Sequential()
        for key, layer in self.net.items():
            self.model.add_module(key, layer)
            idx = ''.join([n for n in key if n.isdigit()])
            if int(idx) < len(self.net):
                self.model.add_module('activation_'+idx, nn.Tanh()) #nn.LeakyReLU(0.2)
                if dropout:
                    self.model.add_module('dropout_'+idx, nn.Dropout(dropout))
            elif int(idx) == len(self.net):
                if last_fn:
                    self.model.add_module('activation_'+idx, last_fn)
        self.to(device, dtype=torch.float)

    def forward(self, x, map=None):
        # x = x.float()
        if self.shortcuts:
            output = []; i = 0
            for name, layer in self.model.named_children():
                x = layer(x)
                # print(x.shape)
                if 'activation' in name or len(self.model) - 1 == i:
                    output.append(x)     
                i += 1
            return output
        elif map: 
            origin = list(map.values())[0]
            ending = list(map.values())[1]
            fx = 0.
            for name, layer in self.model.named_children():
                idx = ''.join([n for n in name if n.isdigit()])
                if 'conv' in name and int(idx) in ending:
                    fx = layer(x[origin[ending.index(int(idx))] - 1] + fx)
                else:
                    fx = layer(fx)
            return fx
        else:
            return self.model(x)

#%%
encoder = Net(enc, shortcuts=True, last_fn=nn.Tanh()) # 
[print(name, '-->', layer) for name, layer in encoder.model.named_children()]
decoder = Net(dec, last_fn=nn.Tanh()) # 
[print(name, '-->', layer) for name, layer in decoder.model.named_children()]
discriminator = Net(dis, last_fn=nn.Sigmoid()) # 
[print(name, '-->', layer) for name, layer in discriminator.model.named_children()]
#%%
lr = 0.0001
encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
test_log = {'AE': [], 'D': []}
#%%
_k = X_train.shape[0]
batch_zeros = torch.zeros((_k, 1, 1)).to(device)
batch_ones = torch.ones((_k, 1, 1)).to(device)
print(batch_ones.shape)
#%%
def make_nets(x, y):
    encoder.zero_grad(); decoder.zero_grad(); discriminator.zero_grad
    latent_noisy = encoder(x)
    latent_clean = encoder(y)
    true_pred = discriminator(latent_clean, map_2)
    fake_pred = discriminator(latent_noisy, map_2)
    reconstructed = decoder(latent_noisy, map_1)
    return true_pred, fake_pred, reconstructed
#%%
def train():
    train_loss = 0.
    true_pred, fake_pred, reconstructed = make_nets(X_train, Y_train)

    # train discriminator
    fake_loss = F.binary_cross_entropy_with_logits(fake_pred, batch_zeros)
    true_loss = F.binary_cross_entropy_with_logits(true_pred, batch_ones)
    Disc_loss = 0.5*(fake_loss + true_loss)        
    Disc_loss.backward()
    discriminator_optimizer.step()

    # train autoencoder
    AE_loss = F.l1_loss(reconstructed, Y_train)
    AE_loss.backward()
    encoder_optimizer.step(); decoder_optimizer.step()

    return Disc_loss, AE_loss
#%%
def test(log=False):
    with torch.no_grad():
        reconstructed = make_nets(X_test)
        AE_loss = F.l1_loss(reconstructed, Y_test)        
        if log:
            test_log['AE'].append(AE_loss)
    return reconstructed, AE_loss, test_log                          
#%%
def sample(recon, x=X_test, y=Y_test, samples=1):
    indices = [int((len(x)-samples)/(i+1)) for i in range(samples)]
    for sample in indices:
        x_np = x[sample].cpu().numpy().reshape(-1)
        y_np = y[sample].cpu().numpy().reshape(-1)
        recon_np = recon[sample].cpu().detach().numpy().reshape(-1)
        plot_ecg(x_np, y_np, recon_np)
#%%
for epoch in range(501):

    encoder.train(); decoder.train(); discriminator.train()
    Disc_loss, AE_loss = train()
    if epoch % 100 == 0:
        print(f'Train Epoch: {epoch}, D_loss: {Disc_loss:.4f}, AE_loss: {AE_loss:.4f}')

    encoder.eval(); decoder.eval(); discriminator.eval()
    if epoch % 100 == 0:
        recon, AE_loss, _ = test(log=True)
        print(f'Test loss AE: {AE_loss:.4f}')
        sample(recon, X_test, Y_test, 1)
#%%
sample(recon, X_test, Y_test, 10)