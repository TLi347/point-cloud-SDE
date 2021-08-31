import torch
import tqdm
from collections import OrderedDict
from operators import get_clamped_psnr
from operators import get_mgrid
import matplotlib.pyplot as plt

import pri_gen
from pri_gen import *
from siren import Mapping_Net
import skimage
from skimage.io import imread, imsave, imshow
import mcubes
dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

class Trainer():
    def __init__(self, representation, width=128, lr=1e-3, print_freq=1):
        """Model to learn a representation of a single datapoint.

        Args:
            representation (siren.Siren): Neural net representation of image to
                be trained.
            lr (float): Learning rate to be used in Adam optimizer.
            print_freq (int): Frequency with which to print losses.
        """
        self.representation = representation
        self.optimizer = torch.optim.Adam(self.representation.parameters(), lr=lr)
        self.print_freq = print_freq
        self.steps = 0  # Number of steps taken in training
        self.loss_func = torch.nn.MSELoss()
        self.best_vals = {'psnr': 0.0, 'loss': 1e8}
        self.logs = {'psnr': [], 'loss': []}
        # Store parameters of best model (in terms of highest PSNR achieved)
        self.best_model = OrderedDict((k, v.detach().clone()) for k, v in self.representation.state_dict().items())
        self.width = width

    def train(self, num_iters):
        """Fit neural net to image.

        Args:
            coordinates (torch.Tensor): Tensor of coordinates.
                Shape (num_points, coordinate_dim).
            features (torch.Tensor): Tensor of features. Shape (num_points, feature_dim).
            num_iters (int): Number of iterations to train for.
        """
        with tqdm.trange(num_iters, ncols=100) as t:
            for i in t:
                # Load Primitive
                action = np.random.random(4).reshape(4)
                pri = torch.from_numpy(ini_bpri(action,length=128, width=128, height=128)).to(device, dtype)
                features = pri.view(-1, 1)
                coordinates = get_mgrid(128, 3)
                coordinates, features = coordinates.to(device, dtype), features.to(device, dtype)
                coordinates = torch.cat((coordinates, torch.from_numpy(action).unsqueeze(0).repeat(coordinates.shape[0],1).to(device, dtype)),dim=1)

                # Update model
                predicted,_ = self.representation(coordinates)
                loss = self.loss_func(predicted, features)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Calculate psnr
                psnr = get_clamped_psnr(predicted, features)

                # Print results and update logs
                log_dict = {'loss': loss.item(),
                            'psnr': psnr,
                            'best_psnr': self.best_vals['psnr']}
                t.set_postfix(**log_dict)
                for key in ['loss', 'psnr']:
                    self.logs[key].append(log_dict[key])

                # Update best values
                if loss.item() < self.best_vals['loss']:
                    self.best_vals['loss'] = loss.item()
                # if psnr > self.best_vals['psnr']:
                if not (i+1) % (2000):
                    self.best_vals['psnr'] = psnr
                    # If model achieves best PSNR seen during training, update
                    # model
                    if i > int(num_iters / 2.):
                        for k, v in self.representation.state_dict().items():
                            self.best_model[k].copy_(v)
                    action = np.random.random(4).reshape(4)
                    pri = torch.from_numpy(ini_bpri(action,length=128, width=128, height=128)).to(device, dtype)
                    features = pri.view(-1, 1)
                    coordinates = get_mgrid(128, 3)
                    coordinates, features = coordinates.to(device, dtype), features.to(device, dtype)
                    coordinates = torch.cat((coordinates, torch.from_numpy(action).unsqueeze(0).repeat(coordinates.shape[0],1).to(device, dtype)),dim=1)
                    # coordinates = torch.cat((coordinates, torch.from_numpy(action).unsqueeze(0).repeat(coordinates.shape[0],1).to(device, dtype)),dim=1)
                    model_output = self.representation(coordinates)[0].cpu().view(128,128,128).detach().numpy()
                    vertices, triangles = mcubes.marching_cubes(model_output, 0)
                    mcubes.export_obj(vertices, triangles, f'/data/litingting/SIREN/siren2/log1/tres_reconstruction_{i}.obj')
                    torch.save(self.best_model, f'/data/litingting/SIREN/siren2/log1/rrrbest_model_{i}.pt')

                


        # # load data
        # action = (np.random.random(3).reshape(3))
        # img = torch.from_numpy(ini_bimg(action,width=128, height=128)).to(device, dtype)

        # func_mapping = Mapping_Net(
        #     dim_in = 3,
        #     dim_hidden = 256,
        #     dim_out = 256,
        #     num_layers = 3,
        #     use_bias=True
        # ).to(device)
        # print(func_mapping)
        # action = torch.from_numpy(action).to(device, dtype)
        # gamma = func_mapping(action)
        # print(gamma.shape)