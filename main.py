import argparse
from ast import parse
import getpass
import imageio
import json
import os
import random
import numpy as np
import torch
import operators
from siren import *
# from BVPNet import *
from torchvision import transforms
from torchvision.utils import save_image
from training import Trainer

from pri_gen import *
import mcubes


parser = argparse.ArgumentParser()
parser.add_argument("-ld", "--logdir", help="Path to save logs", default=f"/data/litingting/SIREN/siren2/log1")
parser.add_argument("-ni", "--num_iters", help="Number of iterations to train for", type=int, default=10000)
parser.add_argument("-lr", "--learning_rate", help="Learning rate", type=float, default=1e-4)
parser.add_argument("-se", "--seed", help="Random seed", type=int, default=random.randint(1, int(1e6)))
parser.add_argument("-fd", "--full_dataset", help="Whether to use full dataset", action='store_true')
parser.add_argument("-iid", "--image_id", help="Image ID to train on, if not the full dataset", type=int, default=15)
parser.add_argument("-lss", "--layer_size", help="Layer sizes as list of ints", type=int, default=256)
parser.add_argument("-nl", "--num_layers", help="Number of layers", type=int, default=3)
parser.add_argument("-w0", "--w0", help="w0 parameter for SIREN model.", type=float, default=30.0)
parser.add_argument("-w0i", "--w0_initial", help="w0 parameter for first layer of SIREN model.", type=float, default=30.0)
parser.add_argument("-model_type", "--model", help="Options are sine (all sine activations) and mixed (first layer sine, other layers tanh)", type=str, default='sine')

args = parser.parse_args()

# Set up torch and cuda
dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

# Set random seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Dictionary to register mean values (both full precision and half precision)
results = {'fp_bpp': [], 'hp_bpp': [], 'fp_psnr': [], 'hp_psnr': []}

# Create directory to store experiments
if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)

# Fit images
for i in range(1):
    print(f'Image {i}')

    # # Load image
    # img = imageio.imread(f"kodak-dataset/kodim{str(i).zfill(2)}.png")
    # img = transforms.ToTensor()(img).float().to(device, dtype)
    action = np.random.random(4).reshape(4)
    print(action)
    pri = torch.from_numpy(ini_bpri(action,length=128, width=128, height=128)).to(device, dtype)
    # vertices, triangles = mcubes.marching_cubes(pri.cpu().detach().numpy(), 0)
    # print(vertices.shape,triangles.shape)
    # mcubes.export_obj(vertices, triangles, args.logdir + f'/gtpri_reconstruction_{i}.obj')
    
    # Setup model
    func_rep = Siren(
        in_features=3+4,
        out_features=1,
        hidden_features=args.layer_size,
        hidden_layers=args.num_layers,
        outermost_linear=True
    ).to(device)

    # Set up training
    trainer = Trainer(func_rep, lr=args.learning_rate)
    # coordinates, features = operators.to_coordinates_and_features(img)
    features = pri.view(-1, 1)
    coordinates = operators.get_mgrid(128, 3)
    coordinates, features = coordinates.to(device, dtype), features.to(device, dtype)
    coordinates = torch.cat((coordinates, torch.from_numpy(action).unsqueeze(0).repeat(coordinates.shape[0],1).to(device, dtype)),dim=1)

    # Calculate model size. Divide by 8000 to go from bits to kB
    model_size = operators.model_size_in_bits(func_rep) / 8000.
    print(f'Model size: {model_size:.1f}kB')
    fp_bpp = operators.bpp(model=func_rep, image=pri)
    print(f'Full precision bpp: {fp_bpp:.2f}')

    # Train model in full precision
    trainer.train(num_iters=args.num_iters)
    print(f'Best training psnr: {trainer.best_vals["psnr"]:.2f}')

    # Log full precision results
    results['fp_bpp'].append(fp_bpp)
    results['fp_psnr'].append(trainer.best_vals['psnr'])

    # Save best model
    torch.save(trainer.best_model, args.logdir + f'/rrrbest_model_{i}.pt')

    # Update current model to be best model
    func_rep.load_state_dict(trainer.best_model)

    model_output = func_rep(coordinates)[0].cpu().view(128,128,128).detach().numpy()
    vertices, triangles = mcubes.marching_cubes(model_output, 0)
    mcubes.export_obj(vertices, triangles, f'/data/litingting/SIREN/siren2/log/tres_reconstruction_{i}.obj')

    # Save full precision image reconstruction
    with torch.no_grad():
        model_output = func_rep(coordinates)[0].cpu().view(128,128,128).detach().numpy()
        vertices, triangles = mcubes.marching_cubes(model_output, 0)
        mcubes.export_obj(vertices, triangles, args.logdir + f'/fp_reconstruction_{i}.obj')


    # Convert model and coordinates to half precision. Note that half precision
    # torch.sin is only implemented on GPU, so must use cuda
    if torch.cuda.is_available():
        func_rep = func_rep.half().to('cuda')
        coordinates = coordinates.half().to('cuda')

        # Calculate model size in half precision
        hp_bpp = operators.bpp(model=func_rep, image=pri)
        results['hp_bpp'].append(hp_bpp)
        print(f'Half precision bpp: {hp_bpp:.2f}')

        # Compute image reconstruction and PSNR
        with torch.no_grad():
            model_output = func_rep(coordinates)[0].cpu().view(128,128,128).detach().numpy()
            vertices, triangles = mcubes.marching_cubes(model_output, 0)
            mcubes.export_obj(vertices, triangles, args.logdir + f'/hp_reconstruction_{i}.obj')
            hp_psnr = operators.get_clamped_psnr(model_output, pri)
            print(f'Half precision psnr: {hp_psnr:.2f}')
            results['hp_psnr'].append(hp_psnr)
    else:
        results['hp_bpp'].append(fp_bpp)
        results['hp_psnr'].append(0.0)

    # Save logs for individual image
    with open(args.logdir + f'/logs{i}.json', 'w') as f:
        json.dump(trainer.logs, f)

    print('\n')

print('Full results:')
print(results)
with open(args.logdir + f'/results.json', 'w') as f:
    json.dump(results, f)

# Compute and save aggregated results
results_mean = {key: operators.mean(results[key]) for key in results}
with open(args.logdir + f'/results_mean.json', 'w') as f:
    json.dump(results_mean, f)

print('Aggregate results:')
print(f'Full precision, bpp: {results_mean["fp_bpp"]:.2f}, psnr: {results_mean["fp_psnr"]:.2f}')
print(f'Half precision, bpp: {results_mean["hp_bpp"]:.2f}, psnr: {results_mean["hp_psnr"]:.2f}')

