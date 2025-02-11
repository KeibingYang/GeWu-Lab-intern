import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
plt.rc('font', family="Arial")
plt.rcParams['font.size'] = '14'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(202310)

from net import shallow_net, early_fusion, late_fusion, deep_fusion, fission
from config import config
from data import *
from vis import *
from sweep import *


def creat_network(args, in_dim, out_dim):
    # Model instantiation
    if args.mode == "shallow":
        network = shallow_net(in_dim, out_dim, args.init)
    elif args.mode == "early_fusion":
        network = early_fusion(in_dim, args.hid_width, out_dim, args.relu, args.bias, args.init)
    elif args.mode == "late_fusion":
        network = late_fusion(in_dim, args.hid_width, out_dim, args.relu, args.bias, args.init)
    elif args.mode == "deep_fusion":
        network = deep_fusion(in_dim, args.hid_width, out_dim, args.depth, args.fuse_depth, args.relu, args.bias, args.init)
    elif args.mode == "fission":
        network = fission(in_dim, args.hid_width, out_dim, args.bias, args.init)
    print(network)
    return network.to(device)


def unpack_weights(parameters, args, w_dim, in_dim):
    hid = args.hid_width
    Lf = args.fuse_depth - 1
    W_tot = np.ones(w_dim)
    W = [param.data.cpu().detach().numpy() for param in parameters]
    if args.mode == "shallow":
        W_tot = W[0].squeeze()
        in_hid = W[0]
    elif args.mode == "early_fusion":
        W_tot = (W[1] @ W[0]).squeeze()
        in_hid = W[0]
    elif args.mode == "late_fusion":
        W_tot[:in_dim[0]] = W[-1][:, :hid] @ W[0]
        W_tot[in_dim[0]:] = W[-1][:, hid:] @ W[1]
        in_hid = np.concatenate((W[0], W[1]), -1)
    elif args.mode == "deep_fusion":
        if args.fuse_depth == 1:  # deep early fusion
            in_hid = W[0]
            W_tot = np.eye(w_dim)
            for i in range(len(W)):
                W_tot = W[i] @ W_tot
        else:  # deep late fusion
            in_hid = np.concatenate((W[0], W[1]), -1)
            h1, h2 = np.eye(in_dim[0]), np.eye(in_dim[1])
            for i in range(0, Lf):
                h1, h2 = W[2*i] @ h1, W[2*i+1] @ h2
            h = np.concatenate((W[2*Lf][:, :hid] @ h1, W[2*Lf][:, hid:] @ h2), -1)
            for i in range(2*Lf+1, len(W)):
                h = W[i] @ h
            W_tot = h
    elif args.mode == "fission":
        W_tot[0] = W[1] @ W[0][:hid, :] 
        W_tot[1] = W[2] @ W[0][hid:, :] 
        in_hid = W[0]
    return W_tot, in_hid


def loss_func(args, y, y_hat):
    if args.loss == "mse":
        return 0.5*nn.MSELoss()(y, y_hat)
    elif args.loss == "exponential":
        return torch.mean(torch.exp(-y * y_hat))
    elif args.loss == "logistic":
        return torch.mean(torch.log(1+torch.exp(-y * y_hat)))
    else:
        raise NotImplementedError


def train(data, args):
    if args.mode == "late_fusion" or args.mode == "deep_fusion":
        x1_tensor, x2_tensor, y_tensor, in_dim, out_dim = prep_data(args, data, device)
    else:
        x_tensor, y_tensor, in_dim, out_dim = prep_data(args, data, device)
    network = creat_network(args, in_dim, out_dim)
    optimizer = optim.SGD(network.parameters(), lr=args.lr, weight_decay=args.reg)
    
    w_dim = in_dim if isinstance(in_dim, int) else sum(in_dim)
    weights = np.zeros((args.epoch, w_dim)) if args.plot_weight else None
    results = {'Ls': np.zeros(args.epoch),
               'Eg': np.zeros(args.epoch),
               'W': weights}
    if args.vis_feat:
        fig, axs, ims = prep_axs(args)
    # Training loop
    for i in range(args.epoch):
        optimizer.zero_grad()
        if args.mode == "late_fusion" or args.mode == "deep_fusion":
            predictions= network(x1_tensor, x2_tensor)
        else:
            predictions = network(x_tensor)
        loss = loss_func(args, y_tensor, predictions)
        loss.backward()
        optimizer.step()
        results['Ls'][i] = loss.item()
        
        if args.plot_weight:
            results['W'][i, :], feat = unpack_weights(network.parameters(), args, w_dim, in_dim)
            if args.plot_Eg:
                results['Eg'][i] = (results['W'][i, :] - data['w_gt']) @ data['cov'] @ (results['W'][i, :] - data['w_gt']) / 2
            if args.vis_feat and i % 10 == 0:
                y_res = data.copy()
                y_res['y'] = data['y'] - predictions.cpu().detach().numpy()
                ims.append(vis_relu(args, y_res, feat, results['Ls'][:i], axs))
        if args.sweep not in ['single', 'depth_single'] and results['Ls'][i] < 10e-5:
            results['Ls'], results['W'] = results['Ls'][:i], weights[:i]
            print("Converged at epoch ", i)
            break

    if args.vis_feat:
        ani = animation.ArtistAnimation(fig, ims, interval=20, blit=False)
        plt.tight_layout()
        plt.show()
    if args.sweep in ['single', 'depth_single']:
        plot_training(args, data, results)
    return results
 

if __name__ == "__main__":
    args = config().parse_args()
    if args.sweep == 'single':
        data = gen_data(args)
        train(data, args)
    elif args.sweep == 'depth_single':
        depth_single(args)
    elif args.sweep == 'toy_sweep':
        toy_sweep(args)
    elif args.sweep == 'rho_sweep':
        rho_sweep(args)
    elif args.sweep == 'ratio_sweep':
        ratio_sweep(args)
    elif args.sweep == 'init_sweep':
        init_sweep(args)
    plt.show()