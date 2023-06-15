'''
This code provides a runtime comparison of volume rendering methods traditionally used in NeRF applications to baking in geometry in BakedSDF.

Performance is measured in memory usage (KB)(size of model for 'NeRF' vs size of vertex array for BakedSDF) and ms (from model to mesh (NeRF) vs vertices to mesh (Baked))

***There is NO NeRFing in this code as the BakedSDF paper focuses more on the baking process. Most NeRFing in Baked is a combination of preceding SOTA methods so it is not covered.

There is an implicit representation of a sample 2D SDF, however, which serves as a base to both methods and simulates an SDF that could be extracted from a full BakedSDF.
'''
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from marching_squares import marching_squares

# measuring utilities
import time
from sys import getsizeof

res = 201

device = torch.device("mps")
torch.set_default_device(device)
class DenseGrid(nn.Module):

    def __init__(self):
        super().__init__()
        self.feat_dim = 1  # feature dim size
        self.codebook = nn.ParameterList([])

        self.LOD = [50]

        self.init_feature_structure()

    def init_feature_structure(self):
        for LOD in self.LOD:

            fts = torch.zeros(LOD**2, self.feat_dim)
            fts += torch.randn_like(fts)*0.1
            fts = nn.Parameter(fts)

            self.codebook.append(fts)
    
    def forward(self, pts):
        feats = []

        for i, res in enumerate(self.LOD):

            x = pts[:, 0] * (res - 1)
            x = torch.floor(x).int()

            y = pts[:, 1] * (res - 1)
            y = torch.floor(y).int()

            features = self.codebook[i][(x + y * res).long()]

        feats.append((torch.unsqueeze(features, dim=-1)))
        all_features = torch.cat(feats, -1)
        return all_features.sum(-1)


class SimpleModel(nn.Module):
    def __init__(
        self, grid_structure, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.module_list = torch.nn.ModuleList()
        self.module_list.append(torch.nn.Linear(input_dim,hidden_dim,bias=True))
        self.module_list.append(torch.nn.ReLU(inplace=True))
        self.module_list.append(torch.nn.Linear(hidden_dim,output_dim,bias=True))

        self.model = torch.nn.Sequential(*self.module_list)
        self.grid_structure = grid_structure

    def forward(self, coords):
        h,w,c = coords.shape

        coords = torch.reshape(coords,(h*w,c))

        feat = self.grid_structure(coords)

        out = self.model(feat)

        l,c = out.shape

        out = torch.reshape(out,(h,w,c))

        return out

def main():
    # Training the network
    max_epochs = 300

    learning_rate = 1.0e-2

    # defining the dragon SDF
    X,Y = torch.meshgrid(torch.linspace(0,1,res),
                        torch.linspace(0,1,res),
                        indexing="xy")
    
    coords = torch.stack([X,Y],dim=-1)
    
    sdf = torch.tensor(np.load('data/marching_squares.npy'),dtype=torch.float32)

    smart_grid = DenseGrid()
    model = SimpleModel(smart_grid, 1, 16, 1)
    loss_fn = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loop = tqdm(range(max_epochs))
    for epoch in loop:

        output = model(coords)

        output = torch.reshape(output,(res,res))
        
        loss = loss_fn(output, sdf)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch: {epoch}")
        loop.set_postfix_str(f"Loss: {loss.item():.5f}")

        # visualize(output,values)

    output = np.array(output.tolist())

    # visualizing SDF
    # thresh = np.maximum(np.abs(output.min()), output.max())
    # plt.imshow(output, cmap="seismic", norm = mpl.colors.TwoSlopeNorm(vmin=-thresh, vcenter=0., vmax=thresh),
    #      origin="lower")
    # plt.show()

    # baking the vertices
    verts = np.array(marching_squares(output,2))

    plot,(ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Calculate the elapsed time for querying the vert array
    baked_time = time.time()

    ax2.plot(verts[:,:,0], verts[:,:,1], 'r.', markersize=7) #plotting points

    baked_elapsed_time = time.time() - baked_time

    '''
    calculate the time it takes to:
    - construct a coordinate grid
    - feed into the model to query
    - run marching squares
    '''
    non_baked_time = time.time()

    #constructing the grid
    X,Y = torch.meshgrid(torch.linspace(0,1,res),
                        torch.linspace(0,1,res),
                        indexing="xy")
    # grid_coords = torch.stack((X,Y),axis=-1)
    coords = torch.stack([X,Y],dim=-1)

    # querying MLP
    output = model(coords)

    output = np.array(output.tolist())

    verts2 = np.array(marching_squares(output,2))

    # plot in the same way
    ax1.plot(verts2[:,:,0], verts2[:,:,1], 'r.', markersize=7) #plotting points

    non_baked_elapsed_time = time.time() - non_baked_time

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024

    ax1.axis('off')
    ax2.axis('off')
    ax1.set_title(baked_elapsed_time*1e3)
    ax1.set_title(f"MLP Querying\nTime to render: {non_baked_elapsed_time*1e3:.3f} ms\nMemory overhead: {size_all_mb:.3f}KB")
    ax2.set_title(f"BakedSDF\nTime to render: {baked_elapsed_time*1e3:.3f} ms\nMemory overhead: {getsizeof(verts)/1024:.3f}KB")
    plt.show()

if __name__ == "__main__":
    main()
