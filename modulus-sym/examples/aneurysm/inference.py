# code to carry out inference from trained PINN models for the aneurysm case

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from modulus.sym.utils.io.vtk import var_to_polyvtk
from modulus.sym.utils.io.vtk import grid_to_vtk
from sympy import Symbol, sqrt, Max

import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.key import Key
from modulus.sym.utils.io import csv_to_dict
from modulus.sym.geometry.tessellation import Tessellation
from pdb import set_trace

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# function to load trained model and carry out inference at specified points
def run_inference(cfg, checkpoint_dir, output_dir):
    point_path = to_absolute_path("./stl_files")
    interior_mesh = Tessellation.from_stl(
        point_path + "/aneurysm_closed.stl", airtight=True
    )

    center = (-18.40381048596882, -50.285383353981196, 12.848136936899031)
    scale = 0.4

    # Normalize meshes
    def normalize_mesh(mesh, center, scale):
        mesh = mesh.translate([-c for c in center])
        mesh = mesh.scale(scale)
        return mesh

    interior_mesh = normalize_mesh(interior_mesh, center, scale)

    # make list of nodes to unroll graph on
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )

    # nodes = [flow_net.make_node(name="flow_network")]
    print("device: ", device)
    # load checkpoint
    checkpoint = torch.load(os.path.join(checkpoint_dir, "optim_checkpoint.0.pth"), map_location=device)
    # checkpoint keys: dict_keys(['step', 'optimizer_state_dict', 'aggregator_state_dict', 'scheduler_state_dict', 'scaler_state_dict'])
    checkpoint_network = torch.load(os.path.join(checkpoint_dir, "flow_network.0.pth"), map_location=device)
    flow_net.load_state_dict(checkpoint_network)
    # flow_net.load_state_dict(checkpoint["model_state_dict"])
    print("Total number of trainable model parameters: ", sum(p.numel() for p in flow_net.parameters() if p.requires_grad))
    # Inference on the interior mesh points
    try:
        invar = interior_mesh.sample_interior(nr_points=10000)
    except Exception as e:
        print(f"An error occurred: {e}")

    # invar = interior_mesh.sample_interior(nr_points=10000, compute_sdf_derivatives=True)
    # invar = interior_mesh.sample_interior(10000)  # sample 10000 points for visualization
    invar = {k: torch.tensor(v, dtype=torch.float32) for k, v in invar.items()}

    # normalize invars
    def normalize_invar(invar, center, scale, dims=2):
        invar["x"] -= center[0]
        invar["y"] -= center[1]
        invar["z"] -= center[2]
        invar["x"] *= scale
        invar["y"] *= scale
        invar["z"] *= scale
        if "area" in invar.keys():
            invar["area"] *= scale ** dims
        return invar

    invar = normalize_invar(invar, center, scale, dims=3)

    # convert invars to tensors and run inference
    invar = {k: v.unsqueeze(1) for k, v in invar.items()}
    flow_net.eval()  # set the network to evaluation mode
    with torch.no_grad():
        outvar = flow_net(invar)
        # Prepare the variables for VTK
        save_var = {
            "u": outvar["u"].detach().cpu().numpy().reshape(-1, 1),
            "v": outvar["v"].detach().cpu().numpy().reshape(-1, 1),
            "w": outvar["w"].detach().cpu().numpy().reshape(-1, 1),
            "p": outvar["p"].detach().cpu().numpy().reshape(-1, 1),
            "x": invar["x"].detach().cpu().numpy().reshape(-1, 1),
            "y": invar["y"].detach().cpu().numpy().reshape(-1, 1),
            "z": invar["z"].detach().cpu().numpy().reshape(-1, 1)
        }
        # https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/features/post_processing.html#var-to-polyvtk
        var_to_polyvtk(save_var, os.path.join(output_dir, "field_tessellated_interior"))

    set_trace()
    # visualize the flow fields
    x, y, z = invar["x"].squeeze().numpy(), invar["y"].squeeze().numpy(), invar["z"].squeeze().numpy()
    u, v, w = outvar["u"].detach().numpy(), outvar["v"].detach().numpy(), outvar["w"].detach().numpy()
    p = outvar["p"].detach().numpy()

    set_trace()
    fig = plt.figure(figsize=(12, 6))

    # velocity vector quiver plot
    ax = fig.add_subplot(121, projection='3d')
    ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)
    ax.set_title("Velocity Field")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # pressure field
    ax = fig.add_subplot(122, projection='3d')
    img = ax.scatter(x, y, z, c=p, cmap='viridis')
    fig.colorbar(img)
    ax.set_title("Pressure Field")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()

    fig.savefig(os.path.join(output_dir, "flow_fields.png"))


@modulus.sym.main(config_path="conf", config_name="config")
def main(cfg: ModulusConfig) -> None:
    # set checkpoint directory and output directory
    checkpoint_dir = to_absolute_path("./outputs/aneurysm/")
    output_dir = to_absolute_path("./postprocess_output")

    os.makedirs(output_dir, exist_ok=True)

    # run inference
    run_inference(cfg, checkpoint_dir, output_dir)

# run the inference function
if __name__ == "__main__":
    main()

