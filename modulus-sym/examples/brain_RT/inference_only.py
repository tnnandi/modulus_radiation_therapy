import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from modulus.sym.utils.io.vtk import var_to_polyvtk
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.key import Key
from modulus.sym.geometry.tessellation import Tessellation
import modulus.sym
# cd /mnt/Modulus_24p04/modulus-sym/examples/brain_RT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# function to load trained model and carry out inference at specified points
def run_inference(cfg, checkpoint_dir, output_dir, inference_times, inference_kps):
    point_path = to_absolute_path("./stl_files")
    interior_mesh = Tessellation.from_stl(
        point_path + "/brain_volume_subsample_4X.stl", airtight=True
    )

    scale = 1  # keep dimensions in mm
    interior_mesh.scale(scale)

    # override defaults
    cfg.arch.fully_connected.layer_size = 128
    cfg.arch.fully_connected.nr_layers = 4

    # Create the architecture for inference
    tumor_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z"), Key("t"), Key("kp")],
        output_keys=[Key("N")],
        cfg=cfg.arch.fully_connected,
    )

    tumor_net.to(device)
    tumor_net.eval()  # set model to evaluation mode

    # load checkpoint
    checkpoint_network = torch.load(os.path.join(checkpoint_dir, "flow_network.0.pth"), map_location=device)
    tumor_net.load_state_dict(checkpoint_network)
    print(f"Model loaded with {sum(p.numel() for p in tumor_net.parameters() if p.requires_grad)} trainable parameters")

    # iterate over time and kp values for inference
    interior_points = interior_mesh.sample_interior(100000)
    invar = {k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in interior_points.items()}

    time_index = 0
    for time_value in inference_times:
        kp_index = 0
        for kp_value in inference_kps:
            print(f"Inference for time: {time_value}, kp: {kp_value}")

            # set parameter values
            invar['t'] = torch.full_like(invar['x'], time_value)
            invar['kp'] = torch.full_like(invar['x'], kp_value)

            # perform inference
            with torch.no_grad():
                outvar = tumor_net(invar)
                save_var = {
                    "N": outvar["N"].detach().cpu().numpy(),
                    "x": invar["x"].detach().cpu().numpy(),
                    "y": invar["y"].detach().cpu().numpy(),
                    "z": invar["z"].detach().cpu().numpy(),
                }

                # save the results in VTK format
                output_filename = f"field_tessellated_interior_t_{time_index}_kp_{kp_index}.vtk"
                var_to_polyvtk(save_var, os.path.join(output_dir, output_filename))
            kp_index += 1
        time_index += 1

# main function to load configuration and run inference
@modulus.sym.main(config_path="conf", config_name="config")
def main(cfg: ModulusConfig) -> None:
    checkpoint_dir = to_absolute_path("./outputs/brain_param_kp_time.lambda_1_100_50/")
    output_dir = to_absolute_path("./postprocess_output")
    os.makedirs(output_dir, exist_ok=True)

    # time and kp values for inference
    inference_times = [0.0, 5.0, 10.0, 15.0]
    inference_kps = [0.02, 0.05, 0.3, 1.0]

    # Run the inference
    run_inference(cfg, checkpoint_dir, output_dir, inference_times, inference_kps)

if __name__ == "__main__":
    main()
