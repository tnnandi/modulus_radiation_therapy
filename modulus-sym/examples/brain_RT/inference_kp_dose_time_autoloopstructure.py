import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from modulus.sym.utils.io.vtk import var_to_polyvtk
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.key import Key
from modulus.sym.geometry.tessellation import Tessellation
import modulus.sym

# execute within a modulus container on polaris
# cd /mnt/Modulus_24p04/modulus-sym/examples/brain_RT

# after execution of this, use animate_kp_files_C.py (with paraview python shell) to generate png files from Paraview

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# function to load trained model and carry out inference at specified points
def run_inference(cfg,
                  vary_time=False,
                  vary_dose=False,
                  vary_kp=False,
                  inference_times=None,
                  inference_doses=None,
                  inference_kps=None,
                  checkpoint_dir=None,
                  output_dir=None):

    if sum([vary_time, vary_dose, vary_kp]) != 1:
        raise ValueError("Please specify exactly one parameter to vary: time, dose, or kp.")

    if inference_times is None or inference_doses is None or inference_kps is None:
        raise ValueError("Please supply inference values for time, dose, and kp.")

    if checkpoint_dir is None or output_dir is None:
        raise ValueError("Please supply checkpoint_dir and output_dir.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    point_path = to_absolute_path("./stl_files")
    interior_mesh = Tessellation.from_stl(
        point_path + "/brain_volume_subsample_4X.stl", airtight=True
    )

    scale = 1  # keep dimensions in mm
    interior_mesh.scale(scale)

    # override defaults
    cfg.arch.fully_connected.layer_size = 128
    cfg.arch.fully_connected.nr_layers = 4

    tumor_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z"), Key("t"), Key("kp"), Key("dose")],
        output_keys=[Key("N")],
        cfg=cfg.arch.fully_connected,
    )

    tumor_net.to(device)
    tumor_net.eval()

    # load checkpoint
    checkpoint_network = torch.load(os.path.join(checkpoint_dir, "flow_network.0.pth"), map_location=device)
    tumor_net.load_state_dict(checkpoint_network)
    print(f"Model loaded with {sum(p.numel() for p in tumor_net.parameters() if p.requires_grad)} trainable parameters")

    # iterate over time and kp values for inference
    interior_points = interior_mesh.sample_interior(1000000)
    invar = {k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in interior_points.items()}

    indices = {
        "time": (inference_times, 't'),
        "dose": (inference_doses, 'dose'),
        "kp": (inference_kps, 'kp')
    }

    if vary_time:
        outer_loop, middle_loop, inner_loop = indices["dose"], indices["kp"], indices["time"]
    elif vary_dose:
        outer_loop, middle_loop, inner_loop = indices["kp"], indices["time"], indices["dose"]
    elif vary_kp:
        outer_loop, middle_loop, inner_loop = indices["dose"], indices["time"], indices["kp"]

    # run inference
    for outer_value in outer_loop[0]:
        for middle_value in middle_loop[0]:
            for inner_value in inner_loop[0]:
                invar[outer_loop[1]] = torch.full_like(invar['x'], outer_value)
                invar[middle_loop[1]] = torch.full_like(invar['x'], middle_value)
                invar[inner_loop[1]] = torch.full_like(invar['x'], inner_value)

                print(
                    f"Inference for {outer_loop[1]}: {outer_value}, {middle_loop[1]}: {middle_value}, {inner_loop[1]}: {inner_value}")

                # inference
                with torch.no_grad():
                    outvar = tumor_net(invar)
                    save_var = {
                        "N": outvar["N"].detach().cpu().numpy(),
                        "x": invar["x"].detach().cpu().numpy(),
                        "y": invar["y"].detach().cpu().numpy(),
                        "z": invar["z"].detach().cpu().numpy(),
                    }

                    # save outputs in VTK format
                    output_filename = f"field_tessellated_interior_{outer_loop[1]}_{outer_value}_{middle_loop[1]}_{middle_value}_{inner_loop[1]}_{inner_value}.vtk"
                    output_filepath = os.path.join(output_dir, output_filename)
                    print(f"Saving to: {output_filename}")
                    var_to_polyvtk(save_var, output_filepath)

@modulus.sym.main(config_path="conf", config_name="config")
def main(cfg: ModulusConfig) -> None:
    # edit output_dir based on the parameters chosen
    checkpoint_dir = to_absolute_path(
        "./outputs/brain_param_kp_dose_time/brain_param_kp_dose_time.lambda_1_100_50.heaviside_source_plus/")
    # output_dir = to_absolute_path("./postprocess_output.vary_t_kp_0p1_dose_8")
    output_dir = to_absolute_path("./postprocess_output.vary_dose_t_0_kp_0p2")
    # output_dir = to_absolute_path("./postprocess_output.vary_kp_t_10_dose_2")
    os.makedirs(output_dir, exist_ok=True)

    # # time and kp values for inference
    # # inference_times = [0.0, 5.0, 10.0, 15.0]
    # # inference_times = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    #
    # inference_times = [i for i in range(20)]
    # inference_kps = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    # inference_doses = [i for i in range(11)]
    #
    # run_inference(cfg, checkpoint_dir, output_dir, inference_times, inference_kps, inference_doses)

    # inference_times = [i for i in range(20)]
    # inference_kps = [0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    inference_doses = [i for i in range(11)]

    # Example usage with external values
    inference_times = [0]
    # inference_doses = [8]
    inference_kps = [0.2]

    # choose True for either vary_time/dose/k_p
    run_inference(
        cfg,
        vary_time=False,
        vary_dose=True,
        vary_kp=False,
        inference_times=inference_times,
        inference_doses=inference_doses,
        inference_kps=inference_kps,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()
