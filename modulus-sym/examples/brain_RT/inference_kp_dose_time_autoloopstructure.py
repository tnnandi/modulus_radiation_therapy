import os
import torch
import numpy as np
from sympy import Symbol, sqrt, Max, Eq
import matplotlib.pyplot as plt
from modulus.sym.utils.io.vtk import var_to_polyvtk
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.key import Key
from modulus.sym.geometry.tessellation import Tessellation
import modulus.sym
from moving_time_window import MovingTimeWindowArch

# execute within a modulus container on polaris
# cd /mnt/Modulus_24p04/modulus-sym/examples/brain_RT

# after execution of this, use animate_kp_files_C.py (with paraview python shell) to generate png files from Paraview

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# function to load trained model and carry out inference at specified points
def run_inference(cfg,
                  # vary_time=False,
                  # vary_dose=False,
                  # vary_kp=False,
                  inference_times=None,
                  inference_doses=None,
                  inference_kps=None,
                  checkpoint_dir=None,
                  output_dir=None):

    # if sum([vary_time, vary_dose, vary_kp]) != 1:
    #     raise ValueError("Please specify exactly one parameter to vary: time, dose, or kp.")

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
    cfg.arch.fully_connected.layer_size = 64 #128
    cfg.arch.fully_connected.nr_layers = 4

    tumor_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z"), Key("t"), Key("kp"), Key("dose")],
        # add "kp" and "t" as additional input (parameterized)
        output_keys=[Key("N")],
        cfg=cfg.arch.fully_connected,
    )

    # moving window approach for time evolution over larger durations
    # parameterized time
    time_window_size = 1.0
    alpha_value = 0.035
    alpha_by_beta_value = 10
    t_treatment = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19] # using zero-based indexing

    # Iterate over the checkpoint directories for each time window
    if len(inference_times) == 1:
        # for a fixed time window
        fixed_time_index = int(inference_times[0])  # Ensure time index is an integer (and not a list)
        time_window_dirs = [
            d for d in os.listdir(checkpoint_dir)
            if d.startswith(f"time_window_{fixed_time_index:04d}")
        ]
    # elif len(inference_times) > 1:
    #     # For a specific list of time windows
    #     chosen_time_indices = [int(t) for t in inference_times]  # Ensure all indices are integers
    #     time_window_dirs = [
    #         d for d in os.listdir(checkpoint_dir)
    #         if any(d.startswith(f"time_window_{idx:04d}") for idx in chosen_time_indices)
    #     ]
    else:
        # For all time windows
        time_window_dirs = [
            d for d in os.listdir(checkpoint_dir)
            if d.startswith("time_window_")
        ]

    for time_window_dir in time_window_dirs:
        print("Loading ", time_window_dir)
        # Extract the actual time index from the directory name
        time_idx = int(time_window_dir.split("_")[-1])  # Get the time index from the directory name
        checkpoint_path = os.path.join(checkpoint_dir, time_window_dir, "time_window_network.0.pth")
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            continue

        time_window_net = MovingTimeWindowArch(tumor_net,
                                               time_window_size,
                                               alpha=alpha_value,
                                               alpha_by_beta=alpha_by_beta_value,
                                               t_treatment=t_treatment)
        time_window_net.to(device)
        time_window_net.load_state_dict(torch.load(checkpoint_path, map_location=device))
        time_window_net.eval()
        print(f"Loaded model for time window {time_idx} from {checkpoint_path}")

        # Sample interior points for inference
        interior_points = interior_mesh.sample_interior(1000000)
        invar = {k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in interior_points.items()}

        # Set the fixed time value for the current time window
        fixed_time = time_idx * time_window_size
        invar["t"] = torch.full_like(invar["x"], fixed_time)

        # Run inference for varying dose and kp
        for kp in inference_kps:
            invar["kp"] = torch.full_like(invar["x"], kp)
            for dose in inference_doses:
                invar["dose"] = torch.full_like(invar["x"], dose)

                print(f"Inference for time: {fixed_time}, kp: {kp}, dose: {dose}")

                # Perform inference
                with torch.no_grad():
                    outvar = time_window_net(invar)
                    save_var = {
                        "N": outvar["N"].detach().cpu().numpy(),
                        "x": invar["x"].detach().cpu().numpy(),
                        "y": invar["y"].detach().cpu().numpy(),
                        "z": invar["z"].detach().cpu().numpy(),
                    }

                    # Save outputs in VTK format
                    # output_filename = f"field_tessellated_interior_time_{time_idx}_kp_{kp}_dose_{dose}.vtk"

                    # Dynamically construct the filename with varying quantity at the end
                    if len(inference_times) > 1:
                        # Time is varying
                        output_filename = f"field_tessellated_interior_kp_{kp}_dose_{dose}_time_{time_idx}.vtk"
                    elif len(inference_doses) > 1:
                        # Dose is varying
                        output_filename = f"field_tessellated_interior_kp_{kp}_time_{time_idx}_dose_{dose}.vtk"
                    elif len(inference_kps) > 1:
                        # kp is varying
                        output_filename = f"field_tessellated_interior_dose_{dose}_time_{time_idx}_kp_{kp}.vtk"
                    else:
                        # Default to time varying
                        output_filename = f"field_tessellated_interior_time_{time_idx}_kp_{kp}_dose_{dose}.vtk"

                    output_filepath = os.path.join(output_dir, output_filename)
                    print(f"Saving to: {output_filepath}")
                    var_to_polyvtk(save_var, output_filepath)

@modulus.sym.main(config_path="conf", config_name="config")
def main(cfg: ModulusConfig) -> None:
    # Directory for checkpoints and output
    checkpoint_dir = to_absolute_path(
        "./outputs/brain_param_kp_dose_time.windowing/brain_param_kp_dose_time.lambda_1_100_100_1000.layer_size_64_test/"
    )
    # output_dir = to_absolute_path("./postprocess_output.vary_kp_dose_fixed_time")
    # output_dir = to_absolute_path("./postprocess_output.vary_dose_time_3_kp_0p05")
    # output_dir = to_absolute_path("./postprocess_output.vary_time_dose_1_kp_0p05")

    # Inference parameters
    # inference_times = [3, 6] #
    inference_times = [i for i in range(20)]  # Time indices (fixed by outer loop)
    inference_doses = [4] #[i for i in range(9)]  # Varying dose
    inference_kps = [0.05] #[0.05, 0.1, 0.2]  # Varying kp

    # Determine the varying quantity and construct the directory name dynamically
    if len(inference_times) > 1:
        # Time is varying
        varying_param = "vary_time"
        fixed_params = f"dose_{inference_doses[0]}_kp_{inference_kps[0]}"
    elif len(inference_doses) > 1:
        # Dose is varying
        varying_param = "vary_dose"
        fixed_params = f"time_{inference_times[0]}_kp_{inference_kps[0]}"
    elif len(inference_kps) > 1:
        # kp is varying
        varying_param = "vary_kp"
        fixed_params = f"time_{inference_times[0]}_dose_{inference_doses[0]}"
    else:
        # Default to time varying if all are fixed (unlikely)
        varying_param = "vary_time"
        fixed_params = f"dose_{inference_doses[0]}_kp_{inference_kps[0]}"

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    output_dir = to_absolute_path(f"./postprocess_output.{varying_param}_{fixed_params}_{timestamp}")

    print(f"Output directory: {output_dir}")

    # Run inference
    run_inference(
        cfg,
        inference_times=inference_times,
        inference_doses=inference_doses,
        inference_kps=inference_kps,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()