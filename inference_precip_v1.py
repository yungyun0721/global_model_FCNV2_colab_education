import os
import numpy as np
from modules.afnonet import  unlog_tp_torch  
from modules.inference_helper import nan_extend, normalise, load_precip_model, load_statistics
import torch
import argparse


# setting
# weight_path_global = 'FCNV1_precip_v1/weight_precip'

# Input
area = [90, 0, -90, 360 - 0.25]
grid = [0.25, 0.25]
# u10, v10, t2m, sp, mslp, 
# t850, u1000, v1000, z1000,  u850,
# v850, z850, u500, v500, z500, 
# t500, z50, r500, r850, tcwvv 7

vars_index = [ 0,  1,  4,  5,  6, 57, 20, 33, 46, 18,
              31, 44, 15, 28, 41, 54, 34, 67, 70, 7 ]

def FCNV1_precip(FCNV2_output_folder, save_path, 
                 FCNV1_precip_weight='FCNV1_precip_v1/weight_precip',device="cuda"):
    # load model and parameters
    global_means, global_stds = load_statistics(FCNV1_precip_weight, channels = 20 )
    precip_ckpt = os.path.join(FCNV1_precip_weight, "precip.ckpt")
    files = os.listdir(FCNV2_output_folder)
    files.sort()
    
    precip_model = load_precip_model(precip_ckpt,device=device)
    
    if not os.path.isdir(f'{save_path}'):
        os.mkdir(f'{save_path}')

    for time_index in range(len(files)):
        weather_data = np.load(f'{FCNV2_output_folder}/output_weather_{time_index*6:0>3}h.npy').astype(np.float32)
        for_precip_data = weather_data[vars_index,...]
        all_fields_numpy = for_precip_data[np.newaxis, :20, :-1, :]
        all_fields_numpy = normalise(all_fields_numpy,global_means,global_stds)

        # Run the inference precipitation session
        input_iter = torch.from_numpy(all_fields_numpy).to(device)
        torch.set_grad_enabled(False)
        precip_output = precip_model(input_iter)
        precip_output = nan_extend(unlog_tp_torch(precip_output.cpu()).numpy())
        
        np.save(os.path.join(save_path, f'output_precipitation_{(time_index)*6:0>3}h'), precip_output)
        if np.mod(time_index,4)==3:
            print(f'finish {int(time_index/4)+1} days')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',"--FCNV2_path", required=True, help="FCNV2 output folder")
    parser.add_argument('-s',"--save_path", required=True, help="output_folder (folder)") 
    parser.add_argument('-w',"--FCNV1_precip_weight", help="path to FCNV1_precip weight folder", default='FCNV1_precip_v1/weight_precip')
    parser.add_argument('-d',"--device", help="cpu or cuda", default='cpu')
    args = parser.parse_args()
    FCNV1_precip(args.FCNV2_path, args.save_path, FCNV1_precip_weight=args.FCNV1_precip_weight, device=args.device) 
