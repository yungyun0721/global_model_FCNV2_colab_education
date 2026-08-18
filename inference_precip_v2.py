import numpy as np
import xarray
import torch
import tarfile, os
from datetime import datetime
from FCNV1_precip_v2.inference_helper import compute_sza,rh_to_q
from FCNV1_precip_v2.precip_v2 import PrecipNet_v2
import argparse

# Open tar file and extract contents to temporary directory
precip_weight = 'FCNV1_precip_v2/AFNOprecip_weight'
cached_file_name = f"{precip_weight}/afno_precip.mdlus"
local_path = f"{precip_weight}/model_weight"
@staticmethod
def safe_members(tar, local_path):
    for member in tar.getmembers():
        if (
            ".." in member.name
            or os.path.isabs(member.name)
            or os.path.realpath(os.path.join(local_path, member.name)).startswith(
                os.path.realpath(local_path)
            )
        ):
            yield member
        else:
            print(f"Skipping potentially malicious file: {member.name}")
            
def unzip_weight():
    with tarfile.open(cached_file_name, "r") as tar:
        # Safely extract while supporting Python < 3.12
        extract_kwargs = dict(
            path=local_path,
            members=list(safe_members(tar, local_path)),
        )
        
        if "filter" in tar.extractall.__code__.co_varnames:
            extract_kwargs["filter"] = "data"
        tar.extractall(**extract_kwargs)  # noqa: S202

# prepare data
variables = [
    "u10m",  "v10m",  "t2m", "sp", "msl", "tcwv",
    "u500", "u850", "u1000", 
    "v500", "v850", "v1000",
    "z50",  "z500",  "z850", "z1000",
    "t500", "t850",
    "q500", "q850",
]

ordering = [ "10u",  "10v", "100u", "100v",  "2t",  "sp",  "msl", "tcwv",
        "u50",  "u100", "u150", "u200", "u250", "u300", "u400", "u500", "u600", "u700", "u850", "u925", "u1000",
        "v50",  "v100", "v150", "v200", "v250", "v300", "v400", "v500", "v600", "v700", "v850", "v925", "v1000",
        "z50",  "z100", "z150", "z200", "z250", "z300", "z400", "z500", "z600", "z700", "z850", "z925", "z1000",
        "t50",  "t100", "t150", "t200", "t250", "t300", "t400", "t500", "t600", "t700", "t850", "t925", "t1000",
        "r50",  "r100", "r150", "r200", "r250", "r300", "r400", "r500", "r600", "r700", "r850", "r925", "r1000"]

vars_index = [ 0,  1,  4,  5,  6, 7, 
              15, 18, 20, 28, 31, 33, 
              34, 41, 44, 46, 54, 57, 67, 70]

lat = np.linspace(90, -90, 720, endpoint=False)
lon = np.linspace(0, 360, 1440, endpoint=False)
grid_x, grid_y = torch.meshgrid(
    torch.tensor(lat), torch.tensor(lon)
)
# grid_y, grid_x = np.meshgrid(
#     lat, lon, indexing='ij'
# )


def FCNV2_to_precip_v2(input_folder, IC_time, save_folder,device='cpu'):
    '''
    input_folder: the folder came from FCNV2 (ex: /wk2/yungyun/code_space/FCNV2_test/output_data_2023072400)
    IC_time: the initial time get from initial data (ex: 2023072400)
    save_folder: the folder for saving (ex: prceip_data)
    '''
    if not os.path.exists(local_path):   
        unzip_weight()
    # load model
    model_dict = torch.load(f"{local_path}/model.pt", map_location="cpu")
    # model precipnet model
    model = PrecipNet_v2(
        inp_shape=[720, 1440],
        patch_size=[8, 8],
        in_channels=23,
        out_channels=1,
        embed_dim=768,
        depth=12,
        num_blocks=8,
        mlp_ratio=8,   
    )
    # combine model
    model_dict.pop('device_buffer', None)
    model_dict.pop('backbone.device_buffer', None)
    model.load_state_dict(model_dict,strict=True)

    # prepare data
    files = os.listdir(input_folder)
    os.makedirs(save_folder, exist_ok=True)
    TC_time = datetime.strptime(str(IC_time),"%Y%m%d%H").strftime("%Y-%m-%dT%H")
    IC_time = np.datetime64(f'{TC_time}:00:00','s')
    
    # model
    input_center = np.load(f"{precip_weight}/global_means.npy")
    input_scale = np.load(f"{precip_weight}/global_stds.npy")
    lsm = xarray.open_dataset(f"{precip_weight}/land_sea_mask.nc")["LSM"].values[ :, :-1]
    orography = xarray.open_dataset(f"{precip_weight}/orography.nc")["Z"].values[ :, :-1]
    orography = (orography - orography.mean()) / orography.std()
    
    # run  precip. model 
    for i in range(len(files)):
    # for i in range(2):
        # data = np.load(os.path.join(input_folder,files[i]))
        data = np.load(os.path.join(input_folder,f"output_weather_{i*6:0>3}h.npy"))
        # print(data.shape)
        precip_data = data[vars_index,:-1,:]
        precip_data[18:20,:,:] = rh_to_q(precip_data[18:20,:,:],precip_data[16:18,:,:],[500,850])
        precip_data = (precip_data-input_center)/input_scale
        time_data = compute_sza(grid_x, grid_y, IC_time, np.timedelta64(i*6, "h"))[np.newaxis,...]
        precip_data = np.concatenate([precip_data, orography, lsm, time_data],axis=0)[np.newaxis,...]
        precip_data = torch.Tensor(precip_data)
        
        precip_data = precip_data.to(device)
        model = model.to(device)
        model.eval()
        data = model(precip_data)
        out = 1e-5 * (torch.exp(data) - 1)
        # convert from mm to m
        out = out.cpu().detach().numpy() / 1000.0
        out[out < 0] = 0
        np.save(os.path.join(save_folder, f'output_precipitation_{(i)*6:0>3}h'),out.squeeze())
        print(f'finishing  output_precipitation_{(i)*6:0>3}h')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--FCNV2_path", required=True, help="input_folder for FCNV1_precip_v2")
    parser.add_argument("--IC_time", required=True, help="2023072400")
    parser.add_argument("--save_path", required=True, help="save_folder (folder)")
    parser.add_argument("--device", help="cpu or cuda", default='cuda')
    args = parser.parse_args()
    FCNV2_to_precip_v2(args.FCNV2_path, args.IC_time, args.save_path, device=args.device)