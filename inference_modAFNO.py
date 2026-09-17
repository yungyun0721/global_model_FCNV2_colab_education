import argparse
import numpy as np
import xarray
import torch
import tarfile, os
from datetime import datetime, timezone
from modAFNO_model.inference_helper import cos_zenith, rh_to_q
from modAFNO_model.modafno import ModAFNO

## Open tar file and extract contents to temporary directory
def main(input_folder, save_folder, IC_time,
         modAFNO_weight="modAFNO_weight",modAFNO_device='cuda'):    

    cached_file_name = f"{modAFNO_weight}/fcinterp-modafno-2x2.mdlus"
    local_path = f"{modAFNO_weight}/model_weight"
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

    if not os.path.exists(local_path):                  
        with tarfile.open(cached_file_name, "r") as tar:
            # Safely extract while supporting Python < 3.12
            extract_kwargs = dict(
                path=local_path,
                members=list(safe_members(tar, local_path)),
            )
            
            if "filter" in tar.extractall.__code__.co_varnames:
                extract_kwargs["filter"] = "data"
            tar.extractall(**extract_kwargs)  # noqa: S202

    #%% load model
    # load model checkpoint
    model_dict = torch.load(
        f"{local_path}/model.pt", map_location="cpu"
    )

    # model precipnet model
    model = ModAFNO(
        inp_shape=[720, 1440],
        in_channels=155,
        out_channels=73,
        embed_model={"dim": 64, 
                    "depth": 1, 
                    "method": "sinusoidal"},
        patch_size=(2, 2),
        embed_dim=512,
        mod_dim=64,
        modulate_filter=True,
        modulate_mlp=True,
        scale_shift_mode="complex",
        depth=12,
        num_blocks=1,
        mlp_ratio=2,   
        drop_rate=0.0,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0
    )

    # combine model
    model_dict.pop('device_buffer', None)
    model_dict.pop('backbone.device_buffer', None)
    model.load_state_dict(model_dict,strict=True)


    #%% 
    ## prepare data
    lat = np.linspace(90, -90, 720, endpoint=False)
    lon = np.linspace(0, 360, 1440, endpoint=False)

    grid_y, grid_x = np.meshgrid(lat, lon, indexing="ij")
    sincos_latlon = np.stack([np.sin(grid_y), np.cos(grid_y), np.sin(grid_x), np.cos(grid_x)], axis=0)

    # setting
    # input_folder = '/wk2/yungyun/code_space/FCNV2_test/output_data_2023072400'
    files = os.listdir(input_folder)
    # for i in range(len(files)):
    # IC_time = "2025072400"
    # modAFNO_device = 'cpu'
    save_folder = 'test_interpolation_IFS'
    os.makedirs(save_folder, exist_ok=True)

    IC_time = datetime.strptime(IC_time,"%Y%m%d%H").strftime("%Y-%m-%dT%H")
    IC_time = np.datetime64(f'{IC_time}:00:00','s')
    # model
    input_center = np.load(f"{modAFNO_weight}/global_means.npy")[0,:73,:,:]
    input_scale = np.load(f"{modAFNO_weight}/global_stds.npy")[0,:73,:,:]
    lsm = xarray.open_dataset(f"{modAFNO_weight}/land_sea_mask.nc")["LSM"].values[ :, :-1]
    orography = xarray.open_dataset(f"{modAFNO_weight}/orography.nc")["Z"].values[ :, :-1]
    orography = (orography - orography.mean()) / orography.std()
    pressure_level = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    # intro_data = np.full([len(files),155,720,1440],0.0)
    static_data = np.concatenate([sincos_latlon,orography,lsm], axis=0)
    start_i = 0
    for i in range(len(files)):
        # data1 = np.load(os.path.join(input_folder,files[i]))
        data1 = np.load(os.path.join(input_folder,f'output_weather_{((start_i+i)*6):0>3}h.npy'))
        x1_data = data1[:,:-1,:].copy()
        x1_data[60:73,:,:] = rh_to_q(data1[60:,:-1,:],data1[47:60,:-1,:],pressure_level)
        x1_data[  :73,:,:] = (x1_data[:73,:,:]-input_center)/input_scale
        
        data2 = np.load(os.path.join(input_folder,f'output_weather_{((start_i+i+1)*6):0>3}h.npy'))
        x2_data = data2[:,:-1,:].copy()
        x2_data[60:73,:,:] = rh_to_q(data2[60:,:-1,:],data2[47:60,:-1,:],pressure_level)
        x2_data[:73,:,:]   = (x2_data[:73,:,:]-input_center)/input_scale
        inter_data = np.concatenate([x1_data, x2_data],axis=0)
        
        for intro_i in range(6):
            target_time = (start_i+i)*6+intro_i
            time_data = cos_zenith([IC_time+np.timedelta64((start_i+i)*6, "h"), IC_time+np.timedelta64((start_i+i+1)*6, "h"), IC_time+np.timedelta64((start_i+i)*6+intro_i, "h")])
            total_data = np.concatenate([inter_data,time_data,static_data], axis=0)
            total_data = torch.Tensor(total_data[np.newaxis,...])
            t_norm = torch.Tensor([intro_i / 6])
            total_data = total_data.to(modAFNO_device)
            t_norm = t_norm.to(modAFNO_device)
            model = model.to(modAFNO_device)
            model.eval()
            print(f'start predict {target_time:0>3}h')

            with torch.inference_mode():
                if modAFNO_device == "cuda":
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16
                    ):
                        out = model(total_data, t_norm)
                else:
                    out = model(total_data, t_norm)
            # out = model(total_data, t_norm)
            # backto CPU
            out = out.float().cpu().numpy()
            del total_data, t_norm
            out *= input_scale
            out += input_center    
            np.save(os.path.join(save_folder, f'output_weather_{target_time:0>3}h'),out.squeeze())
        # intro_data = torch.Tensor(intro_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Define arguments to get YAML config file.
    parser.add_argument('-i','--input_folder',  required=True, help='FCNV2 output folder')
    parser.add_argument('-s','--save_folder',  required=True, help='Folder to save results')
    parser.add_argument('-t','--IC_time',  required=True, help='start time')
    parser.add_argument('-w','--modAFNO_weight',  default="modAFNO_weight", help='Path to modAFNO weight file')
    parser.add_argument('-d','--modAFNO_device',  default='cuda', help='Device for modAFNO model')
    args = parser.parse_args()  
    
    main(args.input_folder, args.save_folder, args.IC_time,
         modAFNO_weight=args.modAFNO_weight, modAFNO_device=args.modAFNO_device) 
