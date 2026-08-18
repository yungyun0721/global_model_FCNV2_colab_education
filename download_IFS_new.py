import xarray as xr
import numpy as np
import os, glob
import requests
import argparse
# prepare for FCNV2 data
# IC_time = '2024072100'
## IFS data: https://console.cloud.google.com/marketplace/product/bigquery-public-data/open-data-ecmwf
def main(IC_time: str, save_folder: str)-> None:
    output_data = f'{save_folder}/IFS_initial_condition.npy'
    os.makedirs(save_folder, exist_ok=True)

    IC_path = f'{save_folder}/IFS_IC_raw_data.grib2'
    if IC_time[-2:]=='06' or IC_time[-2:]=='18':
        download_name =f'https://storage.googleapis.com/ecmwf-open-data/{IC_time[:8]}/{IC_time[8:]}z/ifs/0p25/scda/{IC_time}0000-0h-scda-fc.grib2'
    else:
        download_name =f'https://storage.googleapis.com/ecmwf-open-data/{IC_time[:8]}/{IC_time[8:]}z/ifs/0p25/oper/{IC_time}0000-0h-oper-fc.grib2'
        print(download_name)
    response = requests.get(download_name)
    with open(IC_path, "wb") as f:
        f.write(response.content)
        f.close()

    # get data
    sfc_paramID = [165, 166, 167, 168, 151, 134, 179, 137, 228, 228246, 228247]
    sfc_datasets = None 
    for sfc_i in sfc_paramID:
        temp = xr.open_dataset(IC_path, filter_by_keys={'paramId': sfc_i})
        sfc_datasets = xr.merge([sfc_datasets,temp],compat='override') if sfc_datasets != None else  temp
    # sfc_paramID = [131, 132]
    # for sfc_i in sfc_paramID:
    #     temp = xr.open_dataset(IC_path, filter_by_keys={'typeOfLevel': 'heightAboveGround','paramId': sfc_i})
    #     sfc_datasets = xr.merge([sfc_datasets,temp],compat='override') if sfc_datasets != None else  temp
    
    sfc_datasets = sfc_datasets.assign_coords(longitude=(sfc_datasets.longitude % 360))
    sfc_datasets = sfc_datasets.sortby('longitude')

    upper_paramID = [157, 131, 132, 156, 135, 130]
    upper_datasets = None 
    for upper_i in upper_paramID:
        temp = xr.open_dataset(IC_path, filter_by_keys={'typeOfLevel': 'isobaricInhPa','paramId': upper_i})
        upper_datasets = xr.merge([upper_datasets,temp],compat='override') if upper_datasets != None else  temp
    upper_datasets = upper_datasets.assign_coords(longitude=(upper_datasets.longitude % 360))
    upper_datasets['z'] = upper_datasets['gh']*9.8
    upper_datasets = upper_datasets.drop_vars('gh')
    upper_datasets = upper_datasets.sortby('longitude')
    upper_datasets = upper_datasets.rename({'isobaricInhPa':'level'})
    upper_datasets =  upper_datasets.sortby('level') 

    target_lev = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    upper_datasets = upper_datasets.sel(level=target_lev)
    ordering = [ "10u",   "10v", "100u", "100v",   "2t",   "sp",  "msl", "tcwv",
                "u50",  "u100", "u150", "u200", "u250", "u300", "u400", "u500", "u600", "u700", "u850", "u925","u1000",
                "v50",  "v100", "v150", "v200", "v250", "v300", "v400", "v500", "v600", "v700", "v850", "v925","v1000",
                "z50",  "z100", "z150", "z200", "z250", "z300", "z400", "z500", "z600", "z700", "z850", "z925","z1000",
                "t50",  "t100", "t150", "t200", "t250", "t300", "t400", "t500", "t600", "t700", "t850", "t925","t1000",
                "r50",  "r100", "r150", "r200", "r250", "r300", "r400", "r500", "r600", "r700", "r850", "r925", "r1000"]

    u10 = sfc_datasets.variables['u10']
    v10 = sfc_datasets.variables['v10']
    # u100 = sfc_datasets.variables['100u']
    # v100 = sfc_datasets.variables['100v']
    u100 = sfc_datasets.variables['u100']
    v100 = sfc_datasets.variables['v100']
    t2m = sfc_datasets.variables['t2m']
    sp = sfc_datasets.variables['sp']
    msl = sfc_datasets.variables['msl']
    tcwv = sfc_datasets.variables['tcwv']
    surface = np.stack([u10, v10, u100, v100, t2m, sp, msl, tcwv])

    u = upper_datasets.variables['u']
    v = upper_datasets.variables['v']
    z = upper_datasets.variables['z']
    t = upper_datasets.variables['t']
    RH = upper_datasets.variables['r'] 
    total_input = np.concatenate([surface, u, v, z, t, RH], axis=0)  
    np.save(f'{output_data}', total_input.astype(np.float32))
    for file in glob.glob(f'{IC_path[:-6]}*'):
        os.remove(file)
    print(f'finish {IC_time}')

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t',"--scheduled-time", required=True, help="format: 2023072000")
    parser.add_argument('-s',"--save-folder", required=True, help="folder to save the downloaded data")
    args = parser.parse_args()
    main(args.scheduled_time, args.save_folder)  