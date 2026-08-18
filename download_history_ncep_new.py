import argparse
import requests
import os, glob
import xarray as xr
import numpy as np
from tqdm import tqdm
# Set the path for saving NCEP GFS file(s)
# https://thredds.rda.ucar.edu/thredds/catalog/catalog_d084001.html
# https://thredds.rda.ucar.edu/thredds/fileServer/files/g/d084001/2024/20240108/gfs.0p25.2024010800.f000.grib2
# 2026/08/18 new_data: https://gdex.ucar.edu/datasets/d084001
# https://osdf-director.osg-htc.org/ncar/gdex/d084001/2026/20260816/gfs.0p25.2026081600.f000.grib2

def main(IC_time: str, save_folder: str)->None:
    file = f'{IC_time[0:4]}/{IC_time[0:8]}/gfs.0p25.{IC_time}.f000.grib2'
    IC_path = f'{save_folder}/ncep_data.grib2'
    output_data = f'{save_folder}/ncep_initial_condition.npy'
    os.makedirs(save_folder, exist_ok=True)

    url = "https://osdf-director.osg-htc.org/ncar/gdex/d084001/" + file
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 1024 * 1024  # 1 MiB

        with open(IC_path, "wb") as f, tqdm(
            total=total_size or None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Downloading NCEP GFS",
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))

    sfc_paramID = [165, 166, 167, 260074, 134, 3054, 228246, 228247]
    sfc_datasets = None 
    for sfc_i in sfc_paramID:
        temp = xr.open_dataset(
            IC_path,
            engine="cfgrib",
            filter_by_keys={'paramId': sfc_i},
        )
        sfc_datasets = xr.merge([sfc_datasets,temp],compat='override') if sfc_datasets != None else  temp

    # Get 100 m wind only when it is not already present.
    if 'u100' not in sfc_datasets.variables:
        temp = xr.open_dataset(
            IC_path,
            engine="cfgrib",
            filter_by_keys={'typeOfLevel': 'heightAboveGround', 'paramId': 131},
        )
        u100 = temp['u'].sel(heightAboveGround=100).rename('u100')
        sfc_datasets = xr.merge([sfc_datasets, u100], compat='override')

    if 'v100' not in sfc_datasets.variables:
        temp = xr.open_dataset(
            IC_path,
            engine="cfgrib",
            filter_by_keys={'typeOfLevel': 'heightAboveGround', 'paramId': 132},
        )
        v100 = temp['v'].sel(heightAboveGround=100).rename('v100')
        sfc_datasets = xr.merge([sfc_datasets, v100], compat='override')


    upper_paramID = [131, 132, 156, 157, 130]
    upper_datasets = None 
    for upper_i in upper_paramID:
        temp = xr.open_dataset(
            IC_path,
            engine="cfgrib",
            filter_by_keys={'typeOfLevel': 'isobaricInhPa', 'paramId': upper_i},
        )
        upper_datasets = xr.merge([upper_datasets,temp],compat='override') if upper_datasets != None else  temp
        upper_datasets =  upper_datasets.sortby('isobaricInhPa') 
    upper_datasets['z'] = upper_datasets['gh']*9.8
    upper_datasets = upper_datasets.drop_vars('gh')
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
    u100 = sfc_datasets.variables['u100']
    v100 = sfc_datasets.variables['v100']
    t2m = sfc_datasets.variables['t2m']
    sp = sfc_datasets.variables['sp']
    msl = sfc_datasets.variables['prmsl']
    tcwv = sfc_datasets.variables['pwat']
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