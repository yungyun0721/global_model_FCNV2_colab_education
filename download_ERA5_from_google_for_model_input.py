import numpy as np
import datetime, os, gcsfs
import xarray as xr
import argparse
gcs = gcsfs.GCSFileSystem(token='anon')
era5_path = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
full_era5 = xr.open_zarr(gcs.get_mapper(era5_path), chunks=None,consolidated=True)


def q_to_rh(q, T_K, p):
    """
    q: 比濕 [kg/kg]
    T_K: 溫度 [K]
    p: 氣壓 [hPa]
    回傳 RH [%]
    """
    p = (np.array(p)[:,np.newaxis]*np.ones([1,q.shape[1]]))[:,:,np.newaxis]*np.ones([1,q.shape[2]])
    T_C = T_K - 273.15  # 轉為攝氏
    w = q / (1 - q)
    e = (w * p) / (0.622 + w)
    es = 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5))
    rh = (e / es) * 100
    return rh


# IC_time = "2026041400"
# TC_center_lat = 14.2
# TC_center_lon = 146.6
# save_folder = "input_data"
def main(IC_time: str, save_folder: str):
    os.makedirs(save_folder, exist_ok=True)

    IC_time = datetime.datetime.strptime(IC_time, "%Y%m%d%H")
    # for getting FCNV2 variable names
    ordering = [ "10u",   "10v", "100u", "100v",   "2t",   "sp",  "msl", "tcwv",
                "u50",  "u100", "u150", "u200", "u250", "u300", "u400", "u500", "u600", "u700", "u850", "u925","u1000",
                "v50",  "v100", "v150", "v200", "v250", "v300", "v400", "v500", "v600", "v700", "v850", "v925","v1000",
                "z50",  "z100", "z150", "z200", "z250", "z300", "z400", "z500", "z600", "z700", "z850", "z925","z1000",
                "t50",  "t100", "t150", "t200", "t250", "t300", "t400", "t500", "t600", "t700", "t850", "t925","t1000",
                "r50",  "r100", "r150", "r200", "r250", "r300", "r400", "r500", "r600", "r700", "r850", "r925", "r1000"]

    # for getting LLAT.ty variable names
    upper_variable = ['u_component_of_wind','v_component_of_wind', 'temperature','specific_humidity','geopotential','vertical_velocity']
    surface_variable = ['10m_u_component_of_wind','10m_v_component_of_wind','2m_temperature','2m_dewpoint_temperature', 
                        'mean_sea_level_pressure','surface_pressure','total_column_water_vapour','total_precipitation',
                        'mean_top_net_long_wave_radiation_flux','sea_surface_temperature','100m_u_component_of_wind','100m_v_component_of_wind']
    target_lev = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    
    ############# prepare FCNV2 input data
    era5_surface_original = full_era5[surface_variable].sel(time=IC_time.strftime("%Y-%m-%d %H"))
    era5_upper_original = full_era5[upper_variable].sel(time=IC_time.strftime("%Y-%m-%d %H"),level=target_lev)
    era5_upper_original["relative_humidity"] = (("level", "latitude", "longitude"), q_to_rh(era5_upper_original['specific_humidity'].values, era5_upper_original['temperature'].values, target_lev).astype(np.float32))
    u10 = era5_surface_original.variables['10m_u_component_of_wind']
    v10 = era5_surface_original.variables['10m_v_component_of_wind']
    u100 = era5_surface_original.variables['100m_u_component_of_wind']
    v100 = era5_surface_original.variables['100m_v_component_of_wind']
    t2m = era5_surface_original.variables['2m_temperature']
    sp = era5_surface_original.variables['surface_pressure']
    msl = era5_surface_original.variables['mean_sea_level_pressure']
    tcwv = era5_surface_original.variables['total_column_water_vapour']
    surface = np.stack([u10, v10, u100, v100, t2m, sp, msl, tcwv])

    u = era5_upper_original.variables['u_component_of_wind']
    v = era5_upper_original.variables['v_component_of_wind']
    z = era5_upper_original.variables['geopotential']
    t = era5_upper_original.variables['temperature']
    RH = era5_upper_original.variables['relative_humidity']
    total_FCNV2_input = np.concatenate([surface, u, v, z, t, RH], axis=0)  
    np.save(f'{save_folder}/ERA5_FCNV2_initial_condition.npy', total_FCNV2_input.astype(np.float32))
    print(f'finish FCNV2 IC:{IC_time.strftime("%Y%m%d%H")}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t',"--scheduled-time", required=True, help="format: 2023072000")
    parser.add_argument('-s',"--save-folder", default="input_data", help="folder to save input data")
    args = parser.parse_args()
    main(args.scheduled_time, args.save_folder) 