import os
import numpy as np
import xarray as xr
import pandas as pd
import datetime
import argparse

from inference_helper import FCNV2_model

FCNV2_info = [ "10u",   "10v", "100u", "100v",   "2t",   "sp",  "msl", "tcwv",
                "u50",  "u100", "u150", "u200", "u250", "u300", "u400", "u500", "u600", "u700", "u850", "u925","u1000",
                "v50",  "v100", "v150", "v200", "v250", "v300", "v400", "v500", "v600", "v700", "v850", "v925","v1000",
                "z50",  "z100", "z150", "z200", "z250", "z300", "z400", "z500", "z600", "z700", "z850", "z925","z1000",
                "t50",  "t100", "t150", "t200", "t250", "t300", "t400", "t500", "t600", "t700", "t850", "t925","t1000",
                "r50",  "r100", "r150", "r200", "r250", "r300", "r400", "r500", "r600", "r700", "r850", "r925", "r1000"]


def main(FCNV2_IC_path, save_folder, fore_hour=72,
         FCNV2_weight="FCNV2_weight",FCNV2_device='cuda'):    
    
    # save FCNV2 small domain
    lat = np.flip(np.linspace(-90,90,721))
    lon = np.linspace(0,359.75,1440)

    # lat_min  = np.argwhere(lat==-10)[0][0]
    # lat_max  = np.argwhere(lat==80)[0][0]
    # lon_min  = np.argwhere(lon==80)[0][0]
    # lon_max  = np.argwhere(lon==180)[0][0]
    
    # initialize model
    print('initialize model ...')
    print(f'FCNV2 weight: {FCNV2_weight}, device: {FCNV2_device}')
    FCNV2 = FCNV2_model(FCNV2_weight, device=FCNV2_device)
    FCNV2.initialize()

    # save folder    
    FCNV2_save_path = save_folder
    os.makedirs(FCNV2_save_path, exist_ok=True)
    print(f'save folder: {FCNV2_save_path}')
    
    with open(f"log.txt", "a") as f:
        print('\n',file=f)
        print(f'start time: {datetime.datetime.now()}',file=f)

    # building save folder
    # load and save IC_data
    print('load IC data and save ...')
    print(f'FCNV2 IC path: {FCNV2_IC_path}')
    FCNV2_input = np.load(FCNV2_IC_path)

    # np.save(os.path.join(FCNV2_save_path, f"output_weather_{0:0>3}h"),FCNV2_input[:,lat_max:lat_min,lon_min:lon_max])
    np.save(os.path.join(FCNV2_save_path, f"output_weather_{0:0>3}h"),FCNV2_input)

    for fore_i in range(1, np.int_(fore_hour/6)+1):
        
        # running FCNV2 forecast
        FCNV2_output = FCNV2.predict_one_step(FCNV2_input)
        # np.save(os.path.join(FCNV2_save_path, f"output_weather_{fore_i*6:0>3}h"),FCNV2_output[:,lat_max:lat_min,lon_min:lon_max])
        np.save(os.path.join(FCNV2_save_path, f"output_weather_{fore_i*6:0>3}h"),FCNV2_output)
        FCNV2_input = FCNV2_output

        if np.mod(fore_i,4)==0:
            with open(f"log.txt", "a") as f:
                print(f'finish : forecast {(fore_i*6):0>3} hr',file=f)
            print(f'finish : forecast {(fore_i*6):0>3} hr')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Define arguments to get YAML config file.
    parser.add_argument('-i','--FCNV2_IC_path',  required=True, help='Path to FCNV2 IC data')
    parser.add_argument('-s','--save_folder',  required=True, help='Folder to save results')
    parser.add_argument('-f','--fore_hour',  default=72, help='Forecast hours')
    parser.add_argument('-w','--FCNV2_weight',  default="FCNV2_weight", help='Path to FCNV2 weight file')
    parser.add_argument('-d','--FCNV2_device',  default='cuda', help='Device for FCNV2 model')
    args = parser.parse_args()  
    
    main(args.FCNV2_IC_path, args.save_folder, fore_hour=int(args.fore_hour), 
         FCNV2_weight=args.FCNV2_weight, FCNV2_device=args.FCNV2_device) 