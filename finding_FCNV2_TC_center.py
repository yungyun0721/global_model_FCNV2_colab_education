import numpy as np
import pandas as pd
import datetime, os, argparse
def find_tc_center_msl(FCNV2_data: np.ndarray, prev_lat: float, prev_lon: float, search_radius_deg: float ):
    """
    在 msl 矩陣中找最低值點作為颱風中心，僅在前一中心附近搜尋（不處理經度循環）。

    Parameters:
        FCNV2_data: np.ndarray
        msl: np.ndarray，shape = (721, 1440)，緯度從 80 → -10，經度從 80 → 180
        prev_lat: 上一時刻颱風中心緯度
        prev_lon: 上一時刻颱風中心經度
        search_radius_deg: 搜尋半徑（單位：degree）

    Returns:
        new_lat, new_lon: 找到的最低壓位置（float, float）
        msl: 找最低氣壓
    """

    msl = FCNV2_data[6,:,:]
    u10 = FCNV2_data[0,:,:]
    v10 = FCNV2_data[1,:,:]
    u850 = FCNV2_data[18,:,:]
    v850 = FCNV2_data[18+13,:,:]
    u500 = FCNV2_data[15,:,:]
    v500 = FCNV2_data[15+13,:,:]
    ws10 = np.sqrt(u10*u10+v10*v10)
    # lat = np.flip(np.linspace(-90,90,721))
    # lon = np.linspace(0,359.75,1440)

    # lat_min  = np.argwhere(lat==-10)[0][0]
    # lat_max  = np.argwhere(lat==80)[0][0]
    # lon_min  = np.argwhere(lon==80)[0][0]
    # lon_max  = np.argwhere(lon==180)[0][0]

    # # 建立對應的 lat/lon 座標軸
    # lats = lat[lat_max: lat_min]
    # lons = lon[lon_min: lon_max]
    
    lats = np.flip(np.linspace(-90,90,721))
    lons = np.linspace(0,359.75,1440)

    # 找出搜尋框 index 範圍
    lat_idx = np.where((lats >= prev_lat - search_radius_deg) & (lats <= prev_lat + search_radius_deg))[0]
    lon_idx = np.where((lons >= prev_lon - search_radius_deg) & (lons <= prev_lon + search_radius_deg))[0]
    # lon_idx = np.where((lons >= prev_lon - search_radius_deg) & (lons <= prev_lon + search_radius_deg) & (lons >= 146))[0]

    
    
    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise ValueError("搜尋範圍超出資料界限，請檢查座標或半徑")

    sub_msl = msl[np.ix_(lat_idx, lon_idx)]
    sub_ws10 = ws10[np.ix_(lat_idx, lon_idx)]
    sub_ws_max_idx = np.unravel_index(np.argmax(sub_ws10), sub_ws10.shape)
    
    
    # 對應回全域 index
    i_lat = lat_idx[sub_ws_max_idx[0]]
    i_lon = lon_idx[sub_ws_max_idx[1]]
    
    max_u10 = u10[i_lat,i_lon]
    max_v10 = v10[i_lat,i_lon]
    max_ws_lat = lats[i_lat]
    max_ws_lon = lons[i_lon]
    
    # 在子區域找最小值索引
    sub_min_idx = np.unravel_index(np.argmin(sub_msl), sub_msl.shape)

    # 對應回全域 index
    i_lat = lat_idx[sub_min_idx[0]]
    i_lon = lon_idx[sub_min_idx[1]]
    
    # 算850平均渦度

    lat_sub = lats[i_lat-20:i_lat+21]
    lon_sub = lons[i_lon-20:i_lon+21]
    if len(lat_sub)>0 and len(lon_sub)>0:
        sub_u850 = u850[i_lat-20:i_lat+21, i_lon-20:i_lon+21]
        sub_v850 = v850[i_lat-20:i_lat+21, i_lon-20:i_lon+21]
        
        sub_u500 = u500[i_lat-20:i_lat+21, i_lon-20:i_lon+21]
        sub_v500 = v500[i_lat-20:i_lat+21, i_lon-20:i_lon+21]
        

        # 渦度場
        vort_sub_850 = compute_vorticity(sub_u850, sub_v850, lat_sub, lon_sub)
        vort_sub_500 = compute_vorticity(sub_u500, sub_v500, lat_sub, lon_sub)

        # 建立圓形 mask
        lon2d, lat2d = np.meshgrid(lon_sub, lat_sub)
        dist_deg = np.sqrt((lat2d - lats[i_lat])**2 + ((lon2d - lons[i_lon]) * np.cos(np.radians(lats[i_lat])))**2)
        # mask = dist_deg <= 5.0
    else:
        vort_sub_850 = np.full([41,41],np.nan)
        vort_sub_500 = np.full([41,41],np.nan)
        dist_deg = np.full([41,41],0.0)
        

    
    # 回傳實際經緯度
    return float(np.round(lats[i_lat], 2)), float(np.round(lons[i_lon], 2)), np.round(np.min(sub_msl),2),\
            float(np.round(max_ws_lat, 2)), float(np.round(max_ws_lon, 2)),\
                np.round(max_u10,2), np.round(max_v10,2), vort_sub_850, vort_sub_500, dist_deg

def compute_vorticity(u, v, lat, lon):
    """
    計算 850hPa 渦度 ζ = dv/dx - du/dy
    u, v shape: (lat, lon)
    lat, lon: 1D array
    """
    Re = 6371000  # 地球半徑 (m)
    lat_rad = np.radians(lat)
    dlat = np.radians(lat[1] - lat[0])
    dlon = np.radians(lon[1] - lon[0])
    
    # 經緯度網格
    lon2d, lat2d = np.meshgrid(lon, lat)

    # dx, dy in meters
    dx = Re * dlon * np.cos(lat2d * np.pi / 180)
    dy = Re * dlat

    # 中央差分法計算 dv/dx, du/dy
    dvdx = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * dx)
    dudy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dy)

    vort = dvdx - dudy
    return vort

#%%

# LLAT_path = '../output_data/FCNV2'
# initial_time = datetime.datetime.strptime('2026041300', "%Y%m%d%H")
# save_folder = "../output_data"

def main(FCNV2_path, IC_for_TC_center, IC_time, save_folder):
    
    initial_time = datetime.datetime.strptime(str(IC_time), "%Y%m%d%H")
    FCNV2_output_path = f'{FCNV2_path}/output_weather_000h.npy'

    FCNV2_data = np.load(FCNV2_output_path)
    msl = FCNV2_data[6,:,:]
    past_lat, past_lon, center_pressure, max_ws_lat, max_ws_lon, max_u10, max_v10, vort850, vort500, dist_deg \
        = find_tc_center_msl(FCNV2_data, IC_for_TC_center[0], IC_for_TC_center[1], 5.0)

        
    total_lat = [past_lat]
    total_lon = [past_lon]
    total_time = [initial_time.strftime("%Y%m%d%H")]
    total_pressure = [center_pressure]
    total_max_ws_lat = [max_ws_lat]
    total_max_ws_lon = [max_ws_lon]
    total_max_u10 = [max_u10]
    total_max_v10 = [max_v10]
    total_avg_vort850 = [np.nanmean(vort850[dist_deg <= 5.0])]
    total_avg_vort850_r3 = [np.nanmean(vort850[dist_deg <= 3.0])]
    total_avg_vort850_r2 = [np.nanmean(vort850[dist_deg <= 2.0])]
    total_avg_vort500 = [np.nanmean(vort500[dist_deg <= 5.0])]
    total_avg_vort500_r3 = [np.nanmean(vort500[dist_deg <= 3.0])]
    total_avg_vort500_r2 = [np.nanmean(vort500[dist_deg <= 2.0])]

    endpoint = len(os.listdir(f'{FCNV2_path}/'))
    for fore_time in range(1,endpoint):
        
        FCNV2_output_path = f'{FCNV2_path}/output_weather_{fore_time*6:0>3}h.npy'
        # FCNV2_output_path = f'{FCNV2_path}/forecast/output_weather_{fore_time*6:0>3}h.npy'
        FCNV2_data = np.load(FCNV2_output_path)

        past_lat, past_lon, center_pressure, max_ws_lat, max_ws_lon, max_u10, max_v10, vort850, vort500, dist_deg \
            = find_tc_center_msl(FCNV2_data, total_lat[-1], total_lon[-1], 5.0)
        total_lat.append(past_lat)
        total_lon.append(past_lon)
        total_pressure.append(center_pressure)
        fore_datetime = (initial_time+datetime.timedelta(hours=6*fore_time)).strftime("%Y%m%d%H")
        total_time.append(np.int_(fore_datetime))
        total_max_ws_lat.append(max_ws_lat)
        total_max_ws_lon.append(max_ws_lon)
        total_max_u10.append(max_u10)
        total_max_v10.append(max_v10)
        total_avg_vort850.append(np.nanmean(vort850[dist_deg <= 5.0]))
        total_avg_vort850_r3.append(np.nanmean(vort850[dist_deg <= 3.0]))
        total_avg_vort850_r2.append(np.nanmean(vort850[dist_deg <= 2.0]))
        total_avg_vort500.append(np.nanmean(vort500[dist_deg <= 5.0]))
        total_avg_vort500_r3.append(np.nanmean(vort500[dist_deg <= 3.0]))
        total_avg_vort500_r2.append(np.nanmean(vort500[dist_deg <= 2.0]))

    df = pd.DataFrame({
        "time": total_time,
        "lat": total_lat,
        "lon": total_lon,
        "Pressure (hPa)": total_pressure,
        "max_ws_lat":total_max_ws_lat,
        "max_ws_lon":total_max_ws_lon,
        "max_ws_u10":total_max_u10,
        "max_ws_v10":total_max_v10,
        "avg_vort850":np.round(total_avg_vort850,7),         
        "avg_vort850_r3":np.round(total_avg_vort850_r3,7),     
        "avg_vort850_r2":np.round(total_avg_vort850_r2,7),         
        "avg_vort500":np.round(total_avg_vort500,7),         
        "avg_vort500_r3":np.round(total_avg_vort500_r3,7),     
        "avg_vort500_r2":np.round(total_avg_vort500_r2,7), 
    })

    # 儲存成 CSV 檔
    df.to_csv(f'{save_folder}/FCNV2_TC_track_radius5.csv', index=False)
    print (f'finish TC initial time = {initial_time.strftime("%Y%m%d%H")}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--FCNV2_path',  required=True, help='Path to FCNV2 output data')
    parser.add_argument('--IC_for_TC_center', nargs=2, type=float,required=True, help='Initial coordinates for TC center (lat, lon)')
    parser.add_argument('--IC_time',  required=True, help='Initial time for IC data (ex: 2026041300)')
    parser.add_argument('--save_folder',  required=True, help='Folder to save results')
    args = parser.parse_args()  
    
    main(args.FCNV2_path, args.IC_for_TC_center, str(args.IC_time), args.save_folder)     

