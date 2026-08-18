import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import argparse

def plot_figure(data_source_file,save_file):
  if not os.path.isdir(f'{save_file}'):
      os.mkdir(f'{save_file}')
  plt.ioff()  
  coast = pd.read_csv('coast.csv')

  lat = np.linspace(-90,90,721)
  lon = np.linspace(0,359.75,1440)

  lat_min  = np.argwhere(lat==0)[0][0]
  lat_max  = np.argwhere(lat==50.25)[0][0]
  lon_min  = np.argwhere(lon==90)[0][0]
  lon_max  = np.argwhere(lon==155.25)[0][0]

  lat = lat[lat_min:lat_max]
  lon = lon[lon_min:lon_max]

  files = 0
  file_list = os.listdir(data_source_file)
  for file_name in file_list:
      if file_name.startswith('output_precipitation'):
          files += 1
          
  print(f'files count: {files}')


  precip_lev =  [0, 0.5, 1, 2, 6, 10, 15, 20, 30, 40, 50, 70, 90, 110, 130, 150, 200, 300, 400]
  precip_color = [
      "#fdfdfd",  # 0.01 - 0.10 inches
      "#c9c9c9",  # 0.10 - 0.25 inches
      "#9dfeff",
      "#01d2fd",  # 0.25 - 0.50 inches
      "#00a5fe",  # 0.50 - 0.75 inches
      "#0177fd",  # 0.75 - 1.00 inches
      "#27a31b",  # 1.00 - 1.50 inches
      "#00fa2f",  # 1.50 - 2.00 inches
      "#fffe33",  # 2.00 - 2.50 inches
      "#ffd328",  # 2.50 - 3.00 inches
      "#ffa71f",  # 3.00 - 4.00 inches
      "#ff2b06",
      "#da2304",  # 4.00 - 5.00 inches
      "#aa1801",  # 5.00 - 6.00 inches
      "#ab1fa2",  # 6.00 - 8.00 inches
      "#db2dd2",  # 8.00 - 10.00 inches
      "#ff38fb",  # 10.00+
      "#ffd5fd"]


  for i in range(0,files):
      print(i)
      # data = np.load(f'{plot_dir}output_precipitation_{i}.npy')
      data = np.load(f'{data_source_file}/output_precipitation_{i*6:0>3}h.npy')
      data = data.squeeze()

      precip = np.flip(data[:, :], axis=0)[lat_min:lat_max, lon_min:lon_max]

      plt.figure(dpi=300)
      contourf = plt.contourf(lon, lat, precip*1000, levels=precip_lev, colors=precip_color)
      # contourf = plt.contourf(lon, lat, precip)
      # plt.streamplot(lon, lat, u, v, color='k', linewidth=0.5, density=1.5)

      plt.plot(coast.lon_map, coast.lat_map, color='k', linewidth=0.7)
      plt.xlim([90, 155])
      plt.ylim([0,50])
      plt.colorbar(contourf)
      plt.title(f'+{i*6-6}~{i*6} hour, precipitation')
      plt.savefig(f'{save_file}/precipitation_{i:0>3}h.png')
      plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--data_source_file", help="output data", default='../output_data')
    parser.add_argument("-s","--save_folder", help="plot save folder", default='plot_figure' )
    args = parser.parse_args()
    plot_figure(args.data_source_file, args.save_folder) 
