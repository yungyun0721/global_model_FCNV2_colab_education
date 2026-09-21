"""Download ERA5 in the format of the accompanying WeatherNext training samples.

Example (initialization time is UTC, forecast length is 120 hours)::
    python download_ERA5_from_google_for_model_input.py -t 2024100700 -f 120

Outputs: inputs_data.nc (t-6h, t) and forcings_data.nc (t+6h ... forecast length).
Static fields are downloaded from ERA5 together with the weather variables.
Requires numpy, xarray, gcsfs, zarr and a NetCDF backend (e.g. netCDF4).
The 1-degree grid, 13 pressure levels, variable names and dimension order match
train_inputs.nc and train_forcings.nc supplied in this directory. ERA5's 0.25
degree grid is subsampled at the exact output coordinates, without averaging.
"""
"""
add the code after "Extract training and eval data"

!pip install -q -U xarray zarr gcsfs fsspec dask
!pip install -q netcdf4 cfgrib pygrib
!wget https://raw.githubusercontent.com/yungyun0721/global_model_FCNV2_colab_education/main/different_IC_for_WN2/PTC_download_ERA5_from_google_for_model_input.py
!python PTC_download_ERA5_from_google_for_model_input.py -t 2025072400 -f 120 -s input_data
eval_inputs = xarray.open_dataset('input_data/inputs_data.nc')
eval_forcings = xarray.open_dataset('input_data/forcings_data.nc')
"""


import argparse
import datetime
from pathlib import Path

import gcsfs
import numpy as np
import xarray as xr

ERA5_PATH = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
LAT = np.arange(-90, 91, dtype=np.float32)
LON = np.arange(0, 360, dtype=np.float32)
UPPER_VARIABLES = (
    'temperature', 'geopotential', 'u_component_of_wind',
    'v_component_of_wind', 'vertical_velocity', 'specific_humidity',
)
SURFACE_VARIABLES = (
    '2m_temperature', 'mean_sea_level_pressure', '10m_v_component_of_wind',
    '10m_u_component_of_wind', 'sea_surface_temperature',
)
STATIC_VARIABLES = ('geopotential_at_surface', 'land_sea_mask')

gcs = gcsfs.GCSFileSystem(token='anon')
full_era5 = xr.open_zarr(gcs.get_mapper(ERA5_PATH), chunks=None, consolidated=True)


def _time_features(initial_time, lead_hours, longitude=LON):
    """Match WeatherNext's get_year_progress/get_day_progress/featurize_progress.

    Reference: https://github.com/google-deepmind/weathernext/blob/main/
    weathernext/utils/data_utils.py
    Preserve the official float32 cast *before* computing the sin/cos phase.
    """
    leads = np.asarray(lead_hours, dtype='timedelta64[h]').astype('timedelta64[ns]')
    dates = np.datetime64(initial_time, 'ns') + leads
    seconds = dates.astype('datetime64[s]').astype(np.int64)
    year = np.mod(seconds / 86400 / np.float64(365.24219), 1.0).astype(np.float32)
    offsets = np.deg2rad(longitude) / (2 * np.pi)
    day = np.mod((seconds % 86400 / 86400)[:, None] + offsets, 1.0).astype(np.float32)
    data = xr.Dataset(coords={'time': leads, 'lon': longitude})
    for name, progress, dims in (
        ('year_progress', year[None, :], ('batch', 'time')),
        ('day_progress', day[None, :, :], ('batch', 'time', 'lon')),
    ):
        phase = progress * (2 * np.pi)
        data[name + '_sin'] = (dims, np.sin(phase))
        data[name + '_cos'] = (dims, np.cos(phase))
    return data


def make_inputs(initial_time):
    """Download ERA5 and create WeatherNext inputs at t-6h and t."""
    initial_time = np.datetime64(initial_time, 'ns')
    dates = initial_time + np.array([-6, 0], dtype='timedelta64[h]')

    start = full_era5.attrs.get('valid_time_start')
    stop = full_era5.attrs.get(
        'valid_time_stop_era5t',
        full_era5.attrs.get('valid_time_stop'),
    )
    if start and dates[0] < np.datetime64(start):
        raise ValueError(f'Input t-6h ({dates[0]}) precedes ERA5 availability: {start}')
    if stop and dates[-1] >= np.datetime64(stop, 'D') + np.timedelta64(1, 'D'):
        raise ValueError(f'Initialization {initial_time} is after available ERA5 data: {stop}')

    print('Downloading all ERA5 input variables together ...', flush=True)
    era5_data = full_era5[
        list(UPPER_VARIABLES + SURFACE_VARIABLES + STATIC_VARIABLES)
    ].sel(
        time=dates,
        level=LEVELS,
        latitude=LAT,
        longitude=LON,
    ).load()

    static_fields = era5_data[list(STATIC_VARIABLES)].isel(time=-1, drop=True)
    inputs = era5_data[list(UPPER_VARIABLES + SURFACE_VARIABLES)]

    relative_time = np.array([-6, 0], dtype='timedelta64[h]').astype('timedelta64[ns]')
    inputs = inputs.rename({'latitude': 'lat', 'longitude': 'lon'})
    inputs = inputs.assign_coords(
        time=relative_time,
        level=np.array(LEVELS, dtype=np.int32),
        lat=LAT,
        lon=LON,
    )
    inputs = inputs.astype(np.float32).expand_dims(batch=1, axis=0)

    for name in STATIC_VARIABLES:
        field = static_fields[name].rename(
            {'latitude': 'lat', 'longitude': 'lon'}
        ).transpose('lat', 'lon')
        inputs[name] = field.astype(np.float32)

    inputs = inputs[list(UPPER_VARIABLES + SURFACE_VARIABLES + STATIC_VARIABLES)]
    inputs.update(_time_features(initial_time, [-6, 0]))
    inputs.attrs['initialization_time_utc'] = np.datetime_as_string(initial_time, unit='s')
    inputs.attrs['source'] = ERA5_PATH
    inputs.attrs['static_fields_source'] = ERA5_PATH
    return inputs


def make_forcings(initial_time, forecast_hours):
    """Create WeatherNext forcings every 6 hours up to forecast_hours."""
    if forecast_hours < 6 or forecast_hours % 6 != 0:
        raise ValueError('forecast_hours must be a positive multiple of 6')

    lead_hours = np.arange(6, forecast_hours + 1, 6)
    forcings = _time_features(initial_time, lead_hours)
    forcings.attrs['initialization_time_utc'] = np.datetime_as_string(
        np.datetime64(initial_time, 'ns'),
        unit='s',
    )
    forcings.attrs['forecast_hours'] = forecast_hours
    forcings.attrs['source'] = (
        'Calculated with the official WeatherNext time-feature formula'
    )
    return forcings


def main(
    IC_time: str,
    save_folder: str,
    forecast_hours: int,
):
    if len(IC_time) != 10 or not IC_time.isdigit():
        raise ValueError('scheduled-time must be YYYYMMDDHH in UTC, e.g. 2024100700')

    initial_time = datetime.datetime.strptime(IC_time, '%Y%m%d%H')
    inputs = make_inputs(initial_time)
    forcings = make_forcings(initial_time, forecast_hours)

    output = Path(save_folder)
    output.mkdir(parents=True, exist_ok=True)
    for name, data in [('inputs_data.nc', inputs), ('forcings_data.nc', forcings)]:
        temporary = output / (name + '.tmp')
        try:
            data.to_netcdf(temporary)
            temporary.replace(output / name)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'Saved {output / name}', flush=True)

    print(
        f'Initialization (UTC): {initial_time:%Y-%m-%d %H:%M}; '
        f'inputs: -6h, 0h; forcings: +6h to +{forecast_hours}h'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-t', '--scheduled-time', required=True, help='UTC initialization time: YYYYMMDDHH')
    parser.add_argument('-s', '--save-folder', default='input_data', help='output folder (default: input_data)')
    parser.add_argument('-f', '--forecast-hours', type=int, default=6, help='forecast length in hours (multiple of 6)')
    args = parser.parse_args()
    main(args.scheduled_time, args.save_folder, args.forecast_hours)
