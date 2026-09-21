# ERA5 Initial Conditions for WeatherNext 2

This directory provides a simple way to run WeatherNext 2 with a different ERA5 initial condition.

The downloader reads ERA5 data from the public Google Cloud ARCO-ERA5 archive and creates two NetCDF files compatible with the WeatherNext 2 demo:

- `inputs_data.nc`: model inputs at `-6 h` and `0 h`
- `forcings_data.nc`: time-dependent forcings from `+6 h` to the requested forecast length, at 6-hour intervals

The atmospheric, surface, land-sea mask, and surface geopotential fields are all read from ERA5. The year/day sine and cosine forcing variables follow the WeatherNext implementation.

## Related WeatherNext resources

- [WeatherNext repository](https://github.com/google-deepmind/weathernext)
- [Official WeatherNext 2 Colab demo](https://github.com/google-deepmind/weathernext/blob/main/docs/weathernext2/wn2_demo.ipynb)
- [ERA5 input downloader](https://github.com/yungyun0721/global_model_FCNV2_colab_education/blob/main/different_IC_for_WN2/PTC_download_ERA5_from_google_for_model_input.py)

## Usage in Google Colab

1. Open the official [WeatherNext 2 demo notebook](https://github.com/google-deepmind/weathernext/blob/main/docs/weathernext2/wn2_demo.ipynb) in Google Colab.
2. Connect to a Colab runtime.
3. Run the official notebook through the cell named **Extract training and eval data**.
4. Add a new code cell immediately after that cell and paste the following code. The downloader will be fetched directly from GitHub, so no manual file upload is needed.

```python
!pip install -q -U xarray zarr gcsfs fsspec dask
!pip install -q netcdf4 cfgrib pygrib

!wget -q -O download_ERA5_from_google_for_model_input.py \
    https://raw.githubusercontent.com/yungyun0721/global_model_FCNV2_colab_education/main/different_IC_for_WN2/download_ERA5_from_google_for_model_input.py
    
!wget -q -O download_GFS_from_ncep_for_model_input.py \
    https://raw.githubusercontent.com/yungyun0721/global_model_FCNV2_colab_education/main/different_IC_for_WN2/download_GFS_from_ncep_for_model_input.py


```
```python
!python download_ERA5_from_google_for_model_input.py \
    -t 2025072400 \
    -f 120 \
    -s input_data


eval_inputs = xarray.open_dataset('input_data/inputs_data.nc')
eval_forcings = xarray.open_dataset('input_data/forcings_data.nc')
```


Continue running the remaining WeatherNext 2 demo cells. The new `eval_inputs` and `eval_forcings` replace the example initial conditions and forcings loaded by the official notebook.

## Command-line options

```text
-t, --scheduled-time   ERA5 initialization time in UTC, formatted as YYYYMMDDHH
-f, --forecast-hours   Forecast length in hours; it must be a positive multiple of 6
-s, --save-folder      Directory for inputs_data.nc and forcings_data.nc
```

For example:

```bash
python PTC_download_ERA5_from_google_for_model_input.py \
    --scheduled-time 2025072400 \
    --forecast-hours 120 \
    --save-folder input_data
```

With `--forecast-hours 120`, `forcings_data.nc` contains 20 forecast times:

```text
6 h, 12 h, 18 h, ..., 120 h
```

The forcing length must match the time length of `eval_targets` used as the prediction template in the official notebook. The official demo setting `steps = "20"` corresponds to a 120-hour forecast with 20 six-hour steps.

| Official demo `steps` | Forecast length |
|---:|---:|
| `"01"` | 6 hours |
| `"04"` | 24 hours |
| `"12"` | 72 hours |
| `"20"` | 120 hours |
| `"30"` | 180 hours |
