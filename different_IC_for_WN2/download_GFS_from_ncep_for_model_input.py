"""Create WeatherNext 2 inputs and forcings from NCEP GFS analyses.

Two GFS 0.25-degree f000 files are downloaded: the initialization time and
the preceding 6 hours. The data are sampled on the WeatherNext 1-degree grid.

Example:
    python download_GFS_from_ncep_for_model_input.py -t 2025072400 -f 120 -s input_data

Outputs:
    input_data/inputs_data.nc
    input_data/forcings_data.nc
"""

import argparse
import datetime
from pathlib import Path

import numpy as np
import requests
import xarray as xr
from tqdm import tqdm


GFS_BASE_URL = "https://osdf-director.osg-htc.org/ncar/gdex/d084001"
GRAVITY = np.float32(9.80665)

LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
LAT = np.arange(-90, 91, dtype=np.float32)
LON = np.arange(0, 360, dtype=np.float32)

UPPER_VARIABLES = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
)
SURFACE_VARIABLES = (
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_v_component_of_wind",
    "10m_u_component_of_wind",
    "sea_surface_temperature",
)
STATIC_VARIABLES = ("geopotential_at_surface", "land_sea_mask")

# WeatherNext name: (GFS paramId, cfgrib variable name)
UPPER_GFS_FIELDS = {
    "temperature": (130, "t"),
    "geopotential": (156, "gh"),
    "u_component_of_wind": (131, "u"),
    "v_component_of_wind": (132, "v"),
    "vertical_velocity": (135, "w"),
    "specific_humidity": (133, "q"),
}
SURFACE_GFS_FIELDS = {
    "2m_temperature": ("heightAboveGround", 167, "t2m"),
    "mean_sea_level_pressure": ("meanSea", 260074, "prmsl"),
    "10m_v_component_of_wind": ("heightAboveGround", 166, "v10"),
    "10m_u_component_of_wind": ("heightAboveGround", 165, "u10"),
}


def download_gfs_analysis(valid_time, save_folder):
    """Download one GFS 0.25-degree analysis (f000) and return its path."""
    time_text = valid_time.strftime("%Y%m%d%H")
    filename = f"gfs.0p25.{time_text}.f000.grib2"
    url = (
        f"{GFS_BASE_URL}/{valid_time:%Y}/{valid_time:%Y%m%d}/{filename}"
    )
    output_path = Path(save_folder) / filename

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Using existing file: {output_path}")
        return output_path

    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    print(f"Downloading GFS analysis: {time_text} UTC")
    print(url)
    try:
        with requests.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            with temporary_path.open("wb") as output_file, tqdm(
                total=total_size or None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=time_text,
            ) as progress:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output_file.write(chunk)
                        progress.update(len(chunk))
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return output_path


def _read_grib_field(grib_path, type_of_level, param_id, variable_name):
    """Read one GRIB field and select the exact 1-degree grid points."""
    with xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {
                "typeOfLevel": type_of_level,
                "paramId": param_id,
            },
            "indexpath": "",
        },
    ) as dataset:
        field = dataset[variable_name].sel(
            latitude=LAT,
            longitude=LON,
        ).load()

    return field.reset_coords(drop=True)


def read_gfs_weather(grib_path):
    """Read the time-dependent WeatherNext fields from one GFS file."""
    data = xr.Dataset()

    for output_name, (param_id, grib_name) in UPPER_GFS_FIELDS.items():
        field = _read_grib_field(
            grib_path,
            "isobaricInhPa",
            param_id,
            grib_name,
        )
        field = field.rename(
            {
                "isobaricInhPa": "level",
                "latitude": "lat",
                "longitude": "lon",
            }
        ).sel(level=LEVELS)
        field = field.transpose("level", "lat", "lon").astype(np.float32)

        # GFS HGT is geopotential height (gpm); WeatherNext uses geopotential.
        if output_name == "geopotential":
            field = field * GRAVITY
        data[output_name] = field

    for output_name, (type_of_level, param_id, grib_name) in SURFACE_GFS_FIELDS.items():
        field = _read_grib_field(
            grib_path,
            type_of_level,
            param_id,
            grib_name,
        ).rename({"latitude": "lat", "longitude": "lon"})
        data[output_name] = field.transpose("lat", "lon").astype(np.float32)

    # GFS surface temperature is used as SST and masked over land below.
    surface_temperature = _read_grib_field(
        grib_path,
        "surface",
        130,
        "t",
    ).rename({"latitude": "lat", "longitude": "lon"})
    land_sea_mask = _read_grib_field(
        grib_path,
        "surface",
        172,
        "lsm",
    ).rename({"latitude": "lat", "longitude": "lon"})
    data["sea_surface_temperature"] = surface_temperature.where(
        land_sea_mask < 0.5
    ).transpose("lat", "lon").astype(np.float32)

    data = data.assign_coords(
        level=np.array(LEVELS, dtype=np.int32),
        lat=LAT,
        lon=LON,
    )
    return data[list(UPPER_VARIABLES + SURFACE_VARIABLES)]


def read_gfs_static_fields(grib_path):
    """Read land-sea mask and surface geopotential from the t analysis."""
    land_sea_mask = _read_grib_field(
        grib_path,
        "surface",
        172,
        "lsm",
    ).rename({"latitude": "lat", "longitude": "lon"})
    surface_height = _read_grib_field(
        grib_path,
        "surface",
        228002,
        "orog",
    ).rename({"latitude": "lat", "longitude": "lon"})

    return xr.Dataset(
        {
            "geopotential_at_surface": (
                ("lat", "lon"),
                (surface_height * GRAVITY).transpose("lat", "lon").values.astype(np.float32),
            ),
            "land_sea_mask": (
                ("lat", "lon"),
                land_sea_mask.transpose("lat", "lon").values.astype(np.float32),
            ),
        },
        coords={"lat": LAT, "lon": LON},
    )


def _time_features(initial_time, lead_hours):
    """Create the four time features with the official WeatherNext formula."""
    leads = np.asarray(lead_hours, dtype="timedelta64[h]").astype("timedelta64[ns]")
    dates = np.datetime64(initial_time, "ns") + leads
    seconds = dates.astype("datetime64[s]").astype(np.int64)

    year = np.mod(
        seconds / 86400 / np.float64(365.24219),
        1.0,
    ).astype(np.float32)
    longitude_offsets = np.deg2rad(LON) / (2 * np.pi)
    day = np.mod(
        (seconds % 86400 / 86400)[:, None] + longitude_offsets,
        1.0,
    ).astype(np.float32)

    data = xr.Dataset(coords={"time": leads, "lon": LON})
    for name, progress, dims in (
        ("year_progress", year[None, :], ("batch", "time")),
        ("day_progress", day[None, :, :], ("batch", "time", "lon")),
    ):
        phase = progress * (2 * np.pi)
        data[name + "_sin"] = (dims, np.sin(phase))
        data[name + "_cos"] = (dims, np.cos(phase))
    return data


def make_inputs(initial_time, grib_paths):
    """Combine the t-6h and t GFS analyses into WeatherNext inputs."""
    relative_times = np.array([-6, 0], dtype="timedelta64[h]").astype(
        "timedelta64[ns]"
    )
    weather = []
    for relative_time, grib_path in zip(relative_times, grib_paths):
        print(f"Reading {grib_path} ...", flush=True)
        one_time = read_gfs_weather(grib_path).expand_dims(
            time=[relative_time],
            axis=0,
        )
        weather.append(one_time)

    inputs = xr.concat(weather, dim="time").expand_dims(batch=1, axis=0)
    static_fields = read_gfs_static_fields(grib_paths[-1])
    inputs = xr.merge([inputs, static_fields], compat="override")
    inputs = inputs[list(UPPER_VARIABLES + SURFACE_VARIABLES + STATIC_VARIABLES)]
    inputs.update(_time_features(initial_time, [-6, 0]))
    inputs.attrs["initialization_time_utc"] = np.datetime_as_string(
        np.datetime64(initial_time, "ns"),
        unit="s",
    )
    inputs.attrs["source"] = "NCEP GFS 0.25-degree analysis (f000)"
    inputs.attrs["gfs_input_files"] = ", ".join(str(path) for path in grib_paths)
    return inputs


def make_forcings(initial_time, forecast_hours):
    """Create WeatherNext forcings every 6 hours to the forecast length."""
    if forecast_hours < 6 or forecast_hours % 6 != 0:
        raise ValueError("forecast_hours must be a positive multiple of 6")

    lead_hours = np.arange(6, forecast_hours + 1, 6)
    forcings = _time_features(initial_time, lead_hours)
    forcings.attrs["initialization_time_utc"] = np.datetime_as_string(
        np.datetime64(initial_time, "ns"),
        unit="s",
    )
    forcings.attrs["forecast_hours"] = forecast_hours
    forcings.attrs["source"] = "Official WeatherNext time-feature formula"
    return forcings


def save_netcdf(dataset, output_path):
    """Write a complete temporary NetCDF before replacing the output."""
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        dataset.to_netcdf(temporary_path)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Saved: {output_path}")


def remove_grib_files(grib_paths):
    """Remove the two downloaded GFS files after successful conversion."""
    for grib_path in grib_paths:
        grib_path = Path(grib_path)
        grib_path.unlink(missing_ok=True)
        print(f"Removed: {grib_path}")


def main(IC_time, save_folder, forecast_hours):
    initial_time = datetime.datetime.strptime(IC_time, "%Y%m%d%H")
    if initial_time.hour not in (0, 6, 12, 18):
        raise ValueError("GFS initialization hour must be 00, 06, 12, or 18 UTC")

    output_folder = Path(save_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    input_times = [initial_time - datetime.timedelta(hours=6), initial_time]
    grib_paths = [
        download_gfs_analysis(valid_time, output_folder)
        for valid_time in input_times
    ]

    inputs = make_inputs(initial_time, grib_paths)
    forcings = make_forcings(initial_time, forecast_hours)
    save_netcdf(inputs, output_folder / "inputs_data.nc")
    save_netcdf(forcings, output_folder / "forcings_data.nc")
    remove_grib_files(grib_paths)

    print(
        f"Initialization: {initial_time:%Y-%m-%d %H:%M} UTC; "
        f"inputs: -6h and 0h; forcings: +6h to +{forecast_hours}h"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-t",
        "--scheduled-time",
        required=True,
        help="GFS initialization time in UTC: YYYYMMDDHH",
    )
    parser.add_argument(
        "-s",
        "--save-folder",
        default="input_data",
        help="folder for GRIB2 and NetCDF files",
    )
    parser.add_argument(
        "-f",
        "--forecast-hours",
        type=int,
        default=6,
        help="forecast length in hours (positive multiple of 6)",
    )
    args = parser.parse_args()
    main(args.scheduled_time, args.save_folder, args.forecast_hours)
