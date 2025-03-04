""" Runs OSSE
This script runs an Observing System Simulation Experiment (OSSE) using the configuration
     file provided.
The configuration file look as follows:

```
prior:
  prior_loader: ''              # Name of the prior loader class
  kwargs: {}                    # Arguments to pass to the prior loader
prior_covariance:
  prior_covariance_loader: ''   # Name of the prior covariance loader class
  kwargs: {}                    # Arguments to pass to the prior covariance loader
target:
  target_loader: ''             # Name of the target loader class
  kwargs: {}                    # Arguments to pass to the target loader
measurement:
  measurement_loader: ''        # Name of the measurement loader class
  kwargs: {}                    # Arguments to pass to the measurement loader
measurement_covariance:
  measurement_covariance_loader: '' # Name of the measurement covariance loader class
  kwargs: {}                    # Arguments to pass to the measurement covariance loader
footprint:
  footprint_loader: ''          # Name of the footprint loader class
  kwargs: {}                    # Arguments to pass to the footprint loader

n_permutations: #               # Number of permutations to run
n_stations: #                   # Number of stations to use
output_dir: ''                  # Directory to save output
output_name: ''                 # Name of the output file
(permutation_seed: #)           # Seed for the permutation (optional)
(start_index: #)                # Start index for the permutation (optional)
```
"""
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from dask.distributed import Client
from loguru import logger
from pyinverse.loss import BayesianYM
from pyinverse.solver import BayesianAnalyticalYM
from tqdm.auto import tqdm

# flake8: noqa: F401
from flexwrfinversion.loaders.footprint import FootprintLoader
from flexwrfinversion.loaders.measurement import MeasurementLoader
from flexwrfinversion.loaders.measurement_bias import (
    ConstantBias,
    MeasurementBias,
    RandomStaticBias,
    RelativeBias,
)
from flexwrfinversion.loaders.measurement_covariance import MeasurementCovarianceLoader
from flexwrfinversion.loaders.prior import PriorLoader
from flexwrfinversion.loaders.prior_covariance import PriorCovarianceLoader
from flexwrfinversion.loaders.target import TargetLoader
from flexwrfinversion.run_osse import (
    _compute_inversion,
    _get_args,
    _initialize_loaders,
    _prepare_permutations,
    _select_sites,
)

FLOAT_PRECISION = np.float32

mplace_value_permutations = None
prior_emissions = None
prior_standard_deviation = None
prior_temporal_correlation = None
prior_spatial_correlation = None
footprints = None
measurements = None
measurement_covariance = None
target_emissions = None
output_buffer_path = None
measurement_bias_loader = None
n_temporal_slices = None
temporal_buffer_steps = 0
meas_buffer_steps = 0
temporal_slice_data = None


def _initialize_measurement_bias_loader(
    config: dict,
    measurement_loader: MeasurementLoader,
    measurement_covariance_loader: MeasurementCovarianceLoader,
) -> MeasurementBias:
    measurement_bias_loader = None
    if "measurement_bias" in config:
        measurement_bias_loader = eval(
            config["measurement_bias"]["measurement_bias_loader"]
        )(
            measurement_loader=measurement_loader,
            measurement_covariance_loader=measurement_covariance_loader,
            **config["measurement_bias"]["kwargs"],
        )
    return measurement_bias_loader


def select_times(
    data: xr.DataArray,
    variable: str,
    dim: str,
    start: np.datetime64,
    end: np.datetime64,
) -> xr.DataArray:
    """
    Selects a time slice from the data.

    Args:
        data (xr.DataArray): Data to slice.
        variable (str): Variable to slice.
        start (np.datetime64): Start time.
        end (np.datetime64): End time.

    Returns:
        xr.DataArray: Sliced data.
    """
    return data.isel(
        {dim: (data[variable].values >= start) & (data[variable].values <= end)}
    )


def _run_inversion_ym_with_globals(
    i,
) -> xr.Dataset:
    """
    Runs the inversion using the given data.

    Args:
        i (int): Index of the permutation.

    Returns:
        xr.Dataset: Inversion result.
    """

    global mplace_value_permutations
    global prior_emissions
    global prior_standard_deviation
    global prior_temporal_correlation
    global prior_spatial_correlation
    global footprints
    global measurements
    global measurement_covariance
    global target_emissions
    global output_buffer_path
    global measurement_bias_loader
    global n_temporal_slices
    global temporal_buffer_steps
    global meas_buffer_steps
    global temporal_slice_data

    logger.info(f"Running permutation {i}")
    sites = mplace_value_permutations[i]

    site_selection = measurements.unstack().MPlace.isin(sites)
    (
        measurements_subset,
        measurement_covariance_subset,
        footprints_subset,
    ) = _select_sites(measurements, measurement_covariance, footprints, sites)
    state_coordinates = prior_emissions.coords

    if measurement_bias_loader is not None:
        measurement_bias_loader.set_bias()
        measurements_subset = (
            measurements_subset
            + measurement_bias_loader.generate_bias(measurements_subset)
        )

    if n_temporal_slices is not None:  # say n_temporal_slices = 9
        posterior_emissions = xr.full_like(prior_emissions, np.nan).unstack()
        posterior_std = xr.full_like(prior_emissions, np.nan).unstack()
        times = np.sort(np.unique(prior_emissions.Time))
        times_start = (
            np.sort(np.unique(prior_emissions.Time_start))
            if "Time_start" in prior_emissions.coords
            else times
        )
        times_end = (
            np.sort(np.unique(prior_emissions.Time_end))
            if "Time_end" in prior_emissions.coords
            else times
        )
        n_times = len(times)  # say n=100
        boundary_indices = np.linspace(
            0, n_times, n_temporal_slices + 1, dtype=int, endpoint=True
        )
        boundary_indices = np.floor(boundary_indices).astype(int)  # [0, 10, ..., 100]
        start_indices = boundary_indices[:-1]  # [0, 10, ..., 90]
        # say temporal_buffer_steps = 2
        start_with_buffer_indices = np.max(
            [start_indices - temporal_buffer_steps, np.zeros_like(start_indices)],
            axis=0,
        )  # [0, 8, ..., 88]
        end_indices = boundary_indices[1:] - 1  # [9, 19, ..., 99]

        end_with_buffer_indices = np.min(
            [
                end_indices + temporal_buffer_steps,
                np.ones_like(end_indices) * (n_times - 1),
            ],
            axis=0,
        )  # [11, 21 ..., 100]
        start_times = times[start_indices]
        end_times = times[end_indices]
        start_with_buffer_times = times[start_with_buffer_indices]
        measurement_start_with_buffer_times = times_start[
            start_with_buffer_indices + meas_buffer_steps
        ]
        end_with_buffer_times = times[end_with_buffer_indices]
        measurement_end_with_buffer_times = times_end[end_with_buffer_indices]
        for (
            start_time,
            end_time,
            start_with_buffer_time,
            end_with_buffer_time,
            measurement_start_with_buffer_time,
            measurement_end_with_buffer_time,
        ) in zip(
            start_times,
            end_times,
            start_with_buffer_times,
            end_with_buffer_times,
            measurement_start_with_buffer_times,
            measurement_end_with_buffer_times,
        ):
            prior_emissions_timeframe = prior_emissions.sel(
                Time=slice(start_with_buffer_time, end_with_buffer_time)
            )
            prior_standard_deviation_timeframe = prior_standard_deviation.sel(
                Time=slice(start_with_buffer_time, end_with_buffer_time)
            )
            prior_temporal_correlation_timeframe = prior_temporal_correlation.sel(
                Time0=slice(start_with_buffer_time, end_with_buffer_time),
                Time1=slice(start_with_buffer_time, end_with_buffer_time),
            )
            footprints_subset_timeframe = select_times(
                footprints_subset.sel(
                    Time=slice(start_with_buffer_time, end_with_buffer_time)
                ),
                "MTime",
                "measurement",
                measurement_start_with_buffer_time,
                measurement_end_with_buffer_time,
            )
            measurements_subset_timeframe = select_times(
                measurements_subset,
                "MTime",
                "measurement",
                measurement_start_with_buffer_time,
                measurement_end_with_buffer_time,
            )
            measurement_covariance_subset_timeframe = select_times(
                select_times(
                    measurement_covariance_subset,
                    "MTime0",
                    "measurement0",
                    measurement_start_with_buffer_time,
                    measurement_end_with_buffer_time,
                ),
                "MTime1",
                "measurement1",
                measurement_start_with_buffer_time,
                measurement_end_with_buffer_time,
            )
            state_coordinate_timeframe = prior_emissions_timeframe.coords
            solver = _compute_inversion(
                loss_class=BayesianYM,
                solver_class=BayesianAnalyticalYM,
                prior=prior_emissions_timeframe.values,
                prior_standard_deviation=prior_standard_deviation_timeframe.values,
                prior_temporal_correlation=prior_temporal_correlation_timeframe.values,
                prior_spatial_correlation=prior_spatial_correlation.values,
                forward_model=footprints_subset_timeframe.data,
                measurement=measurements_subset_timeframe.values,
                measurement_covariance=measurement_covariance_subset_timeframe.values,
            )
            (
                posterior_emissions_timeframe,
                posterior_standard_deviations_timeframe,
            ) = solver()
            posterior_emissions_timeframe = (
                xr.DataArray(
                    posterior_emissions_timeframe, coords=state_coordinate_timeframe
                )
                .unstack()
                .sel(Time=slice(start_time, end_time))
            )
            posterior_standard_deviations_timeframe = xr.DataArray(
                posterior_standard_deviations_timeframe,
                coords=state_coordinate_timeframe,
            ).unstack()
            # replace the values according to the start and end times
            posterior_emissions.loc[
                {"Time": slice(start_time, end_time)}
            ] = posterior_emissions_timeframe.sel(Time=slice(start_time, end_time))
            posterior_std.loc[
                {"Time": slice(start_time, end_time)}
            ] = posterior_standard_deviations_timeframe.sel(
                Time=slice(start_time, end_time)
            )

            logger.debug("-" * 50)
            logger.debug(f"{start_time=}")
            logger.debug(f"{end_time=}")
            logger.debug(f"{start_with_buffer_time=}")
            logger.debug(f"{end_with_buffer_time=}")
            # logger.debug(f"{prior_emissions_timeframe.Time.values=}")
            # logger.debug(f"{footprints_subset_timeframe.Time.values=}")
            # logger.debug(f"{footprints_subset_timeframe.unstack().MTime.values=}")
            # logger.debug(f"{measurements_subset_timeframe.unstack().MTime.values=}")
            # logger.debug(f"{footprints_subset_timeframe.unstack().sizes=}")
            # logger.debug(f"{measurements_subset_timeframe.unstack().sizes=}")
            logger.debug(f"{posterior_emissions_timeframe.Time.values=}")
            logger.debug(f"{measurements_subset_timeframe.unstack().MTime.values=}")

    elif temporal_slice_data is not None:
        posterior_emissions = xr.full_like(prior_emissions, np.nan).unstack()
        posterior_std = xr.full_like(prior_emissions, np.nan).unstack()

        for slice_id in temporal_slice_data.slice_id:
            start_time = temporal_slice_data.start_time.sel(slice_id=slice_id).values
            end_time = temporal_slice_data.end_time.sel(slice_id=slice_id).values
            start_with_buffer_time = temporal_slice_data.start_with_buffer_time.sel(
                slice_id=slice_id
            ).values
            end_with_buffer_time = temporal_slice_data.end_with_buffer_time.sel(
                slice_id=slice_id
            ).values
            measurement_start_with_buffer_time = (
                temporal_slice_data.measurement_start_with_buffer_time.sel(
                    slice_id=slice_id
                ).values
            )
            measurement_end_with_buffer_time = (
                temporal_slice_data.measurement_end_with_buffer_time.sel(
                    slice_id=slice_id
                ).values
            )
            prior_emissions_timeframe = prior_emissions.sel(
                Time=slice(start_with_buffer_time, end_with_buffer_time)
            )
            prior_standard_deviation_timeframe = prior_standard_deviation.sel(
                Time=slice(start_with_buffer_time, end_with_buffer_time)
            )
            prior_temporal_correlation_timeframe = prior_temporal_correlation.sel(
                Time0=slice(start_with_buffer_time, end_with_buffer_time),
                Time1=slice(start_with_buffer_time, end_with_buffer_time),
            )
            print("fp")
            footprints_subset_timeframe = select_times(
                footprints_subset.sel(
                    Time=slice(start_with_buffer_time, end_with_buffer_time)
                ),
                "MTime",
                "measurement",
                measurement_start_with_buffer_time,
                measurement_end_with_buffer_time,
            )
            print("ms")
            measurements_subset_timeframe = select_times(
                measurements_subset,
                "MTime",
                "measurement",
                measurement_start_with_buffer_time,
                measurement_end_with_buffer_time,
            )
            print("mc")
            measurement_covariance_subset_timeframe = select_times(
                select_times(
                    measurement_covariance_subset,
                    "MTime0",
                    "measurement0",
                    measurement_start_with_buffer_time,
                    measurement_end_with_buffer_time,
                ),
                "MTime1",
                "measurement1",
                measurement_start_with_buffer_time,
                measurement_end_with_buffer_time,
            )
            state_coordinate_timeframe = prior_emissions_timeframe.coords
            solver = _compute_inversion(
                loss_class=BayesianYM,
                solver_class=BayesianAnalyticalYM,
                prior=prior_emissions_timeframe.values,
                prior_standard_deviation=prior_standard_deviation_timeframe.values,
                prior_temporal_correlation=prior_temporal_correlation_timeframe.values,
                prior_spatial_correlation=prior_spatial_correlation.values,
                forward_model=footprints_subset_timeframe.data,
                measurement=measurements_subset_timeframe.values,
                measurement_covariance=measurement_covariance_subset_timeframe.values,
            )
            (
                posterior_emissions_timeframe,
                posterior_standard_deviations_timeframe,
            ) = solver()
            posterior_emissions_timeframe = (
                xr.DataArray(
                    posterior_emissions_timeframe, coords=state_coordinate_timeframe
                )
                .unstack()
                .sel(Time=slice(start_time, end_time))
            )
            posterior_standard_deviations_timeframe = xr.DataArray(
                posterior_standard_deviations_timeframe,
                coords=state_coordinate_timeframe,
            ).unstack()
            # replace the values according to the start and end times
            posterior_emissions.loc[
                {"Time": slice(start_time, end_time)}
            ] = posterior_emissions_timeframe.sel(Time=slice(start_time, end_time))
            posterior_std.loc[
                {"Time": slice(start_time, end_time)}
            ] = posterior_standard_deviations_timeframe.sel(
                Time=slice(start_time, end_time)
            )

    else:
        solver = _compute_inversion(
            loss_class=BayesianYM,
            solver_class=BayesianAnalyticalYM,
            prior=prior_emissions.values,
            prior_standard_deviation=prior_standard_deviation.values,
            prior_temporal_correlation=prior_temporal_correlation.values,
            prior_spatial_correlation=prior_spatial_correlation.values,
            forward_model=footprints_subset.data,
            measurement=measurements_subset.values,
            measurement_covariance=measurement_covariance_subset.values,
        )
        posterior_emissions, posterior_std = solver()
        posterior_emissions = xr.DataArray(
            posterior_emissions, coords=state_coordinates
        ).unstack()
        posterior_std = xr.DataArray(posterior_std, coords=state_coordinates).unstack()
    inversion_result = xr.merge(
        [
            prior_emissions.unstack().rename("prior_emissions"),
            prior_standard_deviation.unstack().rename("prior_std"),
            target_emissions.unstack().rename("target_emissions"),
            posterior_emissions.rename("posterior_emissions"),
            posterior_std.rename("posterior_std"),
            site_selection.rename("site_selection"),
        ]
    ).expand_dims(permutation=[i])
    inversion_result.to_netcdf(output_buffer_path / f"permutation_{i}.nc")


def _setup_restacking(
    prior_loader: PriorLoader,
) -> tuple[dict, list, list, list, dict]:
    """
    Setup restacking based on the prior loader.

    Args:
        prior_loader (PriorLoader): Prior loader used to analyze prior shape.

    Returns:
        tuple[dict, list, list, list, dict]: Dictionary of dimensions to stack, state
        order, footprint order, prior spatial correlation order, prior spatial
        correlation dimensions to stack.
    """
    dims_to_stack = dict()
    state_order = ["Time", "subsector"]
    footprint_order = ["measurement", "Time", "subsector"]
    prior_spatial_correlation_order = ["subsector0", "subsector1"]
    prior_spatial_correlation_dims_to_stack = dict()
    if "sector" in prior_loader.prior.unstack().dims:
        dims_to_stack["subsector_sector"] = ["subsector", "sector"]
        state_order = ["Time", "subsector_sector"]
        footprint_order = ["measurement", "Time", "subsector_sector"]
        prior_spatial_correlation_order = ["subsector0_sector0", "subsector1_sector1"]
        prior_spatial_correlation_dims_to_stack = dict(
            subsector0_sector0=["subsector0", "sector0"],
            subsector1_sector1=["subsector1", "sector1"],
        )
    return (
        dims_to_stack,
        state_order,
        footprint_order,
        prior_spatial_correlation_order,
        prior_spatial_correlation_dims_to_stack,
    )


def _load_inversion_data(
    prior_loader: PriorLoader,
    prior_covariance_loader: PriorCovarianceLoader,
    footprint_loader: FootprintLoader,
    measurement_loader: MeasurementLoader,
    measurement_covariance_loader: MeasurementCovarianceLoader,
    target_loader: TargetLoader,
) -> tuple[xr.DataArray]:
    """
    Load inversion data. All data is cast to float32. Returns flattened data.

    Args:
        prior_loader (PriorLoader): Prior loader.
        prior_covariance_loader (PriorCovarianceLoader): Prior covariance loader.
        footprint_loader (FootprintLoader): Footprint loader.
        measurement_loader (MeasurementLoader): Measurement loader.
        measurement_covariance_loader (MeasurementCovarianceLoader): Measurement
        covariance loader.
        target_loader (TargetLoader): Target loader.

    Returns:
        tuple[xr.DataArray]: Tuple of prior emissions, prior standard deviation, prior
        temporal correlation, prior spatial correlation, footprints, target emissions,
        measurements, measurement covariance.
    """
    prior_emissions = prior_loader.prior.astype(FLOAT_PRECISION)
    prior_standard_deviation = prior_covariance_loader.prior_std.astype(FLOAT_PRECISION)
    prior_temporal_correlation = prior_covariance_loader.temporal_correlation.astype(
        FLOAT_PRECISION
    )
    prior_spatial_correlation = prior_covariance_loader.spatial_correlation.astype(
        FLOAT_PRECISION
    )
    if "sector" in prior_loader.prior.unstack().dims:
        prior_spatial_correlation = (
            prior_spatial_correlation
            * prior_covariance_loader.sector_correlation.astype(FLOAT_PRECISION)
        )
    footprints = footprint_loader.footprint.astype(FLOAT_PRECISION)
    target_emissions = target_loader.target.astype(FLOAT_PRECISION)
    measurements = measurement_loader.measurements.astype(FLOAT_PRECISION).copy()
    measurement_covariance = (
        measurement_covariance_loader.load_timeframe(
            measurement_loader.measurements.MTime[0],
            measurement_loader.measurements.MTime[-1],
        )
        .astype(FLOAT_PRECISION)
        .copy()
    )
    return (
        prior_emissions,
        prior_standard_deviation,
        prior_temporal_correlation,
        prior_spatial_correlation,
        footprints,
        target_emissions,
        measurements,
        measurement_covariance,
    )


def _restack_coords(
    dataarray: xr.DataArray, unstack_dims, stack_dict=dict(), dim_order=[]
) -> xr.DataArray:
    """
    Restack coordinates.

    Args:
        dataarray (xr.DataArray): DataArray to restack.
        unstack_dims (list): List of dimensions to unstack.
        stack_dict (dict, optional): Dictionary of dimensions to stack. Defaults to
            dict().
        dim_order (list, optional): Order of dimensions. Defaults to [].

    Returns:
        xr.DataArray: Restacked DataArray.
    """
    if not unstack_dims == []:
        dataarray = dataarray.unstack(*unstack_dims)
    if not stack_dict == dict():
        dataarray = dataarray.stack(**stack_dict)
    if not dim_order == []:
        dataarray = dataarray.transpose(*dim_order)
    return dataarray


def main(args):
    global mplace_value_permutations
    global prior_emissions
    global prior_standard_deviation
    global prior_temporal_correlation
    global prior_spatial_correlation
    global footprints
    global measurements
    global measurement_covariance
    global target_emissions
    global output_buffer_path
    global measurement_bias_loader
    global n_temporal_slices
    global temporal_buffer_steps
    global meas_buffer_steps
    global temporal_slice_data

    # load config yaml
    with args.config.open("r") as f:
        config = yaml.safe_load(f)
    n_processes = 1
    if "n_processes" in config:
        n_processes = config["n_processes"]

    if "n_temporal_slices" in config and "temporal_slice_data_path" in config:
        raise ValueError(
            "Both 'n_temporal_slices' and 'temporal_slice_data' are provided. "
            "Please provide only one."
        )
    elif "n_temporal_slices" in config:
        n_temporal_slices = config["n_temporal_slices"]
        if "temporal_buffer_steps" in config:
            temporal_buffer_steps = config["temporal_buffer_steps"]
        if "meas_buffer_steps" in config:
            meas_buffer_steps = config["meas_buffer_steps"]
        logger.info(
            f"Running with {n_temporal_slices} temporal slices and "
            f"{temporal_buffer_steps} temporal buffer steps and "
            f"{meas_buffer_steps} measurement buffer steps."
        )

    elif "temporal_slice_data_path" in config:
        temporal_slice_data = xr.open_dataset(config["temporal_slice_data_path"])
        logger.info(
            f"Running with temporal slice data from {config['temporal_slice_data_path']}."
        )

    # build paths for the inversion and setup directories
    output_dir = Path(config["output_dir"])
    output_name = config["output_name"]
    output_buffer_path = output_dir / output_name.split(".")[0]
    output_buffer_path.mkdir(exist_ok=True, parents=True)

    client = Client()

    # initialize loaders based on the config
    (
        target_loader,
        prior_loader,
        prior_covariance_loader,
        footprint_loader,
        measurement_loader,
        measurement_covariance_loader,
    ) = _initialize_loaders(config)
    measurement_bias_loader = _initialize_measurement_bias_loader(
        config, measurement_loader, measurement_covariance_loader
    )
    # select station permutations
    logger.info("Setting up permutations")
    mplace_value_permutations = _prepare_permutations(config, measurement_loader)

    logger.info("Setting up restacking")
    (
        dims_to_stack,
        state_order,
        footprint_order,
        prior_spatial_correlation_order,
        prior_spatial_correlation_dims_to_stack,
    ) = _setup_restacking(prior_loader)

    logger.info("Loading data")
    (
        prior_emissions,
        prior_standard_deviation,
        prior_temporal_correlation,
        prior_spatial_correlation,
        footprints,
        target_emissions,
        measurements,
        measurement_covariance,
    ) = _load_inversion_data(
        prior_loader,
        prior_covariance_loader,
        footprint_loader,
        measurement_loader,
        measurement_covariance_loader,
        target_loader,
    )

    logger.info("Restacking data")
    prior_emissions = _restack_coords(
        prior_emissions, ["state"], dims_to_stack, state_order
    )
    prior_standard_deviation = _restack_coords(
        prior_standard_deviation, ["state"], dims_to_stack, state_order
    )
    target_emissions = _restack_coords(
        target_emissions, ["state"], dims_to_stack, state_order
    )
    footprints = _restack_coords(footprints, ["state"], dims_to_stack, footprint_order)
    prior_spatial_correlation = _restack_coords(
        prior_spatial_correlation,
        [],
        prior_spatial_correlation_dims_to_stack,
        prior_spatial_correlation_order,
    )
    client.close()

    with Pool(n_processes) as pool:
        # start  processes with tqdm
        with tqdm(total=len(mplace_value_permutations), smoothing=0) as pbar:
            for _ in pool.imap_unordered(
                _run_inversion_ym_with_globals, range(len(mplace_value_permutations))
            ):
                pbar.update()

    permutation_files = list(output_buffer_path.glob("permutation_*.nc"))
    permutation_files.sort()
    output_file = output_dir / output_name
    logger.info("Saving output")
    for i, file in tqdm(enumerate(permutation_files), total=len(permutation_files)):
        ds = xr.load_dataset(file).chunk({"permutation": 1})
        if i == 0:
            ds.to_zarr(output_file, mode="w", zarr_format=2)
        else:
            ds.to_zarr(
                output_file,
                mode="a",
                zarr_format=2,
                append_dim="permutation",
            )

    # Delete the buffer files
    for file in output_buffer_path.glob("permutation_*.nc"):
        file.unlink()

    output_buffer_path.rmdir()


if __name__ == "__main__":
    args = _get_args()
    main(args)
