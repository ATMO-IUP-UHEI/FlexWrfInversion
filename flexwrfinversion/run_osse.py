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

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from dask.distributed import Client
from loguru import logger
from pyinverse.loss import Bayesian, BayesianYM
from pyinverse.solver import BayesianAnalytical, BayesianAnalyticalYM
from tqdm.auto import tqdm

# flake8: noqa: F401
from flexwrfinversion.loaders.footprint import (
    FlexibleFootprintLoaderAnthBio,
    FlexibleFootprintLoaderAnthBioCo,
    FlexibleFootprintLoaderTotal,
    FootprintLoader,
)
from flexwrfinversion.loaders.measurement import (
    FlexibleMeasurementLoaderTotal,
    FlexibleMeasurementLoaderTotalCo,
    MeasurementLoader,
    MeasurementLoaderFromSingleFileTotal,
)
from flexwrfinversion.loaders.measurement_covariance import (
    ConstantNoCorrelation,
    ConstantNoCorrelationCO,
    ConstantPlusRelativeNoCorrelation,
    MeasurementCovarianceLoader,
)
from flexwrfinversion.loaders.prior import (
    FlatPrior,
    FlexiblePriorLoaderTotal_ShiftToBiospheric,
    PriorIsTarget,
    PriorLoader,
    PriorLoaderAnthBio_RelativeError_PointExtra,
    PriorLoaderAnthBioCo_RelativeError_PointExtra,
)
from flexwrfinversion.loaders.prior_covariance import (
    DifferenceOfPriorToTarget,
    DifferenceOfPriorToTargetWithCO_Correlation,
    PriorCovarianceLoader,
    RelativeError,
    RelativeErrorWithSpatialCorrelation,
    TargetAsError,
    TargetAsErrorNoCorrelation,
    TargetAsErrorWithCO_Correlation,
)
from flexwrfinversion.loaders.target import (
    FlexibleTargetLoaderAnthBio,
    FlexibleTargetLoaderAnthBioCo,
    FlexibleTargetLoaderTotal,
    TargetLoader,
)

TIME_BUFFER_FOR_INVERSION_WINDOW = np.timedelta64(30, "h")


def _get_args():
    parser = ArgumentParser(description="Run OSSE")
    parser.add_argument("config", type=Path, help="Path to the configuration file")
    return parser.parse_args()


def _get_kwargs(config, loader_type):
    kwargs = config[loader_type]["kwargs"]
    for key, new_kwargs in config.items():
        if (
            (loader_type in key)
            and (key != loader_type)
            and ((f"{loader_type}_covariance" in key) * key.count(loader_type))
            in [0, 2]
        ):
            kwargs.update(new_kwargs)
    logger.info(f"kwargs for {loader_type}: {kwargs}")
    return kwargs


def _initialize_loaders(
    config: dict,
) -> tuple[
    TargetLoader,
    PriorLoader,
    PriorCovarianceLoader,
    FootprintLoader,
    MeasurementLoader,
    MeasurementCovarianceLoader,
]:
    """
    Read config and initialize loaders.

    Args:
        config (dict): Configuration dictionary (of loaded config yaml).

    Returns:
        tuple[TargetLoader, PriorLoader, PriorCovarianceLoader, FootprintLoader,
            MeasurementLoader, MeasurementCovarianceLoader]: Tuple of loaders.
    """
    target_loader: TargetLoader = eval(config["target"]["target_loader"])(
        **_get_kwargs(config, "target")
    )
    prior_loader: PriorLoader = eval(config["prior"]["prior_loader"])(
        target_loader, **_get_kwargs(config, "prior")
    )
    prior_covariance_loader: PriorCovarianceLoader = eval(
        config["prior_covariance"]["prior_covariance_loader"]
    )(prior_loader, **_get_kwargs(config, "prior_covariance"))

    footprint_loader: FootprintLoader = eval(config["footprint"]["footprint_loader"])(
        **_get_kwargs(config, "footprint")
    )
    measurement_loader: MeasurementLoader = eval(
        config["measurement"]["measurement_loader"]
    )(target_loader, footprint_loader, **_get_kwargs(config, "measurement"))
    measurement_covariance_loader: MeasurementCovarianceLoader = eval(
        config["measurement_covariance"]["measurement_covariance_loader"]
    )(measurement_loader, **_get_kwargs(config, "measurement_covariance"))

    return (
        target_loader,
        prior_loader,
        prior_covariance_loader,
        footprint_loader,
        measurement_loader,
        measurement_covariance_loader,
    )


def _select_sites(
    measurements: xr.DataArray,
    measurement_covariance: xr.DataArray,
    footprint: xr.DataArray,
    sites: np.ndarray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Selects sites from the measurements, measurement covariance, and footprint.

    Args:
        measurements (xr.DataArray): Measurements to select sites from.
        measurement_covariance (xr.DataArray): Measurement covariance to select sites from
        footprint (xr.DataArray): Footprint to select sites from.
        sites (np.ndarray): Sites to select.

    Returns:
        tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]: Tuple of selected
        measurements, measurement covariance, footprint, and site selection. Site
        selection is a boolean array indicating the selected sites.
    """
    measurements = measurements.isel(measurement=measurements.MPlace.isin(sites))
    measurement_covariance = measurement_covariance.isel(
        measurement0=measurement_covariance.MPlace0.isin(sites),
        measurement1=measurement_covariance.MPlace1.isin(sites),
    )
    footprint = footprint.isel(measurement=footprint.MPlace.isin(sites))
    return measurements, measurement_covariance, footprint


def _compute_inversion(
    loss_class: Bayesian | BayesianYM,
    solver_class: BayesianAnalytical | BayesianAnalyticalYM,
    *args,
    **kwargs,
) -> BayesianAnalytical | BayesianAnalyticalYM:
    """
    Computes the inversion using the given data.

    Args:
        loss_class (Bayesian|BayesianYM): Loss function.
        solver_class (BayesianAnalytical|BayesianAnalyticalYM): Solver.
        *args: Arguments to pass to the loss function.
        **kwargs: Keyword arguments to pass to the loss function.

    Returns:
        BayesianAnalytical|BayesianAnalyticalYM: Solver holding inversion results.
    """
    loss = loss_class(
        *args,
        **kwargs,
    )
    solver = solver_class(loss)
    solver()
    return solver


def _format_results(
    solver: BayesianAnalytical,
    state_coordinates: xr.Coordinates,
    state_matrix_coordinates: xr.Coordinates,
    start_date: np.datetime64,
    end_date: np.datetime64,
    i: int,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Formats the results from the solver into xr.DataArrays.

    Args:
        solver (BayesianAnalytical): Solver holding inversion results.
        state_coordinates (xr.Coordinates): Coordinates for the state.
        state_matrix_coordinates (xr.Coordinates): Coordinates for the state matrix.
        start_date (np.datetime64): Start date of the inversion.
        end_date (np.datetime64): End date of the inversion.
        i (int): Index of the inversion used for concatenation dimension of averaging
        kernel sum.

    Returns:
        tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]: Tuple of posterior
        emissions, posterior standard deviation, averaging kernel diagonal, and averaging
        kernel sum.
    """
    posterior_emissions = (
        xr.DataArray(solver.x_posterior, coords=state_coordinates)
        .unstack()
        .sel(Time=slice(start_date, end_date - np.timedelta64(1, "h")))
    )
    posterior_std = (
        xr.DataArray(np.sqrt(np.diag(solver.cov_posterior)), coords=state_coordinates)
        .unstack()
        .sel(Time=slice(start_date, end_date - np.timedelta64(1, "h")))
    )
    averaging_kernel_diag = (
        xr.DataArray(np.diag(solver.averaging_kernel), coords=state_coordinates)
        .unstack()
        .sel(Time=slice(start_date, end_date - np.timedelta64(1, "h")))
    )
    averaging_kernel_sum = (
        xr.DataArray(
            solver.averaging_kernel,
            coords=state_matrix_coordinates,
        )
        .unstack()
        .sel(
            Time0=slice(start_date, end_date - np.timedelta64(1, "h")),
            Time1=slice(start_date, end_date - np.timedelta64(1, "h")),
        )
        .sum("Time0")
        .mean("Time1")
        .expand_dims(buffer=[i])
    )
    return (
        posterior_emissions,
        posterior_std,
        averaging_kernel_diag,
        averaging_kernel_sum,
    )


def _run_inversion(
    dates: np.ndarray,
    sites: np.ndarray,
    prior_loader: PriorLoader,
    prior_covariance_loader: PriorCovarianceLoader,
    target_loader: TargetLoader,
    footprint_loader: FootprintLoader,
    measurement_loader: MeasurementLoader,
    measurement_covariance_loader: MeasurementCovarianceLoader,
) -> xr.Dataset:
    """
    Runs the inversion for the given dates and sites 1 day at a time using a 3-day window
        (only keeping the center day).

    Args:
        dates (np.ndarray): Dates to run the inversion for.
        sites (np.ndarray): Sites to run the inversion for.
        prior_loader (PriorLoader): Prior loader.
        prior_covariance_loader (PriorCovarianceLoader): Prior covariance loader.
        target_loader (TargetLoader): Target loader.
        footprint_loader (FootprintLoader): Footprint loader.
        measurement_loader (MeasurementLoader): Measurement loader.
        measurement_covariance_loader (MeasurementCovarianceLoader): Measurement
        covariance loader.

    Returns:
        xr.Dataset: Inversion result.
    """
    site_selection = measurement_loader.measurements.unstack().MPlace.isin(sites)

    # Initialize lists to save results
    prior_std = []
    posterior_emissions = []
    posterior_std = []
    averaging_kernel_diag = []
    averaging_kernel_sum = []

    inversion_time = 0
    filter_time = 0
    data_getting_time = 0

    # Start inversion that operates dayly (three days run only center one is kept)
    for i, (start_date, end_date) in tqdm(
        enumerate(zip(dates[:-1], dates[1:])), total=len(dates) - 1
    ):
        # Set times to load emissions until 24 hours before the measurements
        emission_start_time = start_date - TIME_BUFFER_FOR_INVERSION_WINDOW
        emission_end_time = end_date + TIME_BUFFER_FOR_INVERSION_WINDOW
        measurement_start_time = emission_start_time + np.timedelta64(24, "h")
        measurement_end_time = end_date + TIME_BUFFER_FOR_INVERSION_WINDOW

        # Load data for set timeframes
        prior_emissions = prior_loader.load_timeframe(
            emission_start_time, emission_end_time
        )
        prior_emission_covariance = prior_covariance_loader.load_timeframe(
            emission_start_time, emission_end_time
        )
        measurements = measurement_loader.load_timeframe(
            measurement_start_time, measurement_end_time
        )
        measurement_covariance = measurement_covariance_loader.load_timeframe(
            measurement_start_time, measurement_end_time
        )
        footprint = footprint_loader.load_timeframe(
            emission_start_time,
            emission_end_time,
            measurement_start_time,
            measurement_end_time,
        )
        # Filter data for the sites selected in the given permutation
        measurements, measurement_covariance, footprint = _select_sites(
            measurements, measurement_covariance, footprint, sites
        )

        state_coordinates = prior_emissions.coords
        state_matrix_coordinates = prior_emission_covariance.coords

        # Inversion using the `pyinverse` module
        solver = _compute_inversion(
            loss_class=Bayesian,
            solver_class=BayesianAnalytical,
            x_prior=prior_emissions.values,
            cov_prior=prior_emission_covariance.values,
            y=measurements.values,
            cov_y=measurement_covariance.values,
            K=footprint.values.T,
        )

        # Build xr.DataArrays from the numpy output
        (
            partial_posterior_emissions,
            partial_posterior_std,
            partial_averaging_kernel_diag,
            partial_averaging_kernel_sum,
        ) = _format_results(
            solver, state_coordinates, state_matrix_coordinates, start_date, end_date, i
        )
        partial_prior_std = prior_covariance_loader.prior_std.unstack().sel(
            Time=slice(start_date, end_date - np.timedelta64(1, "h"))
        )

        # Append results to lists
        prior_std.append(partial_prior_std)
        posterior_emissions.append(partial_posterior_emissions)
        posterior_std.append(partial_posterior_std)
        averaging_kernel_diag.append(partial_averaging_kernel_diag)
        averaging_kernel_sum.append(partial_averaging_kernel_sum)

    # Concatenate the daily results and merge dataarrays
    logger.info(f"Filter time: {filter_time}")
    logger.info(f"Inversion time: {inversion_time}")
    logger.info(f"Data getting time: {data_getting_time}")

    prior_std = xr.concat(prior_std, dim="Time")
    posterior_emissions = xr.concat(posterior_emissions, dim="Time")
    posterior_std = xr.concat(posterior_std, dim="Time")
    averaging_kernel_diag = xr.concat(averaging_kernel_diag, dim="Time")
    averaging_kernel_sum = xr.concat(averaging_kernel_sum, dim="buffer").mean("buffer")
    prior = prior_loader.prior.unstack().sel(Time=prior_std.Time.values)
    target = target_loader.target.unstack().sel(Time=prior_std.Time.values)

    inversion_result = xr.merge(
        [
            prior.rename("prior_emissions"),
            prior_std.rename("prior_std"),
            target.rename("target_emissions"),
            posterior_emissions.rename("posterior_emissions"),
            posterior_std.rename("posterior_std"),
            averaging_kernel_diag.rename("averaging_kernel_diag"),
            averaging_kernel_sum.rename("averaging_kernel_sum"),
            site_selection.rename("site_selection"),
        ]
    )
    return inversion_result


def _prepare_permutations(config: dict, measurement_loader: MeasurementLoader) -> list:
    """Sets up stations to use for the permutations.

    Args:
        config (dict): Config file with n_stations, n_permutations, and permutation_seed.
        measurement_loader (MeasurementLoader): Measurement loader instance.

    Returns:
        list: List of np.arrays of stations to use for the permutations.
    """
    if "permutation_seed" in config:
        global_state = np.random.get_state()
        np.random.seed(config["permutation_seed"])

    mplace_value_permutations = [
        np.random.choice(
            measurement_loader.measurements.unstack().MPlace,
            config["n_stations"],
            replace=False,
        )
        for _ in range(config["n_permutations"])
    ]

    if "permutation_seed" in config:
        np.random.set_state(global_state)

    return mplace_value_permutations


def main(args):
    # load config yaml
    with args.config.open("r") as f:
        config = yaml.safe_load(f)

    # raise a UserWarning if n_processes is set in the config
    if "n_processes" in config:
        logger.warning(
            "The n_processes parameter is not supported in this version of the OSSE."
        )

    # if output_name does not end with .nc raise a warning
    if not config["output_name"].endswith(".nc"):
        raise ValueError("output_name must end with .nc")

    # build paths for the inversion and setup directories
    output_dir = Path(config["output_dir"])
    output_name = config["output_name"]
    output_buffer_path = output_dir / output_name.split(".")[0]
    output_buffer_path.mkdir(exist_ok=True, parents=True)
    client = Client()

    (
        target_loader,
        prior_loader,
        prior_covariance_loader,
        footprint_loader,
        measurement_loader,
        measurement_covariance_loader,
    ) = _initialize_loaders(config)
    mplace_value_permutations = _prepare_permutations(config, measurement_loader)

    # Start osses
    for i, mplace_values in tqdm(
        enumerate(mplace_value_permutations), total=len(mplace_value_permutations)
    ):
        if "start_index" in config:
            if i < config["start_index"]:
                continue
        # set dates for the inversion
        dates = np.arange(
            prior_loader.prior.Time[0].values,
            prior_loader.prior.Time[-1].values + np.timedelta64(1, "D"),
            dtype="datetime64[D]",
        )

        # Obtain actual inversionresult
        inversion_result = _run_inversion(
            dates,
            mplace_values,
            prior_loader,
            prior_covariance_loader,
            target_loader,
            footprint_loader,
            measurement_loader,
            measurement_covariance_loader,
        )

        # Save inversion result to buffer file
        inversion_result.to_netcdf(output_buffer_path / f"permutation_{i}.nc")
    client.restart(wait_for_workers=True)
    # Combine all permutations and save the result
    xr.open_mfdataset(
        output_buffer_path.glob("permutation_*.nc"),
        combine="nested",
        concat_dim="permutation",
    ).compute().to_netcdf(output_dir / output_name)

    # Delete the buffer files
    for file in output_buffer_path.glob("permutation_*.nc"):
        file.unlink()

    output_buffer_path.rmdir()
    client.close()


if __name__ == "__main__":
    args = _get_args()
    main(args)
