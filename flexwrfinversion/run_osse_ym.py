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

from flexwrfinversion.loaders.footprint import FootprintLoader
from flexwrfinversion.loaders.measurement import MeasurementLoader
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
    logger.info(f"Running permutation {i}")
    sites = mplace_value_permutations[i]

    site_selection = measurements.unstack().MPlace.isin(sites)
    (
        measurements_subset,
        measurement_covariance_subset,
        footprints_subset,
    ) = _select_sites(measurements, measurement_covariance, footprints, sites)
    state_coordinates = prior_emissions.coords
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
    posterior_emissions, posterior_standard_deviations = solver()
    posterior_emissions = xr.DataArray(
        posterior_emissions, coords=state_coordinates
    ).unstack()
    posterior_std = xr.DataArray(
        posterior_standard_deviations, coords=state_coordinates
    ).unstack()
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
    # load config yaml
    with args.config.open("r") as f:
        config = yaml.safe_load(f)
    n_processes = 1
    if "n_processes" in config:
        n_processes = config["n_processes"]
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
