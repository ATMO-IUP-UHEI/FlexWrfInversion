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


def _run_inversion_ym(
    sites: np.ndarray,
    prior_emissions: xr.DataArray,
    prior_standard_deviation: xr.DataArray,
    prior_temporal_correlation: xr.DataArray,
    prior_spatial_correlation: xr.DataArray,
    footprints: xr.DataArray,
    measurements: xr.DataArray,
    measurement_covariance: xr.DataArray,
    target_emissions: xr.DataArray,
) -> xr.Dataset:
    """
    Runs the inversion using the given data.

    Args:
        sites (np.ndarray): Sites to run the inversion for.
        prior_emissions (xr.DataArray): Prior emissions.
        prior_standard_deviation (xr.DataArray): Prior standard deviation.
        prior_temporal_correlation (xr.DataArray): Prior temporal correlation.
        prior_spatial_correlation (xr.DataArray): Prior spatial correlation.
        footprints (xr.DataArray): Footprints.
        measurements (xr.DataArray): Measurements.
        measurement_covariance (xr.DataArray): Measurement covariance.
        target_emissions (xr.DataArray): Target emissions.

    Returns:
        xr.Dataset: Inversion result.
    """
    site_selection = measurements.unstack().MPlace.isin(sites)
    measurements, measurement_covariance, footprints = _select_sites(
        measurements, measurement_covariance, footprints, sites
    )
    state_coordinates = prior_emissions.coords
    solver = _compute_inversion(
        loss_class=BayesianYM,
        solver_class=BayesianAnalyticalYM,
        prior=prior_emissions.values,
        prior_standard_deviation=prior_standard_deviation.values,
        prior_temporal_correlation=prior_temporal_correlation.values,
        prior_spatial_correlation=prior_spatial_correlation.values,
        forward_model=footprints.data,
        measurement=measurements.values,
        measurement_covariance=measurement_covariance.values,
    )
    posterior_emissions, prior_standard_deviations = solver()
    posterior_emissions = xr.DataArray(
        posterior_emissions, coords=state_coordinates
    ).unstack()
    posterior_std = xr.DataArray(
        prior_standard_deviations, coords=state_coordinates
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
    )
    return inversion_result


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
    if "sector" in prior_loader.prior.dims:
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
    # load config yaml
    with args.config.open("r") as f:
        config = yaml.safe_load(f)

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
    # Start osses
    for i, mplace_values in tqdm(
        enumerate(mplace_value_permutations), total=len(mplace_value_permutations)
    ):
        if "start_index" in config:
            if i < config["start_index"]:
                continue

        inversion_result = _run_inversion_ym(
            mplace_values,
            prior_emissions,
            prior_standard_deviation,
            prior_temporal_correlation,
            prior_spatial_correlation,
            footprints,
            measurements,
            measurement_covariance,
            target_emissions,
        )
        inversion_result.to_netcdf(output_buffer_path / f"permutation_{i}.nc")

    client.restart(wait_for_workers=True)

    # Combine all permutations and save the result
    xr.open_mfdataset(
        output_buffer_path.glob("permutation_*.nc"),
        combine="nested",
        concat_dim="permutation",
    ).to_netcdf(output_dir / output_name)

    # Delete the buffer files
    for file in output_buffer_path.glob("permutation_*.nc"):
        file.unlink()

    output_buffer_path.rmdir()
    client.close()


if __name__ == "__main__":
    args = _get_args()
    main(args)
