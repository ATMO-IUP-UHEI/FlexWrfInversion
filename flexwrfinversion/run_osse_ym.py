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

import time
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from dask.distributed import Client
from loguru import logger
from pyinverse.loss import Bayesian, BayesianYM
from pyinverse.solver import BayesianAnalytical, BayesianAnalyticalYM
from tqdm.auto import tqdm

# flake8: noqa
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
)
from flexwrfinversion.loaders.measurement_covariance import (
    ConstantNoCorrelation,
    ConstantNoCorrelationCO,
    MeasurementCovarianceLoader,
)
from flexwrfinversion.loaders.prior import (
    FlatPrior,
    FlexiblePriorLoaderTotal_ShiftToBiospheric,
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
from flexwrfinversion.run_osse import _get_args, get_kwargs

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
):
    tic = time.time()
    site_selection = measurements.unstack().MPlace.isin(sites)

    measurements = measurements.isel(measurement=measurements.MPlace.isin(sites))
    measurement_covariance = measurement_covariance.isel(
        measurement0=measurement_covariance.MPlace0.isin(sites),
        measurement1=measurement_covariance.MPlace1.isin(sites),
    )
    footprints = footprints.isel(
        measurement=footprints.MPlace.isin(sites),
    )
    state_coordinates = prior_emissions.coords
    toc = time.time()
    logger.info(f"Time for filtering data: {toc-tic}", flush=True)

    tic = time.time()
    loss = BayesianYM(
        prior=prior_emissions.values,
        prior_standard_deviation=prior_standard_deviation.values,
        prior_temporal_correlation=prior_temporal_correlation.values,
        prior_spatial_correlation=prior_spatial_correlation.values,
        forward_model=footprints.data,
        measurement=measurements.values,
        measurement_covariance=measurement_covariance.values,
    )
    solver = BayesianAnalyticalYM(loss)
    posterior_emissions, prior_standard_deviations = solver()
    toc = time.time()
    logger.info(f"Time for setup and inversion: {toc-tic}", flush=True)
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


def restack_coords(
    dataarray: xr.DataArray, unstack_dims, stack_dict=dict(), dim_order=[]
):
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

    # client = Client()

    # initialize loaders based on the config
    target_loader: TargetLoader = eval(config["target"]["target_loader"])(
        **get_kwargs(config, "target")
    )
    prior_loader: PriorLoader = eval(config["prior"]["prior_loader"])(
        target_loader, **get_kwargs(config, "prior")
    )
    prior_covariance_loader: PriorCovarianceLoader = eval(
        config["prior_covariance"]["prior_covariance_loader"]
    )(prior_loader, **get_kwargs(config, "prior_covariance"))

    footprint_loader: FootprintLoader = eval(config["footprint"]["footprint_loader"])(
        **get_kwargs(config, "footprint")
    )
    measurement_loader: MeasurementLoader = eval(
        config["measurement"]["measurement_loader"]
    )(target_loader, footprint_loader, **get_kwargs(config, "measurement"))
    measurement_covariance_loader: MeasurementCovarianceLoader = eval(
        config["measurement_covariance"]["measurement_covariance_loader"]
    )(measurement_loader, **get_kwargs(config, "measurement_covariance"))

    # select station permutations
    logger.info("Setting up permutations", flush=True)
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
    logger.info("Setting up restacking", flush=True)
    dims_to_stack = dict()
    state_order = ["Time", "subsector"]
    footprint_order = ["measurement", "Time", "subsector"]
    prior_spatial_correlation_order = ["subsector0", "subsector1"]
    prior_spatial_correlation_dims_to_stack = dict()
    if "sector" in prior_loader.prior.coords:
        dims_to_stack["subsector_sector"] = ["subsector", "sector"]
        state_order = ["Time", "subsector_sector"]
        footprint_order = ["measurement", "Time", "subsector_sector"]
        prior_spatial_correlation_order = ["subsector0_sector0", "subsector1_sector1"]
        prior_spatial_correlation_dims_to_stack = dict(
            subsector0_sector0=["subsector0", "sector0"],
            subsector1_sector1=["subsector1", "sector1"],
        )

    logger.info("Loading Prior values and covariance", flush=True)
    prior_emissions = prior_loader.prior.astype(FLOAT_PRECISION)
    prior_standard_deviation = prior_covariance_loader.prior_std.astype(FLOAT_PRECISION)
    prior_temporal_correlation = prior_covariance_loader.temporal_correlation.astype(
        FLOAT_PRECISION
    )
    prior_spatial_correlation = prior_covariance_loader.spatial_correlation.astype(
        FLOAT_PRECISION
    )
    logger.info("Loading Footprints values", flush=True)
    footprints = footprint_loader.footprint.astype(FLOAT_PRECISION)
    logger.info("Loading Target values", flush=True)
    target_emissions = target_loader.target.astype(FLOAT_PRECISION)
    if "sector" in prior_loader.prior.coords:
        logger.info("Combining spatial and sector correlation", flush=True)
        prior_spatial_correlation = (
            prior_spatial_correlation
            * prior_covariance_loader.sector_correlation.astype(FLOAT_PRECISION)
        )

    logger.info("Restacking data", flush=True)
    prior_emissions = restack_coords(
        prior_emissions, ["state"], dims_to_stack, state_order
    )
    prior_standard_deviation = restack_coords(
        prior_standard_deviation, ["state"], dims_to_stack, state_order
    )
    target_emissions = restack_coords(
        target_emissions, ["state"], dims_to_stack, state_order
    )
    footprints = restack_coords(footprints, ["state"], dims_to_stack, footprint_order)
    prior_spatial_correlation = restack_coords(
        prior_spatial_correlation,
        [],
        prior_spatial_correlation_dims_to_stack,
        prior_spatial_correlation_order,
    )

    logger.info("Loading Measurement values", flush=True)
    measurements = measurement_loader.measurements.astype(FLOAT_PRECISION).copy()
    logger.info("Loading Measurement covariance", flush=True)
    measurement_covariance = (
        measurement_covariance_loader.load_timeframe(
            measurement_loader.measurements.MTime[0],
            measurement_loader.measurements.MTime[-1],
        )
        .astype(FLOAT_PRECISION)
        .copy()
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

    # client.restart(wait_for_workers=True)

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
    # client.close()


if __name__ == "__main__":
    args = _get_args()
    main(args)
