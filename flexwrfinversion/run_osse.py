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
```
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from pyinverse.loss import Bayesian
from pyinverse.solver import BayesianAnalytical
from tqdm.auto import tqdm

# flake8: noqa
from flexwrfinversion.loaders.footprint import (
    FlexibleFootprintLoaderTotal,
    FootprintLoader,
    LoadFootprintAnthAndBioSectors,
    LoadFootprintAnthBioCO,
    LoadFootprintForTotalInCity,
)
from flexwrfinversion.loaders.measurement import (
    FlexibleMeasurementLoaderTotal,
    MeasurementFromFile,
    MeasurementFromFileCO,
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
    ShiftToBiospheric,
)
from flexwrfinversion.loaders.prior_covariance import (
    PriorCovarianceLoader,
    RelativeErrorWithSpatialCorrelation,
    TargetAsErrorNoCorrelation,
    TargetAsErrorWithCO_Correlation,
)
from flexwrfinversion.loaders.target import (
    FlexibleTargetLoaderTotal,
    TargetLoader,
    TargetLoaderAnthAndBioSectors,
    TargetLoaderAnthBioCO,
    TargetLoaderTotalInCity,
)

# flake8: noqa
TIME_BUFFER_FOR_INVERSION_WINDOW = np.timedelta64(24, "h")


def _get_args():
    parser = ArgumentParser(description="Run OSSE")
    parser.add_argument("config", type=Path, help="Path to the configuration file")
    return parser.parse_args()


def _run_inversion(
    dates: np.ndarray,
    sites: np.ndarray,
    prior_loader: PriorLoader,
    prior_covariance_loader: PriorCovarianceLoader,
    target_loader: TargetLoader,
    footprint_loader: FootprintLoader,
    measurement_loader: MeasurementLoader,
    measurement_covariance_loader: MeasurementCovarianceLoader,
):
    site_selection = measurement_loader.measurements.unstack().MPlace.isin(sites)

    prior_std = []
    posterior_emissions = []
    posterior_std = []
    averaging_kernel_diag = []
    averaging_kernel_sum = []

    for i, (start_date, end_date) in tqdm(
        enumerate(zip(dates[:-1], dates[1:])), total=len(dates) - 1
    ):
        emission_start_time = start_date - TIME_BUFFER_FOR_INVERSION_WINDOW
        emission_end_time = end_date + TIME_BUFFER_FOR_INVERSION_WINDOW
        measurement_start_time = start_date
        measurement_end_time = end_date + TIME_BUFFER_FOR_INVERSION_WINDOW

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

        measurements = measurements.where(measurements.MPlace.isin(sites), drop=True)
        measurement_covariance = measurement_covariance.where(
            measurement_covariance.MPlace0.isin(sites), drop=True
        ).where(measurement_covariance.MPlace1.isin(sites), drop=True)
        footprint = footprint.where(footprint.MPlace.isin(sites), drop=True)

        state_coordinates = prior_emissions.coords

        loss = Bayesian(
            x_prior=prior_emissions.values,
            cov_prior=prior_emission_covariance.values,
            y=measurements.values,
            cov_y=measurement_covariance.values,
            K=footprint.values.T,
        )

        solver = BayesianAnalytical(loss)

        partial_posterior_emissions, partial_posterior_covariance = solver()

        partial_posterior_emissions = xr.DataArray(
            partial_posterior_emissions, coords=state_coordinates
        ).unstack()
        partial_posterior_std = xr.DataArray(
            np.sqrt(np.diag(partial_posterior_covariance)), coords=state_coordinates
        ).unstack()
        partial_averaging_kernel_diag = xr.DataArray(
            np.diag(solver.averaging_kernel),
            coords=state_coordinates,
        ).unstack()
        partial_averaging_kernel_sum = (
            xr.DataArray(
                solver.averaging_kernel,
                coords=prior_emission_covariance.coords,
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

        prior_std.append(
            prior_covariance_loader.prior_std.unstack().sel(
                Time=slice(start_date, end_date - np.timedelta64(1, "h"))
            )
        )
        posterior_emissions.append(
            partial_posterior_emissions.sel(
                Time=slice(start_date, end_date - np.timedelta64(1, "h"))
            )
        )
        posterior_std.append(
            partial_posterior_std.sel(
                Time=slice(start_date, end_date - np.timedelta64(1, "h"))
            )
        )
        averaging_kernel_diag.append(
            partial_averaging_kernel_diag.sel(
                Time=slice(start_date, end_date - np.timedelta64(1, "h"))
            )
        )
        averaging_kernel_sum.append(partial_averaging_kernel_sum)

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


def main(args):
    # load config yaml
    with args.config.open("r") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["output_dir"])
    output_name = config["output_name"]
    output_buffer_path = output_dir / output_name.split(".")[0]
    output_buffer_path.mkdir(exist_ok=True, parents=True)

    # initialize prior loader based on strin in config
    target_loader: TargetLoader = eval(config["target"]["target_loader"])(
        **config["target"]["kwargs"]
    )
    prior_loader: PriorLoader = eval(config["prior"]["prior_loader"])(
        target_loader, **config["prior"]["kwargs"]
    )
    prior_covariance_loader: PriorCovarianceLoader = eval(
        config["prior_covariance"]["prior_covariance_loader"]
    )(prior_loader, **config["prior_covariance"]["kwargs"])

    footprint_loader: FootprintLoader = eval(config["footprint"]["footprint_loader"])(
        **config["footprint"]["kwargs"]
    )
    measurement_loader: MeasurementLoader = eval(
        config["measurement"]["measurement_loader"]
    )(target_loader, footprint_loader, **config["measurement"]["kwargs"])
    measurement_covariance_loader: MeasurementCovarianceLoader = eval(
        config["measurement_covariance"]["measurement_covariance_loader"]
    )(measurement_loader, **config["measurement_covariance"]["kwargs"])

    for i in tqdm(range(config["n_permutations"])):
        mplace_values = np.random.choice(
            measurement_loader.measurements.unstack().MPlace,
            config["n_stations"],
            replace=False,
        )

        dates = np.arange(
            prior_loader.prior.Time[0].values,
            prior_loader.prior.Time[-1].values + np.timedelta64(1, "D"),
            dtype="datetime64[D]",
        )

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

        inversion_result.to_netcdf(output_buffer_path / f"permutation_{i}.nc")

    xr.open_mfdataset(
        output_buffer_path.glob("permutation_*.nc"),
        combine="nested",
        concat_dim="permutation",
    ).to_netcdf(output_dir / output_name)

    for file in output_buffer_path.glob("permutation_*.nc"):
        file.unlink()

    output_buffer_path.rmdir()


if __name__ == "__main__":
    args = _get_args()
    main(args)
