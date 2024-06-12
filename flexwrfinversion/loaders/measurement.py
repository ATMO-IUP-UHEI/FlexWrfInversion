""" This module descibes classes that can be used to load measurement data for an
     inversion."""

from abc import ABC, abstractmethod

import numpy as np
import xarray as xr

from flexwrfinversion.loaders.footprint import (
    FootprintLoader,
    LoadFootprintForTotalInCity,
)
from flexwrfinversion.loaders.target import TargetLoader, TargetLoaderTotalInCity


class MeasurementLoader(ABC):
    @abstractmethod
    def __init__(
        self,
        target_loader: TargetLoader,
        footprint_loader: FootprintLoader,
        *args,
        **kwargs,
    ):
        pass

    @property
    @abstractmethod
    def measurements(self) -> xr.DataArray:
        """Load the measurement data
        Returns:
            xr.DataArray: The measurement data as 1D array. Coordinates should be stacked
                beforehand.
        """
        pass

    @abstractmethod
    def load_timeframe(
        self, start_time: np.datetime64, end_time: np.datetime64
    ) -> xr.DataArray:
        """Load the prior data
        Returns:
            xr.DataArray: The prior data as 1D array. Coordinates should be stacked
                beforehand.
        """
        pass


class MeasurementFromFile(MeasurementLoader):
    def __init__(
        self,
        target_loader: TargetLoaderTotalInCity,
        footprint_loader: LoadFootprintForTotalInCity,
    ):
        if not isinstance(target_loader, TargetLoaderTotalInCity):
            raise ValueError("target must be an instance of TargetLoaderTotalInCity")

        self.target_loader = target_loader
        self.footprint_loader = footprint_loader
        self._remapped_data_path = target_loader._remapped_data_path
        self._season = target_loader._season
        self._city = target_loader._city
        self._prior_type = target_loader._prior_type
        self._city_suffixes = target_loader._city_suffixes
        self._germany_suffixes = target_loader._germany_suffixes
        self._measurements = None

    @property
    def measurements(self):
        if self._measurements is None:
            city_folder = (
                self._remapped_data_path / self._season / self._city / self._prior_type
            )
            germany_folder = (
                self._remapped_data_path / self._season / "germany" / self._prior_type
            )
            true_concentrations_city = xr.open_dataset(
                city_folder / f"true_concentrations{self._city_suffixes[0]}.nc",
                chunks="auto",
            ).drop_dims(["x_stag", "y_stag"])
            true_concentrations_germany = xr.open_dataset(
                germany_folder / f"true_concentrations{self._germany_suffixes[0]}.nc",
                chunks="auto",
            ).drop_dims(["x_stag", "y_stag"])

            true_concentrations_sums_city = xr.open_dataset(
                city_folder / f"true_concentrations{self._city_suffixes[1]}.nc",
                chunks="auto",
            ).drop_dims(["x_stag", "y_stag"])
            true_concentrations_sums_germany = xr.open_dataset(
                germany_folder / f"true_concentrations{self._germany_suffixes[1]}.nc",
                chunks="auto",
            ).drop_dims(["x_stag", "y_stag"])

            self._measurements = (
                xr.merge(
                    [
                        true_concentrations_city,
                        true_concentrations_sums_city,
                    ],
                )
                + xr.merge(
                    [
                        true_concentrations_germany,
                        true_concentrations_sums_germany,
                    ],
                )
            )[self.target_loader.TOTAL_EMISSION_KEY].stack(
                measurement=self.footprint_loader.MEASUREMENT_DIMS
            )
        return self._measurements

    def load_timeframe(
        self, start_time: np.datetime64, end_time: np.datetime64
    ) -> xr.DataArray:
        if self._measurements is None:
            self.measurements
        return (
            self._measurements.unstack()
            .sel(MTime=slice(start_time, end_time))
            .stack(measurement=self.footprint_loader.MEASUREMENT_DIMS)
        )
