""" This module descibes classes that can be used to load measurement data for an
     inversion."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import xarray as xr

from flexwrfinversion.loaders.footprint import (
    FlexibleFootprintLoaderAnthBioCo,
    FlexibleFootprintLoaderTotal,
    FootprintLoader,
    LoadFootprintAnthBioCO,
    LoadFootprintForTotalInCity,
)
from flexwrfinversion.loaders.target import (
    FlexibleTargetLoaderTotal,
    TargetLoader,
    TargetLoaderAnthAndBioSectors,
    TargetLoaderAnthBioCO,
    TargetLoaderTotalInCity,
)


class MeasurementLoader(ABC):
    @abstractmethod
    def __init__(
        self,
        target_loader: TargetLoader,
        footprint_loader: FootprintLoader,
        *args,
        **kwargs,
    ):
        self.target_loader = target_loader
        self.footprint_loader = footprint_loader

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
        """Load part of the measurements with respect measurement time.

        Args:
            start_time (np.datetime64): Start time of the measurement timeframe
            end_time (np.datetime64): End time of the measurement timeframe

        Returns:
            xr.DataArray: Measurements of given timeframe.
        """
        pass


class MeasurementFromFile(MeasurementLoader):
    TOTAL_EMISSION_KEY = "CO2_TOTAL"

    def __init__(
        self,
        target_loader: TargetLoaderTotalInCity | TargetLoaderAnthAndBioSectors,
        footprint_loader: LoadFootprintForTotalInCity,
    ):
        if not isinstance(
            target_loader, (TargetLoaderTotalInCity, TargetLoaderAnthAndBioSectors)
        ):
            raise ValueError("target must be an instance of TargetLoaderTotalInCity")

        super().__init__(target_loader, footprint_loader)
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
                (
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
                )[self.TOTAL_EMISSION_KEY]
                .stack(measurement=self.footprint_loader.MEASUREMENT_DIMS)
                .astype(np.float32)
                .compute()
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


class MeasurementFromFileCO(MeasurementLoader):
    TOTAL_CONCENTRATION_KEY = "CO2_TOTAL"
    CO_CONCENTRATION_KEY = "E_CO"

    def __init__(
        self,
        target_loader: TargetLoaderAnthBioCO,
        footprint_loader: LoadFootprintAnthBioCO,
    ):
        if not isinstance(target_loader, (TargetLoaderAnthBioCO)):
            raise ValueError("target must be an instance of TargetLoaderTotalInCity")

        if not isinstance(footprint_loader, (LoadFootprintAnthBioCO)):
            raise ValueError("footprint must be an instance of LoadFootprintAnthBioCO")
        super().__init__(target_loader, footprint_loader)
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

            measurements = (
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
            )[[self.TOTAL_CONCENTRATION_KEY, self.CO_CONCENTRATION_KEY]]

            self._measurements = (
                xr.concat(
                    [
                        measurements[self.TOTAL_CONCENTRATION_KEY].expand_dims(
                            species=["CO2"],
                        ),
                        measurements[self.CO_CONCENTRATION_KEY].expand_dims(
                            species=["CO"],
                        ),
                    ],
                    dim="species",
                )
                .sortby("species")
                .stack(measurement=self.footprint_loader.MEASUREMENT_DIMS)
                .astype(np.float32)
                .compute()
            )
        return self._measurements

    def load_timeframe(
        self, start_time: np.datetime64, end_time: np.datetime64
    ) -> xr.DataArray:
        return (
            self.measurements.unstack()
            .sel(MTime=slice(start_time, end_time))
            .stack(measurement=self.footprint_loader.MEASUREMENT_DIMS)
        )


class FlexibleMeasurementLoaderTotal(MeasurementLoader):
    def __init__(
        self,
        target_loader: FlexibleTargetLoaderTotal,
        footprint_loader: FlexibleFootprintLoaderTotal,
        measurement_file_city: str | Path,
        measurement_file_germany: str | Path,
        leave_out: list[str] = None,
        keep_only: list[str] = None,
    ):
        """Flexible implementation of measurement loader to load the total CO2
        measurements directly from files.

        Args:
            target_loader (FlexibleTargetLoaderTotal): Target loader used in the
                 inversion.
            footprint_loader (FlexibleFootprintLoaderTotal): Footprint loader used in the
                 inversion.
            measurement_file_city (str | Path): Measurement/concentration file for the
                 city that contains the `CO2_TOTAL` field.
            measurement_file_germany (str | Path): Measurement/concentration file for
                 germany that contains the `CO2_TOTAL` field.
            leave_out (list[str], optional): List of names of stations to exclude for the
                 runs. Defaults to None.
            keep_only (list[str], optional): List of names of station to only include
                 these. Defaults to None.
        """
        super().__init__(target_loader, footprint_loader)
        self._measurement_file_city = measurement_file_city
        self._measurement_file_germany = measurement_file_germany
        if leave_out is not None and keep_only is not None:
            raise ValueError("leave_out and keep_only cannot be used together.")
        elif leave_out is not None:
            leave_out = np.char.encode(np.array(leave_out, dtype=str))
        elif keep_only is not None:
            keep_only = np.char.encode(np.array(keep_only, dtype=str))
        self._leave_out = leave_out
        self._keep_only = keep_only
        self._measurements = None
        self._unstacked_measurements = None

    @property
    def measurements(self):
        if self._measurements is None:
            measurements_city = xr.open_dataset(self._measurement_file_city)[
                self.target_loader.TOTAL_EMISSION_KEY
            ]
            measurements_germany = xr.open_dataset(self._measurement_file_germany)[
                self.target_loader.TOTAL_EMISSION_KEY
            ]
            self._measurements = measurements_city + measurements_germany
            if self._leave_out is not None:
                self._measurements = self._measurements.isel(
                    MPlace=~np.isin(self._measurements.MPlace.values, self._leave_out)
                )
            if self._keep_only is not None:
                self._measurements = self._measurements.sel(MPlace=self._keep_only)
            self._measurements = (
                self._measurements.stack(
                    measurement=self.footprint_loader.MEASUREMENT_DIMS
                )
                .astype(np.float32)
                .compute()
            )

        return self._measurements

    @property
    def unstacked_measurements(self) -> xr.DataArray:
        """Measurements in original shape.

        Returns:
            xr.DataArray: Measurements.
        """
        if self._unstacked_measurements is None:
            self._unstacked_measurements = self.measurements.unstack()
        return self._unstacked_measurements

    def load_timeframe(
        self, start_time: np.datetime64, end_time: np.datetime64
    ) -> xr.DataArray:
        return self.unstacked_measurements.sel(MTime=slice(start_time, end_time)).stack(
            measurement=self.footprint_loader.MEASUREMENT_DIMS
        )


class FlexibleMeasurementLoaderTotalCo(MeasurementLoader):
    TOTAL_EMISSION_KEY = "CO2_TOTAL"

    def __init__(
        self,
        target_loader: TargetLoader,
        footprint_loader: FlexibleFootprintLoaderAnthBioCo,
        measurement_file_city_co2: str | Path,  # THIS IS THE LAST THING THAT I ADDED
        measurement_file_city_co: str | Path,
        measurement_file_germany_co2: str | Path,
        measurement_file_germany_co: str | Path,
        leave_out: list[str] = None,
        keep_only: list[str] = None,
    ):
        """Flexible implementation of measurement loader to load the total CO2
        measurements directly from files.

        Args:
            target_loader (FlexibleTargetLoaderTotal): Target loader used in the
                 inversion.
            footprint_loader (FlexibleFootprintLoaderAnthBioCo): Footprint loader used in
                 the inversion.
            measurement_file_city (str | Path): Measurement/concentration file for the
                 city that contains the `CO2_TOTAL` and `E_CO` field.
            measurement_file_germany (str | Path): Measurement/concentration file for
                 germany that contains the `CO2_TOTAL` and `E_CO` field.
            leave_out (list[str], optional): List of names of stations to exclude for the
                 runs. Defaults to None.
            keep_only (list[str], optional): List of names of station to only include
                 these. Defaults to None.
        """
        super().__init__(target_loader, footprint_loader)
        self._measurement_file_city_co2 = measurement_file_city_co2
        self._measurement_file_city_co = measurement_file_city_co
        self._measurement_file_germany_co2 = measurement_file_germany_co2
        self._measurement_file_germany_co = measurement_file_germany_co
        if leave_out is not None and keep_only is not None:
            raise ValueError("leave_out and keep_only cannot be used together.")
        elif leave_out is not None:
            leave_out = np.char.encode(np.array(leave_out, dtype=str))
        elif keep_only is not None:
            keep_only = np.char.encode(np.array(keep_only, dtype=str))
        self._leave_out = leave_out
        self._keep_only = keep_only
        self._measurements = None
        self._unstacked_measurements = None

    @property
    def measurements(self):
        if self._measurements is None:
            co2_measurements = (
                xr.open_dataset(self._measurement_file_city_co2)[
                    self.TOTAL_EMISSION_KEY
                ]
                + xr.open_dataset(self._measurement_file_germany_co2)[
                    self.TOTAL_EMISSION_KEY
                ]
            ).expand_dims(species=["CO2"])

            co_measurements = (
                xr.open_dataset(self._measurement_file_city_co)[
                    self.footprint_loader.CO_SECTOR_KEY
                ]
                + xr.open_dataset(self._measurement_file_germany_co)[
                    self.footprint_loader.CO_SECTOR_KEY
                ]
            ).expand_dims(species=["CO"])

            measurements = xr.concat(
                [
                    co2_measurements,
                    co_measurements,
                ],
                dim="species",
            )

            if self._leave_out is not None:
                measurements = measurements.isel(
                    MPlace=~np.isin(measurements.MPlace.values, self._leave_out)
                )
            if self._keep_only is not None:
                measurements = measurements.sel(MPlace=self._keep_only)

            self._measurements = (
                measurements.sortby("species")
                .stack(measurement=self.footprint_loader.MEASUREMENT_DIMS)
                .astype(np.float32)
                .compute()
            )

        return self._measurements

    @property
    def unstacked_measurements(self) -> xr.DataArray:
        """Measurements in original shape.

        Returns:
            xr.DataArray: Measurements.
        """
        if self._unstacked_measurements is None:
            self._unstacked_measurements = self.measurements.unstack()
        return self._unstacked_measurements

    def load_timeframe(
        self, start_time: np.datetime64, end_time: np.datetime64
    ) -> xr.DataArray:
        return self.unstacked_measurements.sel(MTime=slice(start_time, end_time)).stack(
            measurement=self.footprint_loader.MEASUREMENT_DIMS
        )
