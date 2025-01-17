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
)
from flexwrfinversion.loaders.target import FlexibleTargetLoaderTotal, TargetLoader


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


class FlexibleMeasurementLoaderTotal(MeasurementLoader):
    def __init__(
        self,
        target_loader: FlexibleTargetLoaderTotal,
        footprint_loader: FlexibleFootprintLoaderTotal,
        measurement_file_city: str | Path,
        measurement_file_germany: str | Path,
        leave_out: list[str] = None,
        keep_only: list[str] = None,
        times_of_day: list[int] = None,
        ppm_noise: float = None,
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
            times_of_day (list[int], optional): List of times of day to include in the
                 measurements. Defaults to None.
            ppm_noise (bool, optional): Standard deviation of noise to add in ppm.
                 Defaults to None.
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
        self._times_of_day = times_of_day
        self._ppm_noise = ppm_noise
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
            if self._times_of_day is not None:
                self._measurements = self._measurements.isel(
                    MTime=self._measurements.MTime.dt.hour.isin(self._times_of_day)
                )
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
        time_frame_measurements = self.unstacked_measurements.sel(
            MTime=slice(start_time, end_time)
        ).stack(measurement=self.footprint_loader.MEASUREMENT_DIMS)
        if self._ppm_noise is not None:
            time_frame_measurements = time_frame_measurements + np.random.normal(
                scale=self._ppm_noise * 1e-6, size=time_frame_measurements.shape
            )
        return time_frame_measurements


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
        times_of_day: list[int] = None,
        ppm_noise: float = None,
        ppb_noise: float = None,
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
            times_of_day (list[int], optional): List of times of day to include in the
                 measurements. Defaults to None.
            ppm_noise (bool, optional): Standard deviation of noise to add in ppm for CO2.
                 Defaults to None.
            ppb_noise (bool, optional): Standard deviation of noise to add in ppb for CO.
                 Defaults to None.
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
        self._times_of_day = times_of_day
        self._ppm_noise = ppm_noise
        self._ppb_noise = ppb_noise
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

            if self._times_of_day is not None:
                measurements = measurements.isel(
                    MTime=measurements.MTime.dt.hour.isin(self._times_of_day)
                )

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
        time_frame_measurements = self.unstacked_measurements.sel(
            MTime=slice(start_time, end_time)
        ).stack(measurement=self.footprint_loader.MEASUREMENT_DIMS)
        if self._ppm_noise is not None or self._ppb_noise is not None:
            noise = xr.zeros_like(time_frame_measurements)
            ppm_noise = self._ppm_noise * 1e-6 if self._ppm_noise is not None else 0
            ppb_noise = self._ppb_noise * 1e-9 if self._ppb_noise is not None else 0
            noise = xr.where(noise.species == "CO2", ppm_noise, ppb_noise)
            time_frame_measurements = time_frame_measurements + np.random.normal(
                scale=noise, size=time_frame_measurements.shape
            )
        return time_frame_measurements
