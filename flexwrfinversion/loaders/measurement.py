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

FLOAT_PRECISION = np.float32


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

    @staticmethod
    def _select_measurements(
        measurements: xr.DataArray,
        leave_out: list[str] = None,
        keep_only: list[str] = None,
        times_of_day: list[int] = None,
    ):
        """Select measurements from the measurements

        Args:
            measurements (xr.DataArray): measurements to select measurements from
            keep_only (list[str], optional): List of names of stations to only include
                 these. Defaults to None.
            leave_out (list[str], optional): List of names of stations to exclude for the
                 runs. Defaults to None.
            times_of_day (list[int], optional): List of hours of the day to include in the
                 data. Defaults to None.

        Returns:
            xr.DataArray: Selected measurements
        """
        if leave_out is not None:
            measurements = measurements.isel(
                MPlace=~np.isin(measurements.MPlace.values, leave_out)
            )
        if keep_only is not None:
            measurements = measurements.sel(MPlace=keep_only)

        if times_of_day is not None:
            measurements = measurements.isel(
                MTime=measurements.MTime.dt.hour.isin(times_of_day)
            )
        return measurements


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
        total_sector_key: str = "CO2_TOTAL",
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
            total_sector_key (str, optional): Key for the total emission in the
                    measurement files. Defaults to "CO2_TOTAL".
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
        self.total_sector_key = total_sector_key
        self._measurements = None
        self._unstacked_measurements = None

    @property
    def measurements(self):
        if self._measurements is None:
            measurements_city = xr.open_dataset(self._measurement_file_city)[
                self.total_sector_key
            ]
            measurements_germany = xr.open_dataset(self._measurement_file_germany)[
                self.total_sector_key
            ]
            self._measurements = measurements_city + measurements_germany
            self._measurements = self._select_measurements(
                self._measurements,
                self._leave_out,
                self._keep_only,
                self._times_of_day,
            )
            self._measurements = (
                self._measurements.stack(
                    measurement=self.footprint_loader.MEASUREMENT_DIMS
                )
                .astype(FLOAT_PRECISION)
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
    total_sector_key = "CO2_TOTAL"

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
        total_sector_key: str = "CO2_TOTAL",
        co_sector_key: str = "E_CO",
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
            total_sector_key (str, optional): Key for the total emission in the
                 measurement files. Defaults to "CO2_TOTAL".
            co_sector_key (str, optional): Key for the CO emission in the
                 measurement files. Defaults to "E_CO".
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
        self.total_sector_key = total_sector_key
        self.co_sector_key = co_sector_key
        self._measurements = None
        self._unstacked_measurements = None

    @property
    def measurements(self):
        if self._measurements is None:
            co2_measurements = (
                xr.open_dataset(self._measurement_file_city_co2)[self.total_sector_key]
                + xr.open_dataset(self._measurement_file_germany_co2)[
                    self.total_sector_key
                ]
            ).expand_dims(species=["CO2"])

            co_measurements = (
                xr.open_dataset(self._measurement_file_city_co)[self.co_sector_key]
                + xr.open_dataset(self._measurement_file_germany_co)[self.co_sector_key]
            ).expand_dims(species=["CO"])

            measurements = xr.concat(
                [
                    co2_measurements,
                    co_measurements,
                ],
                dim="species",
            )

            self._measurements = self._select_measurements(
                measurements,
                self._leave_out,
                self._keep_only,
                self._times_of_day,
            )

            self._measurements = (
                self._measurements.sortby("species")
                .stack(measurement=self.footprint_loader.MEASUREMENT_DIMS)
                .astype(FLOAT_PRECISION)
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


class MeasurementLoaderFromSingleFileTotal(MeasurementLoader):
    """Just use one file to load all necessary measurements."""

    def __init__(
        self,
        target_loader: TargetLoader,
        footprint_loader: FootprintLoader,
        measurement_file: str | Path,
        leave_out: list[str] = None,
        keep_only: list[str] = None,
        times_of_day: list[int] = None,
        ppm_noise: float = None,
        total_sector_key: str = "CO2_TOTAL",
    ):
        """Flexible implementation of measurement loader to load the total CO2
        measurements directly from files.

        Args:
            target_loader (FlexibleTargetLoaderTotal): Target loader used in the
                 inversion.
            footprint_loader (FlexibleFootprintLoaderTotal): Footprint loader used in the
                 inversion.
            measurement_file (str | Path): Measurement/concentration file that contains
                 the `CO2_TOTAL` field.
            leave_out (list[str], optional): List of names of stations to exclude for the
                 runs. Defaults to None.
            keep_only (list[str], optional): List of names of station to only include
                 these. Defaults to None.
            times_of_day (list[int], optional): List of times of day to include in the
                 measurements. Defaults to None.
            ppm_noise (bool, optional): Standard deviation of noise to add in ppm.
                 Defaults to None.
            total_sector_key (str, optional): Key for the total emission in the
                    measurement files. Defaults to "CO2_TOTAL".
        """
        super().__init__(target_loader, footprint_loader)
        self._measurement_file = measurement_file
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
        self.total_sector_key = total_sector_key
        self._measurements = None
        self._unstacked_measurements = None

    @property
    def measurements(self):
        if self._measurements is None:
            measurements = xr.open_dataset(self._measurement_file)[
                self.total_sector_key
            ]
            self._measurements = self._select_measurements(
                measurements,
                self._leave_out,
                self._keep_only,
                self._times_of_day,
            )
            self._measurements = (
                self._measurements.stack(
                    measurement=self.footprint_loader.MEASUREMENT_DIMS
                )
                .astype(FLOAT_PRECISION)
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


class MeasurementLoaderTotalAndWeeklyCO2_ff(FlexibleMeasurementLoaderTotal):
    """Measurement loader that contains the total CO2 and the weekly integrated CO2_ff
    data e.g. from 14C measurements.
    """

    CO2_FF_MPLACE_NAME = "weekly_co2_ff"

    def __init__(
        self,
        target_loader: FlexibleTargetLoaderTotal,
        footprint_loader: FlexibleFootprintLoaderTotal,
        measurement_file_city: str | Path,
        measurement_file_germany: str | Path,
        measurement_file_weekly_co2_ff: str | Path,
        leave_out: list[str] = None,
        keep_only: list[str] = None,
        times_of_day: list[int] = None,
        ppm_noise: float = None,
        total_sector_key: str = "CO2_TOTAL",
        weekly_co2_ff_sector_key: str = "CO2_FF",
    ):
        """Measurement loader that contains the total CO2 and the weekly integrated CO2_ff
        data e.g. from 14C measurements.

        Args:
            target_loader (FlexibleTargetLoaderTotal): Target loader used in the
                 inversion.
            footprint_loader (FlexibleFootprintLoaderTotal): Footprint loader used in the
                 inversion.
            measurement_file_city (str | Path): Measurement/concentration file for the
                 city that contains the `CO2_TOTAL` field.
            measurement_file_germany (str | Path): Measurement/concentration file for
                 Germany.
            measurement_file_weekly_co2_ff (str | Path): Measurement/concentration file
                 for the weekly integrated CO2_ff data. Expects a field with the weekly
                 integrated CO2_ff data with dimension `measurement_id` and a coordinate
                 containing time information in the dimension "MTime". Exects data
                 variable specified in `weekly_co2_ff_sector_key`.
            leave_out (list[str], optional): List of measurement IDs to exclude.
                 Defaults to None.
            keep_only (list[str], optional): List of measurement IDs to include.
                 Defaults to None.
            times_of_day (list[int], optional): List of times of day to include.
                 Defaults to None.
            ppm_noise (float, optional): Standard deviation of the noise to be added to
                 the measurements in parts per million (ppm). Defaults to None.
            total_sector_key (str, optional): The key for the total CO2 sector. Defaults
                 to "CO2_TOTAL".
            weekly_co2_ff_sector_key (str, optional): The key for the weekly integrated
                 CO2_ff sector. Defaults to "CO2_FF".
        """
        super().__init__(
            target_loader,
            footprint_loader,
            measurement_file_city,
            measurement_file_germany,
            leave_out,
            keep_only,
            times_of_day,
            ppm_noise,
            total_sector_key,
        )
        self._measurement_file_weekly_co2_ff = measurement_file_weekly_co2_ff
        self.weekly_co2_ff_sector_key = weekly_co2_ff_sector_key
        self._combined_measurements = None

    @staticmethod
    def _adjust_total_measurement_coords(total_measurements: xr.DataArray):
        """Adjust the coordinates of the total measurements to be able to combine them
        with the weekly CO2_ff measurements.

        Args:
            total_measurements (xr.DataArray): Total CO2 measurements.

        Returns:
            xr.DataArray: Total CO2 measurements with adjusted coordinates.
        """
        new_measurement_coords = np.arange(total_measurements.measurement.size)
        mtimes = total_measurements.MTime.values
        mplaces = total_measurements.MPlace.values
        return total_measurements.assign_coords(
            measurement=new_measurement_coords,
            MTime=("measurement", mtimes),
            MPlace=("measurement", mplaces),
        )

    @staticmethod
    def _adjust_weekly_co2_ff_measurement_coords(
        weekly_co2_ff_measurements: xr.DataArray,
        start_measurement_id: int,
        mplace_name: str,
    ):
        """Adjust the coordinates of the weekly CO2_ff measurements to be able to combine
        them with the total measurements.

        Args:
            weekly_co2_ff_measurements (xr.DataArray): Weekly integrated CO2_ff
                 measurements.
            start_measurement_id (int): The starting measurement ID for the weekly CO2_ff
                 measurements. Should be equal to the number of total measurements.
            mplace_name (str): The name to assign to the MPlace coordinate for the weekly
                 CO2_ff measurements.
        Returns:
            xr.DataArray: Weekly integrated CO2_ff measurements with adjusted coordinates.
        """
        new_measurement_coords = (
            np.arange(weekly_co2_ff_measurements.measurement_id.size)
            + start_measurement_id
        )
        mpalce_values = np.char.encode(
            np.array(
                [mplace_name] * weekly_co2_ff_measurements.measurement_id.size,
                dtype="str",
            )
        )
        return weekly_co2_ff_measurements.rename(
            measurement_id="measurement"
        ).assign_coords(
            measurement=new_measurement_coords,
            # Mplace as coordinate for measurement but not a s index coordinate
            MPlace=("measurement", mpalce_values),
        )

    @property
    def measurements(self):
        if self._combined_measurements is None:
            total_measurements = self._adjust_total_measurement_coords(
                super().measurements
            )
            weekly_co2_ff_measurements = self._adjust_weekly_co2_ff_measurement_coords(
                xr.open_dataset(self._measurement_file_weekly_co2_ff)[
                    self.weekly_co2_ff_sector_key
                ],
                start_measurement_id=total_measurements.measurement.size,
                mplace_name=self.CO2_FF_MPLACE_NAME,
            )
            self._combined_measurements = (
                xr.concat(
                    [total_measurements, weekly_co2_ff_measurements],
                    dim="measurement",
                )
                .astype(FLOAT_PRECISION)
                .compute()
            )
        return self._combined_measurements

    def load_timeframe(self, start_time, end_time):
        measurement_subset = self.measurements.sel(
            measurement=(self.measurements.MTime >= start_time)
            * (self.measurements.MTime <= end_time)
        )
        return measurement_subset
