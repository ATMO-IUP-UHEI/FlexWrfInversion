""" This module descibes classes that can be used to load footprint data for an
 inversion."""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import xarray as xr

FLOAT_PRECISION = np.float32


class FootprintLoader(ABC):
    @abstractmethod
    def load_timeframe(
        self,
        start_time: np.datetime64,
        end_time: np.datetime64,
        start_mtime: np.datetime64,
        end_mtime: np.datetime64,
    ) -> xr.DataArray:
        """Load timeframe of the data with respect emission and measurement time.

        Args:
            start_time (np.datetime64): Start time of the emission timeframe
            end_time (np.datetime64): End time of the emission timeframe
            start_mtime (np.datetime64): Start time of the measurement timeframe
            end_mtime (np.datetime64): End time of the measurement timeframe

        Returns:
            xr.DataArray: Loaded timeframe
        """
        pass

    @staticmethod
    def _open_and_prepare(
        path: Path,
    ):
        """Opens footprint data and drops unnecessary parts.
        Args:
            path (Path): Path of file to open
        Returns:
            xr.Dataset: Prepared footprint data
        """
        if path.suffix == ".nc":
            data = (
                xr.open_dataset(
                    path,
                    chunks="auto",
                )
                .drop_dims(["x_stag", "y_stag"])
                .fillna(0)
            )

        elif path.suffix == ".pkl":
            with open(path, "rb") as f:
                data = pickle.load(f)
        return data

    @staticmethod
    def _combine_subsectors(
        footprints1: xr.DataArray,
        footprints2: xr.DataArray,
    ):
        """Combine footprints of two different subsectors

        Args:
            footprints1 (xr.DataArray): Footprints of the first subsector
            footprints2 (xr.DataArray): Footprints of the second subsector

        Returns:
            xr.DataArray: Combined footprints
        """
        return xr.concat(
            [
                footprints1.assign_coords(subsector=footprints1.group.values),
                footprints2.assign_coords(subsector=footprints2.group.values),
            ],
            dim="subsector",
        )

    @staticmethod
    def _select_measurements(
        footprints: xr.DataArray,
        leave_out: list[str] = None,
        keep_only: list[str] = None,
        times_of_day: list[int] = None,
    ):
        """Select measurements from the footprints

        Args:
            footprints (xr.DataArray): Footprints to select measurements from
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
            footprints = footprints.isel(
                MPlace=~np.isin(footprints.MPlace.values, leave_out)
            )
        if keep_only is not None:
            footprints = footprints.sel(MPlace=keep_only)

        if times_of_day is not None:
            footprints = footprints.isel(
                MTime=footprints.MTime.dt.hour.isin(times_of_day)
            )
        return footprints


class FlexibleFootprintLoaderTotal(FootprintLoader):
    TOTAL_EMISSION_KEY = "CO2_TOTAL"
    STATE_DIMS = ["subsector", "Time"]
    MEASUREMENT_DIMS = ["MTime", "MPlace"]

    def __init__(
        self,
        footprint_file_city: str | Path,
        footprint_file_germany: str | Path,
        leave_out: list[str] = None,
        keep_only: list[str] = None,
        times_of_day: list[int] = None,
    ):
        """Flexible implementation of footprint loader for total CO2

        Args:
            footprint_file_city (str | Path): Footprint file containing `CO2_TOTAL` field
                 for the city domain
            footprint_file_germany (str | Path): Footprint file containig `CO2_TOTAL`
                 field for the germany domain
            leave_out (list[str], optional): List of names of stations to exclude for the
                 runs. Defaults to None.
            keep_only (list[str], optional): List of names of station to only include
                 these. Defaults to None.
            times_of_day (list[int], optional): List of hours of the day to include in the
                 data. Defaults to None.
        """
        self._footprint_file_city = Path(footprint_file_city)
        self._footprint_file_germany = Path(footprint_file_germany)
        if leave_out is not None and keep_only is not None:
            raise ValueError("leave_out and keep_only cannot be used together.")
        elif leave_out is not None:
            leave_out = np.char.encode(np.array(leave_out, dtype=str))
        elif keep_only is not None:
            keep_only = np.char.encode(np.array(keep_only, dtype=str))
        self._leave_out = leave_out
        self._keep_only = keep_only
        self._times_of_day = times_of_day
        self._footprint = None
        self._footprint_unstacked = None

    @property
    def footprint(self) -> xr.DataArray:
        """Footprint property

        Returns:
            xr.DataArray: 2D DataArray containing the loaded footprints
        """
        if self._footprint is None:
            footprints_city = self._open_and_prepare(self._footprint_file_city)[
                self.TOTAL_EMISSION_KEY
            ]
            footprints_germany = self._open_and_prepare(self._footprint_file_germany)[
                self.TOTAL_EMISSION_KEY
            ]

            self._footprint = self._combine_subsectors(
                footprints_city, footprints_germany
            )

            self._footprint = self._select_measurements(
                self._footprint, self._leave_out, self._keep_only, self._times_of_day
            )

            self._footprint = (
                self._footprint.sortby("subsector")
                .stack(state=self.STATE_DIMS, measurement=self.MEASUREMENT_DIMS)
                .astype(FLOAT_PRECISION)
                .compute()
            )
        return self._footprint

    @property
    def unstacked_footprint(self) -> xr.DataArray:
        """Unstacked footprint data

        Returns:
            xr.DataArray: N-D footprint data in original structure
        """
        if self._footprint_unstacked is None:
            self._footprint_unstacked = self.footprint.unstack()
        return self._footprint_unstacked

    def load_timeframe(
        self,
        start_time: np.datetime64,
        end_time: np.datetime64,
        start_mtime: np.datetime64,
        end_mtime: np.datetime64,
    ) -> xr.DataArray:
        timeframe_data = (
            self.unstacked_footprint.sel(Time=slice(start_time, end_time))
            .sel(MTime=slice(start_mtime, end_mtime))
            .stack(state=self.STATE_DIMS, measurement=self.MEASUREMENT_DIMS)
        )
        if self._footprint_file_city.suffix == ".pkl":
            timeframe_data.values = timeframe_data.data.todense()
        return timeframe_data


class FlexibleFootprintLoaderAnthBio(FootprintLoader):
    ANTH_SECTOR_KEY = "CO2_ANT_TOTAL"
    BIO_SECTOR_KEY = "E_CO2_VPRM"
    STATE_DIMS = ["subsector", "Time", "sector"]
    MEASUREMENT_DIMS = ["MTime", "MPlace"]

    def __init__(
        self,
        footprint_file_city_bio: str | Path,
        footprint_file_city_ant: str | Path,
        footprint_file_germany_bio: str | Path,
        footprint_file_germany_ant: str | Path,
        leave_out: list[str] = None,
        keep_only: list[str] = None,
        times_of_day: list[int] = None,
    ):
        """Flexible implementation of footprint loader to load anthropogenic and biogenic
        parts of the footprints

        Args:
            footprint_file_city_bio (str | Path): Footprint file for the city that
                 contains the `E_CO2_VPRM`
            footprint_file_city_ant (str | Path): Footprint file for the city that
                 contains the `CO2_ANT_TOTAL`
            footprint_file_germany_bio (str | Path): Footprint file for germany that
                 contains the `E_CO2_VPRM`
            footprint_file_germany_ant (str | Path): Footprint file for germany that
                 contains the `CO2_ANT_TOTAL`
            leave_out (list[str], optional): List of names of stations to exclude for the
                 runs. Defaults to None.
            keep_only (list[str], optional): List of names of station to only include
                 these. Defaults to None.
            times_of_day (list[int], optional): List of hours of the day to include in the
                 data. Defaults to None.

        """

        self._footprint_file_city_bio = Path(footprint_file_city_bio)
        self._footprint_file_city_ant = Path(footprint_file_city_ant)
        self._footprint_file_germany_bio = Path(footprint_file_germany_bio)
        self._footprint_file_germany_ant = Path(footprint_file_germany_ant)
        if leave_out is not None and keep_only is not None:
            raise ValueError("leave_out and keep_only cannot be used together.")
        elif leave_out is not None:
            leave_out = np.char.encode(np.array(leave_out, dtype=str))
        elif keep_only is not None:
            keep_only = np.char.encode(np.array(keep_only, dtype=str))
        self._leave_out = leave_out
        self._keep_only = keep_only
        self._times_of_day = times_of_day
        self._footprint = None
        self._footprint_unstacked = None

    @property
    def footprint(self) -> xr.DataArray:
        """Footprint property

        Returns:
            xr.DataArray: 2D DataArray containing the loaded footprints
        """
        if self._footprint is None:
            footprints_bio_city = self._open_and_prepare(self._footprint_file_city_bio)[
                self.BIO_SECTOR_KEY
            ]

            footprints_bio_germany = self._open_and_prepare(
                self._footprint_file_germany_bio
            )[self.BIO_SECTOR_KEY]

            footprints_anth_city = self._open_and_prepare(
                self._footprint_file_city_ant
            )[self.ANTH_SECTOR_KEY]
            footprints_anth_germany = self._open_and_prepare(
                self._footprint_file_germany_ant
            )[self.ANTH_SECTOR_KEY]

            bio_footprints = (
                self._combine_subsectors(footprints_bio_city, footprints_bio_germany)
                .expand_dims(sector=[footprints_bio_city.name])
                .sortby("subsector")
            )

            anth_footprints = (
                self._combine_subsectors(footprints_anth_city, footprints_anth_germany)
                .expand_dims(sector=[footprints_anth_city.name])
                .sortby("subsector")
            )

            self._footprint = xr.concat(
                [bio_footprints, anth_footprints], dim="sector"
            ).sortby("sector")

            self._footprint = self._select_measurements(
                self._footprint, self._leave_out, self._keep_only, self._times_of_day
            )

            self._footprint = (
                self._footprint.sortby("sector")
                .sortby("subsector")
                .stack(state=self.STATE_DIMS, measurement=self.MEASUREMENT_DIMS)
                .astype(FLOAT_PRECISION)
                .compute()
                .fillna(0)
            )
        return self._footprint

    @property
    def footprint_unstacked(self) -> xr.DataArray:
        """Unstacked footprint data

        Returns:
            xr.DataArray: N-D footprint data in original structure
        """
        if self._footprint_unstacked is None:
            self._footprint_unstacked = self.footprint.unstack()
        return self._footprint_unstacked

    def load_timeframe(
        self,
        start_time: np.datetime64,
        end_time: np.datetime64,
        start_mtime: np.datetime64,
        end_mtime: np.datetime64,
    ) -> xr.DataArray:
        timeframe_data = (
            self.footprint_unstacked.sel(Time=slice(start_time, end_time))
            .sel(MTime=slice(start_mtime, end_mtime))
            .stack(state=self.STATE_DIMS, measurement=self.MEASUREMENT_DIMS)
        )
        if self._footprint_file_city_bio.suffix == ".pkl":
            timeframe_data.values = timeframe_data.data.todense()
        return timeframe_data


class FlexibleFootprintLoaderAnthBioCo(FootprintLoader):
    ANTH_SECTOR_KEY = "CO2_ANT_TOTAL"
    BIO_SECTOR_KEY = "E_CO2_VPRM"
    CO_SECTOR_KEY = "E_CO"
    STATE_DIMS = ["subsector", "Time", "sector"]
    MEASUREMENT_DIMS = ["MTime", "MPlace", "species"]

    def __init__(
        self,
        footprint_file_city_bio: str | Path,
        footprint_file_city_ant: str | Path,
        footprint_file_city_co: str | Path,
        footprint_file_germany_bio: str | Path,
        footprint_file_germany_ant: str | Path,
        footprint_file_germany_co: str | Path,
        leave_out: list[str] = None,
        keep_only: list[str] = None,
        times_of_day: list[int] = None,
    ):
        self._footprint_file_city_bio = Path(footprint_file_city_bio)
        self._footprint_file_city_ant = Path(footprint_file_city_ant)
        self._footprint_file_city_co = Path(footprint_file_city_co)
        self._footprint_file_germany_bio = Path(footprint_file_germany_bio)
        self._footprint_file_germany_ant = Path(footprint_file_germany_ant)
        self._footprint_file_germany_co = Path(footprint_file_germany_co)
        if leave_out is not None and keep_only is not None:
            raise ValueError("leave_out and keep_only cannot be used together.")
        elif leave_out is not None:
            leave_out = np.char.encode(np.array(leave_out, dtype=str))
        elif keep_only is not None:
            keep_only = np.char.encode(np.array(keep_only, dtype=str))
        self._leave_out = leave_out
        self._keep_only = keep_only
        self._times_of_day = times_of_day
        self._footprint = None
        self._footprint_unstacked = None

    @property
    def footprint(self):
        if self._footprint is None:
            footprints_city_bio = self._open_and_prepare(self._footprint_file_city_bio)[
                self.BIO_SECTOR_KEY
            ]
            footprints_city_ant = self._open_and_prepare(self._footprint_file_city_ant)[
                self.ANTH_SECTOR_KEY
            ]
            footprints_city_co = self._open_and_prepare(self._footprint_file_city_co)[
                self.CO_SECTOR_KEY
            ]

            footprints_germany_bio = self._open_and_prepare(
                self._footprint_file_germany_bio
            )[self.BIO_SECTOR_KEY]
            footprints_germany_ant = self._open_and_prepare(
                self._footprint_file_germany_ant
            )[self.ANTH_SECTOR_KEY]
            footprints_germany_co = self._open_and_prepare(
                self._footprint_file_germany_co
            )[self.CO_SECTOR_KEY]

            bio_footprints = (
                self._combine_subsectors(footprints_city_bio, footprints_germany_bio)
                .expand_dims(sector=[footprints_city_bio.name], species=["CO2"])
                .sortby("subsector")
            )
            anth_footprints = (
                self._combine_subsectors(footprints_city_ant, footprints_germany_ant)
                .expand_dims(sector=[footprints_city_ant.name], species=["CO2"])
                .sortby("subsector")
            )
            co_footprints = (
                self._combine_subsectors(footprints_city_co, footprints_germany_co)
                .expand_dims(sector=[footprints_city_co.name], species=["CO"])
                .sortby("subsector")
            )

            bio_species_dummy = xr.zeros_like(bio_footprints).assign_coords(
                species=["CO"]
            )
            anth_species_dummy = xr.zeros_like(anth_footprints).assign_coords(
                species=["CO"]
            )
            co_species_dummy = xr.zeros_like(co_footprints).assign_coords(
                species=["CO2"]
            )

            bio_footprints = xr.concat(
                [bio_footprints, bio_species_dummy], dim="species"
            )
            anth_footprints = xr.concat(
                [anth_footprints, anth_species_dummy], dim="species"
            )
            co_footprints = xr.concat([co_species_dummy, co_footprints], dim="species")

            self._footprint = xr.concat(
                [bio_footprints, anth_footprints, co_footprints], dim="sector"
            )

            self._footprint = self._select_measurements(
                self._footprint, self._leave_out, self._keep_only, self._times_of_day
            )

            self._footprint = (
                self._footprint.sortby("species")
                .sortby("sector")
                .sortby("subsector")
                .stack(state=self.STATE_DIMS, measurement=self.MEASUREMENT_DIMS)
                .astype(FLOAT_PRECISION)
                .compute()
                .fillna(0)
            )
        return self._footprint

    @property
    def footprint_unstacked(self):
        if self._footprint_unstacked is None:
            self._footprint_unstacked = self.footprint.unstack()
        return self._footprint_unstacked

    def load_timeframe(
        self,
        start_time: np.datetime64,
        end_time: np.datetime64,
        start_mtime: np.datetime64,
        end_mtime: np.datetime64,
    ) -> xr.DataArray:
        timeframe_data = (
            self.footprint_unstacked.sel(Time=slice(start_time, end_time))
            .sel(MTime=slice(start_mtime, end_mtime))
            .stack(state=self.STATE_DIMS, measurement=self.MEASUREMENT_DIMS)
        )
        if self._footprint_file_city_bio.suffix == ".pkl":
            timeframe_data.values = timeframe_data.data.todense()
        return timeframe_data
