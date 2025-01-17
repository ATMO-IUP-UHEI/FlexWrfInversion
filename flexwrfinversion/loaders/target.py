""" This module descibes classes that can be used to load target data for an inversion."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import xarray as xr

FLOAT_PRECISION = np.float32


class TargetLoader(ABC):
    @property
    @abstractmethod
    def target(self, *args, **kwargs) -> xr.DataArray:
        """Load the target data
        Returns:
            xr.DataArray: The target data as 1D array. Coordinates should be stacked
                beforehand.
        """
        pass

    @property
    @abstractmethod
    def STATE_DIMS(self) -> list[str]:
        """Return the dimensions of the state"""
        pass

    @abstractmethod
    def load_timeframe(
        self, start_time: np.datetime64, end_time: np.datetime64
    ) -> xr.DataArray:
        """Load the target data
        Args:
            start_time (np.datetime64): Start time of the emission timeframe
            end_time (np.datetime64): End time of the emissions timeframe
        Returns:
            xr.DataArray: The target data as 1D array. Coordinates should be stacked
                beforehand.
        """
        pass

    @staticmethod
    def open_and_prepare(
        path: Path,
    ):
        """Open the dataset and prepare it for the inversion.

        Args:
            path (Path): Path to the dataset

        Returns:
            xr.Dataset: The dataset prepared for the inversion
        """
        data = xr.open_dataset(
            path,
            chunks="auto",
        ).fillna(0)
        try:
            data = data.drop_dims(["x_stag", "y_stag"])
        except ValueError:
            pass
        return data

    @staticmethod
    def _combine_subsectors(
        target1: xr.DataArray,
        target2: xr.DataArray,
    ):
        """Combine target of two different subsectors

        Args:
            target1 (xr.DataArray): target of the first subsector
            target2 (xr.DataArray): target of the second subsector

        Returns:
            xr.DataArray: Combined target
        """
        return xr.concat(
            [
                target1.assign_coords(subsector=target1.group.values),
                target2.assign_coords(subsector=target2.group.values),
            ],
            dim="subsector",
        )


class FlexibleTargetLoaderTotal(TargetLoader):
    STATE_DIMS = ["subsector", "Time"]

    def __init__(
        self,
        target_file_city: str | Path,
        target_file_germany: str | Path,
        total_sector_key: str = "CO2_TOTAL",
    ):
        """Flexible implementation of target loader for total CO2.

        Args:
            target_file_city (str | Path): File that contains the emission data for
                 the city and the field specified by `total_sector_key`
            target_file_germany (str | Path): File that contains the emission data for
                 germany and the field specified by `total_sector_key`
            total_sector_key (str): Key for the total emission field in the dataset
        """
        self._target_file_city = target_file_city
        self._target_file_germany = target_file_germany
        self.total_sector_key = total_sector_key
        self._target = None

    @property
    def target(self) -> xr.DataArray:
        if self._target is None:
            target_city = self.open_and_prepare(self._target_file_city)[
                self.total_sector_key
            ]
            target_germany = self.open_and_prepare(self._target_file_germany)[
                self.total_sector_key
            ]
            self._target = (
                self._combine_subsectors(target_city, target_germany)
                .sortby("subsector")
                .stack(state=self.STATE_DIMS)
                .astype(FLOAT_PRECISION)
                .compute()
            )

        return self._target

    def load_timeframe(
        self, start_time: np.datetime64, end_time: np.datetime64
    ) -> xr.DataArray:
        return (
            self.target.unstack()
            .sel(Time=slice(start_time, end_time))
            .stack(state=self.STATE_DIMS)
        )


class FlexibleTargetLoaderAnthBio(TargetLoader):
    STATE_DIMS = ["subsector", "Time", "sector"]

    def __init__(
        self,
        target_file_city_bio: str | Path,
        target_file_city_ant: str | Path,
        target_file_germany_bio: str | Path,
        target_file_germany_ant: str | Path,
        ant_sector_key: str = "CO2_ANT_TOTAL",
        bio_sector_key: str = "E_CO2_VPRM",
    ):
        """Flexible implementation of target loader for loading anthropogenic and
        biogenic emissions.

        Args:
            target_file_city_bio (str | Path): Emission file for the city that contains
                 the field specified by `bio_sector_key`
            target_file_city_ant (str | Path): Emission file for the city that contains
                 the field specified by `ant_sector_key`
            target_file_germany_bio (str | Path): Emission file for germany that contains
                 the field specified by `bio_sector_key`
            target_file_germany_ant (str | Path): Emission file for germany that contains
                 the field specified by `ant_sector_key`
            ant_sector_key (str): Key for the anthropogenic emission field in the dataset
            bio_sector_key (str): Key for the biogenic emission field in the dataset
        """
        self._target_file_city_bio = target_file_city_bio
        self._target_file_city_ant = target_file_city_ant
        self._target_file_germany_bio = target_file_germany_bio
        self._target_file_germany_ant = target_file_germany_ant
        self.ant_sector_key = ant_sector_key
        self.bio_sector_key = bio_sector_key
        self._target = None

    @property
    def target(self) -> xr.DataArray:
        if self._target is None:
            target_city_bio = self.open_and_prepare(
                self._target_file_city_bio,
            )[self.bio_sector_key]
            target_germany_bio = self.open_and_prepare(
                self._target_file_germany_bio,
            )[self.bio_sector_key]

            target_city_ant = self.open_and_prepare(
                self._target_file_city_ant,
            )[self.ant_sector_key]
            target_germany_ant = self.open_and_prepare(
                self._target_file_germany_ant,
            )[self.ant_sector_key]

            bio_emissions = (
                self._combine_subsectors(target_city_bio, target_germany_bio)
                .expand_dims(sector=[target_city_bio.name])
                .sortby("subsector")
            )

            ant_emissions = (
                self._combine_subsectors(target_city_ant, target_germany_ant)
                .expand_dims(sector=[target_city_ant.name])
                .sortby("subsector")
            )

            self._target = (
                xr.concat([bio_emissions, ant_emissions], dim="sector")
                .sortby("sector")
                .sortby("subsector")
                .stack(state=self.STATE_DIMS)
                .astype(FLOAT_PRECISION)
                .compute()
            )
        return self._target

    def load_timeframe(
        self, start_time: np.datetime64, end_time: np.datetime64
    ) -> xr.DataArray:
        return (
            self.target.unstack()
            .sel(Time=slice(start_time, end_time))
            .stack(state=self.STATE_DIMS)
        )


class FlexibleTargetLoaderAnthBioCo(TargetLoader):
    STATE_DIMS = ["subsector", "Time", "sector"]

    def __init__(
        self,
        target_file_city_bio: str | Path,
        target_file_city_ant: str | Path,
        target_file_city_co: str | Path,
        target_file_germany_bio: str | Path,
        target_file_germany_ant: str | Path,
        target_file_germany_co: str | Path,
        ant_sector_key: str = "CO2_ANT_TOTAL",
        bio_sector_key: str = "E_CO2_VPRM",
        co_sector_key: str = "E_CO",
    ) -> None:
        """Flexible implementation of target loader for loading anthropogenic, biogenic
        and CO emissions.

        Args:
            target_file_city_bio (str | Path): Emission file for the city that contains
                 the field specified by `bio_sector_key`
            target_file_city_ant (str | Path): Emission file for the city that contains
                 the field specified by `ant_sector_key`
            target_file_city_co (str | Path): Emission file for the city that contains
                 the field specified by `co_sector_key`
            target_file_germany_bio (str | Path): Emission file for germany that contains
                 the field specified by `bio_sector_key`
            target_file_germany_ant (str | Path): Emission file for germany that contains
                 the field specified by `ant_sector_key`
            target_file_germany_co (str | Path): Emission file for germany that contains
                 the field specified by `co_sector_key`
            ant_sector_key (str): Key for the anthropogenic emission field in the dataset
            bio_sector_key (str): Key for the biogenic emission field in the dataset
            co_sector_key (str): Key for the CO emission field in the dataset
        """
        self._target_file_city_bio = target_file_city_bio
        self._target_file_city_ant = target_file_city_ant
        self._target_file_city_co = target_file_city_co
        self._target_file_germany_bio = target_file_germany_bio
        self._target_file_germany_ant = target_file_germany_ant
        self._target_file_germany_co = target_file_germany_co
        self.ant_sector_key = ant_sector_key
        self.bio_sector_key = bio_sector_key
        self.co_sector_key = co_sector_key
        self._target = None

    @property
    def target(self) -> xr.DataArray:
        if self._target is None:
            target_city_bio = self.open_and_prepare(
                self._target_file_city_bio,
            )[self.bio_sector_key]
            target_city_ant = self.open_and_prepare(
                self._target_file_city_ant,
            )[self.ant_sector_key]
            target_city_co = self.open_and_prepare(
                self._target_file_city_co,
            )[self.co_sector_key]
            target_germany_bio = self.open_and_prepare(
                self._target_file_germany_bio,
            )[self.bio_sector_key]
            target_germany_ant = self.open_and_prepare(
                self._target_file_germany_ant,
            )[self.ant_sector_key]
            target_germany_co = self.open_and_prepare(
                self._target_file_germany_co,
            )[self.co_sector_key]

            bio_emissions = (
                self._combine_subsectors(target_city_bio, target_germany_bio)
                .expand_dims(sector=[self.bio_sector_key])
                .sortby("subsector")
            )
            anth_emissions = (
                self._combine_subsectors(target_city_ant, target_germany_ant)
                .expand_dims(sector=[self.ant_sector_key])
                .sortby("subsector")
            )
            co_emissions = (
                self._combine_subsectors(target_city_co, target_germany_co)
                .expand_dims(sector=[self.co_sector_key])
                .sortby("subsector")
            )

            self._target = (
                xr.concat([bio_emissions, anth_emissions, co_emissions], dim="sector")
                .sortby("sector")
                .sortby("subsector")
                .stack(state=self.STATE_DIMS)
                .astype(FLOAT_PRECISION)
                .compute()
            )
        return self._target

    def load_timeframe(
        self, start_time: np.datetime64, end_time: np.datetime64
    ) -> xr.DataArray:
        return (
            self.target.unstack()
            .sel(Time=slice(start_time, end_time))
            .stack(state=self.STATE_DIMS)
        )
