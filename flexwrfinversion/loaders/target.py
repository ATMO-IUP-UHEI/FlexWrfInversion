""" This module descibes classes that can be used to load target data for an inversion."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import xarray as xr


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


class FlexibleTargetLoaderTotal(TargetLoader):
    TOTAL_EMISSION_KEY = "CO2_TOTAL"
    STATE_DIMS = ["subsector", "Time"]

    def __init__(
        self,
        target_file_city: str | Path,
        target_file_germany: str | Path,
    ):
        """Flexible implementation of target loader for total CO2.

        Args:
            target_file_city (str | Path): File that contatains the emission data for
                 the city and the field `CO2_TOTAL`
            target_file_germany (str | Path): File that contatains the emission data for
                 germany and the field `CO2_TOTAL`
        """
        self._target_file_city = target_file_city
        self._target_file_germany = target_file_germany
        self._target = None

    @property
    def target(self) -> xr.DataArray:
        if self._target is None:
            target_city = xr.open_dataset(self._target_file_city)[
                self.TOTAL_EMISSION_KEY
            ]
            target_germany = xr.open_dataset(self._target_file_germany)[
                self.TOTAL_EMISSION_KEY
            ]
            self._target = (
                xr.concat(
                    [
                        target_city.assign_coords(subsector=target_city.group.values),
                        target_germany.assign_coords(
                            subsector=target_germany.group.values
                        ),
                    ],
                    dim="subsector",
                )
                .stack(state=self.STATE_DIMS)
                .sortby("subsector")
                .astype(np.float32)
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
    ANTH_SECTOR_KEY = "CO2_ANT_TOTAL"
    BIO_SECTOR_KEY = "E_CO2_VPRM"
    TOTAL_EMISSION_KEY = "CO2_TOTAL"
    STATE_DIMS = ["subsector", "Time", "sector"]

    def __init__(
        self,
        target_file_city_bio: str | Path,
        target_file_city_ant: str | Path,
        target_file_germany_bio: str | Path,
        target_file_germany_ant: str | Path,
    ):
        """Flexible implementation of target loader for laoding anthropogenic and
        biogenic emissions.

        Args:
            target_file_city_bio (str | Path):  Emission file for the city that contains
                 `E_CO2_VPRM`
            target_file_city_ant (str | Path): Emission file for the city that contains
                 `CO2_ANT_TOTAL`
            target_file_germany_bio (str | Path): Emission file for germany that contains
                 `E_CO2_VPRM`
            target_file_germany_ant (str | Path): Emission file for germany that
                 contains `CO2_ANT_TOTAL`
        """
        self._target_file_city_bio = target_file_city_bio
        self._target_file_city_ant = target_file_city_ant
        self._target_file_germany_bio = target_file_germany_bio
        self._target_file_germany_ant = target_file_germany_ant
        self._target = None

    @property
    def target(self) -> xr.DataArray:
        if self._target is None:
            true_emissions_city = (
                xr.open_dataset(
                    self._target_file_city_bio,
                    chunks="auto",
                )
                .drop_dims(["x_stag", "y_stag"])
                .fillna(0)
            )[self.BIO_SECTOR_KEY]
            true_emissions_germany = (
                xr.open_dataset(
                    self._target_file_germany_bio,
                    chunks="auto",
                )
                .drop_dims(["x_stag", "y_stag"])
                .fillna(0)
            )[self.BIO_SECTOR_KEY]

            true_emissions_sums_city = (
                xr.open_dataset(
                    self._target_file_city_ant,
                    chunks="auto",
                )
                .drop_dims(["x_stag", "y_stag"])
                .fillna(0)
            )[self.ANTH_SECTOR_KEY]
            true_emissions_sums_germany = (
                xr.open_dataset(
                    self._target_file_germany_ant,
                    chunks="auto",
                )
                .drop_dims(["x_stag", "y_stag"])
                .fillna(0)
            )[self.ANTH_SECTOR_KEY]

            bio_emissions = (
                xr.concat(
                    [
                        true_emissions_city.assign_coords(
                            subsector=true_emissions_city.group.values
                        ),
                        true_emissions_germany.assign_coords(
                            subsector=true_emissions_germany.group.values
                        ),
                    ],
                    dim="subsector",
                )
                .expand_dims(sector=[true_emissions_city.name])
                .sortby("subsector")
            )

            anth_emissions = (
                xr.concat(
                    [
                        true_emissions_sums_city.assign_coords(
                            subsector=true_emissions_sums_city.group.values
                        ),
                        true_emissions_sums_germany.assign_coords(
                            subsector=true_emissions_sums_germany.group.values
                        ),
                    ],
                    dim="subsector",
                )
                .expand_dims(sector=[true_emissions_sums_city.name])
                .sortby("subsector")
            )

            self._target = (
                xr.concat([bio_emissions, anth_emissions], dim="sector")
                .stack(state=self.STATE_DIMS)
                .sortby("sector")
                .sortby("subsector")
                .astype(np.float32)
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
    ANTH_SECTOR_KEY = "CO2_ANT_TOTAL"
    BIO_SECTOR_KEY = "E_CO2_VPRM"
    CO_SECTOR_KEY = "E_CO"
    STATE_DIMS = ["subsector", "Time", "sector"]

    def __init__(
        self,
        target_file_city_bio: str | Path,
        target_file_city_ant: str | Path,
        target_file_city_co: str | Path,
        target_file_germany_bio: str | Path,
        target_file_germany_ant: str | Path,
        target_file_germany_co: str | Path,
    ) -> None:
        """Flexible implementation of target loader for laoding anthropogenic, biogenic
        and CO emissions.

        Args:
            target_file_city_bio (str | Path):  Emission file for the city that contains
                 `E_CO2_VPRM`
            target_file_city_ant (str | Path): Emission file for the city that contains
                 `CO2_ANT_TOTAL`
            target_file_city_co (str | Path): Emission file for the city that contains
                 `E_CO`
            target_file_germany_bio (str | Path): Emission file for germany that contains
                 `E_CO2_VPRM`
            target_file_germany_ant (str | Path): Emission file for germany that
                 contains `CO2_ANT_TOTAL`
            target_file_germany_co (str | Path): Emission file for germany that
                 contains `E_CO`
        """
        self._target_file_city_bio = target_file_city_bio
        self._target_file_city_ant = target_file_city_ant
        self._target_file_city_co = target_file_city_co
        self._target_file_germany_bio = target_file_germany_bio
        self._target_file_germany_ant = target_file_germany_ant
        self._target_file_germany_co = target_file_germany_co
        self._target = None

    @property
    def target(self) -> xr.DataArray:
        if self._target is None:
            target_city_bio = xr.open_dataset(self._target_file_city_bio)[
                self.BIO_SECTOR_KEY
            ]
            target_city_ant = xr.open_dataset(self._target_file_city_ant)[
                self.ANTH_SECTOR_KEY
            ]
            target_city_co = xr.open_dataset(self._target_file_city_co)[
                self.CO_SECTOR_KEY
            ]
            target_germany_bio = xr.open_dataset(self._target_file_germany_bio)[
                self.BIO_SECTOR_KEY
            ]
            target_germany_ant = xr.open_dataset(self._target_file_germany_ant)[
                self.ANTH_SECTOR_KEY
            ]
            target_germany_co = xr.open_dataset(self._target_file_germany_co)[
                self.CO_SECTOR_KEY
            ]

            bio_emissions = (
                xr.concat(
                    [
                        target_city_bio.assign_coords(
                            subsector=target_city_bio.group.values
                        ),
                        target_germany_bio.assign_coords(
                            subsector=target_germany_bio.group.values
                        ),
                    ],
                    dim="subsector",
                )
                .expand_dims(sector=[self.BIO_SECTOR_KEY])
                .sortby("subsector")
            )

            anth_emissions = (
                xr.concat(
                    [
                        target_city_ant.assign_coords(
                            subsector=target_city_ant.group.values
                        ),
                        target_germany_ant.assign_coords(
                            subsector=target_germany_ant.group.values
                        ),
                    ],
                    dim="subsector",
                )
                .expand_dims(sector=[self.ANTH_SECTOR_KEY])
                .sortby("subsector")
            )

            co_emissions = (
                xr.concat(
                    [
                        target_city_co.assign_coords(
                            subsector=target_city_co.group.values
                        ),
                        target_germany_co.assign_coords(
                            subsector=target_germany_co.group.values
                        ),
                    ],
                    dim="subsector",
                )
                .expand_dims(sector=[self.CO_SECTOR_KEY])
                .sortby("subsector")
            )

            self._target = (
                xr.concat([bio_emissions, anth_emissions, co_emissions], dim="sector")
                .stack(state=self.STATE_DIMS)
                .sortby("sector")
                .sortby("subsector")
                .astype(np.float32)
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
