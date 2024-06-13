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

    def load_timeframe(
        self, start_time: np.datetime64, end_time: np.datetime64
    ) -> xr.DataArray:
        """Load the target data
        Returns:
            xr.DataArray: The target data as 1D array. Coordinates should be stacked
                beforehand.
        """
        pass


class TargetLoaderTotalInCity(TargetLoader):
    EMISSION_SECTORS_TO_LOAD = ["CO2_ANT_TOTAL", "E_CO2_VPRM", "CO2_TOTAL"]
    STATE_DIMS = ["subsector", "Time"]
    TOTAL_EMISSION_KEY = "CO2_TOTAL"

    def __init__(
        self,
        remapped_data_path: str,
        season: str,
        city: str,
        prior_type: str = "true",
        city_suffixes: list[str] = None,
        germany_suffixes: list[str] = None,
        time_resolution: int = 3,
    ):
        self._remapped_data_path = Path(remapped_data_path)
        self._season = season
        self._city = city
        self._prior_type = prior_type
        self._city_suffixes = (
            city_suffixes if city_suffixes is not None else ["", "_sums"]
        )
        self._germany_suffixes = (
            germany_suffixes if germany_suffixes is not None else ["_vprm", "_sums"]
        )
        self._time_resolution = time_resolution

        self._target_with_additional_sectors = None
        self._target = None

    @property
    def target_with_additional_sectors(self) -> xr.Dataset:
        if self._target_with_additional_sectors is None:
            city_folder = (
                self._remapped_data_path / self._season / self._city / self._prior_type
            )
            germany_folder = (
                self._remapped_data_path / self._season / "germany" / self._prior_type
            )
            time_res_string = f"_{self._time_resolution}H"

            true_emissions_city = (
                xr.open_dataset(
                    city_folder
                    / (
                        "remapped_true_emissions"
                        + f"{self._city_suffixes[0]}{time_res_string}.nc"
                    ),
                    chunks="auto",
                )
                .drop_dims(["x_stag", "y_stag"])
                .fillna(0)
            )
            true_emissions_germany = (
                xr.open_dataset(
                    germany_folder
                    / (
                        "remapped_true_emissions"
                        + f"{self._germany_suffixes[0]}{time_res_string}.nc"
                    ),
                    chunks="auto",
                )
                .drop_dims(["x_stag", "y_stag"])
                .fillna(0)
            )

            true_emissions_sums_city = (
                xr.open_dataset(
                    city_folder
                    / (
                        "remapped_true_emissions"
                        + f"{self._city_suffixes[1]}{time_res_string}.nc"
                    ),
                    chunks="auto",
                )
                .drop_dims(["x_stag", "y_stag"])
                .fillna(0)
            )
            true_emissions_sums_germany = (
                xr.open_dataset(
                    germany_folder
                    / (
                        "remapped_true_emissions"
                        + f"{self._germany_suffixes[1]}{time_res_string}.nc"
                    ),
                    chunks="auto",
                )
                .drop_dims(["x_stag", "y_stag"])
                .fillna(0)
            )

            self._target_with_additional_sectors = (
                (
                    xr.merge(
                        [
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
                            ),
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
                            ),
                        ]
                    )[self.EMISSION_SECTORS_TO_LOAD].sortby("subsector")
                )
                .stack(state=self.STATE_DIMS)
                .compute()
            )
        return self._target_with_additional_sectors

    @property
    def target(self) -> xr.DataArray:
        return self.target_with_additional_sectors[self.TOTAL_EMISSION_KEY].astype(
            np.float32
        )

    def load_timeframe(
        self, start_time: np.datetime64, end_time: np.datetime64
    ) -> xr.DataArray:
        return (
            self.target.unstack()
            .sel(Time=slice(start_time, end_time))
            .stack(state=self.STATE_DIMS)
        )


class TargetLoaderAnthAndBioSectors(TargetLoader):
    ANTH_SECTOR_KEY = "CO2_ANT_TOTAL"
    BIO_SECTOR_KEY = "E_CO2_VPRM"
    STATE_DIMS = ["subsector", "Time", "sector"]

    def __init__(
        self,
        remapped_data_path: str,
        season: str,
        city: str,
        prior_type: str = "true",
        city_suffixes: list[str] = None,
        germany_suffixes: list[str] = None,
        time_resolution: int = 3,
    ):
        self._remapped_data_path = Path(remapped_data_path)
        self._season = season
        self._city = city
        self._prior_type = prior_type
        self._city_suffixes = (
            city_suffixes if city_suffixes is not None else ["", "_sums"]
        )
        self._germany_suffixes = (
            germany_suffixes if germany_suffixes is not None else ["_vprm", "_sums"]
        )
        self._time_resolution = time_resolution

        self._target = None

    @property
    def target(self) -> xr.DataArray:
        if self._target is None:
            city_folder = (
                self._remapped_data_path / self._season / self._city / self._prior_type
            )
            germany_folder = (
                self._remapped_data_path / self._season / "germany" / self._prior_type
            )
            time_res_string = f"_{self._time_resolution}H"

            true_emissions_city = (
                xr.open_dataset(
                    city_folder
                    / (
                        "remapped_true_emissions"
                        + f"{self._city_suffixes[0]}{time_res_string}.nc"
                    ),
                    chunks="auto",
                )
                .drop_dims(["x_stag", "y_stag"])
                .fillna(0)
            )[self.BIO_SECTOR_KEY]
            true_emissions_germany = (
                xr.open_dataset(
                    germany_folder
                    / (
                        "remapped_true_emissions"
                        + f"{self._germany_suffixes[0]}{time_res_string}.nc"
                    ),
                    chunks="auto",
                )
                .drop_dims(["x_stag", "y_stag"])
                .fillna(0)
            )[self.BIO_SECTOR_KEY]

            true_emissions_sums_city = (
                xr.open_dataset(
                    city_folder
                    / (
                        "remapped_true_emissions"
                        + f"{self._city_suffixes[1]}{time_res_string}.nc"
                    ),
                    chunks="auto",
                )
                .drop_dims(["x_stag", "y_stag"])
                .fillna(0)
            )[self.ANTH_SECTOR_KEY]
            true_emissions_sums_germany = (
                xr.open_dataset(
                    germany_folder
                    / (
                        "remapped_true_emissions"
                        + f"{self._germany_suffixes[1]}{time_res_string}.nc"
                    ),
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
