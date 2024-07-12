""" This module descibes classes that can be used to load footprint data for an
 inversion."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import xarray as xr


class FootprintLoader(ABC):
    @abstractmethod
    def load_timeframe(
        self,
        start_time: np.datetime64,
        end_time: np.datetime64,
        start_mtime: np.datetime64,
        end_mtime: np.datetime64,
    ) -> xr.DataArray:
        """Load the footprint data
        Returns:
            xr.DataArray: The footprint data as 2D array. Coordinates should be
                stacked beforehand.
        """
        pass

    @staticmethod
    def _open_and_prepare(
        path: Path,
    ):
        return (
            xr.open_dataset(
                path,
                chunks="auto",
            )
            .drop_dims(["x_stag", "y_stag"])
            .fillna(0)
        )


class LoadFootprintForTotalInCity(FootprintLoader):
    TOTAL_EMISSION_KEY = "CO2_TOTAL"
    STATE_DIMS = ["subsector", "Time"]
    MEASUREMENT_DIMS = ["MTime", "MPlace"]

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

        self._footprint = None

    @property
    def footprint(self) -> xr.DataArray:
        if self._footprint is None:
            city_folder = (
                self._remapped_data_path / self._season / self._city / self._prior_type
            )
            germany_folder = (
                self._remapped_data_path / self._season / "germany" / self._prior_type
            )
            time_res_string = (
                "" if self._time_resolution is None else f"_{self._time_resolution}H"
            )

            footprints_city = self._open_and_prepare(
                city_folder
                / f"remapped_footprints{self._city_suffixes[0]}{time_res_string}.nc",
            )
            footprints_germany = self._open_and_prepare(
                germany_folder
                / (
                    "remapped_footprints"
                    + f"{self._germany_suffixes[0]}{time_res_string}.nc"
                )
            )
            footprints_sums_city = self._open_and_prepare(
                city_folder
                / f"remapped_footprints{self._city_suffixes[1]}{time_res_string}.nc"
            )
            footprints_sums_germany = self._open_and_prepare(
                germany_folder
                / (
                    "remapped_footprints"
                    + f"{self._germany_suffixes[1]}{time_res_string}.nc"
                )
            )

            self._footprint = (
                (
                    xr.merge(
                        [
                            xr.concat(
                                [
                                    footprints_city.assign_coords(
                                        subsector=footprints_city.group.values
                                    ),
                                    footprints_germany.assign_coords(
                                        subsector=footprints_germany.group.values
                                    ),
                                ],
                                dim="subsector",
                            ),
                            xr.concat(
                                [
                                    footprints_sums_city.assign_coords(
                                        subsector=footprints_city.group.values
                                    ),
                                    footprints_sums_germany.assign_coords(
                                        subsector=footprints_germany.group.values
                                    ),
                                ],
                                dim="subsector",
                            ),
                        ]
                    )[self.TOTAL_EMISSION_KEY]
                    .sortby("subsector")
                    .stack(state=self.STATE_DIMS, measurement=self.MEASUREMENT_DIMS)
                )
                .astype(np.float32)
                .compute()
            )
        return self._footprint

    def load_timeframe(
        self,
        start_time: np.datetime64,
        end_time: np.datetime64,
        start_mtime: np.datetime64,
        end_mtime: np.datetime64,
    ) -> xr.DataArray:
        return (
            self.footprint.unstack()
            .sel(Time=slice(start_time, end_time), MTime=slice(start_mtime, end_mtime))
            .stack(state=self.STATE_DIMS, measurement=self.MEASUREMENT_DIMS)
        )


class LoadFootprintAnthAndBioSectors(FootprintLoader):
    ANTH_SECTOR_KEY = "CO2_ANT_TOTAL"
    BIO_SECTOR_KEY = "E_CO2_VPRM"
    STATE_DIMS = ["subsector", "Time", "sector"]
    MEASUREMENT_DIMS = ["MTime", "MPlace"]

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

        self._footprint = None

    @property
    def footprint(self) -> xr.DataArray:
        if self._footprint is None:
            city_folder = (
                self._remapped_data_path / self._season / self._city / self._prior_type
            )
            germany_folder = (
                self._remapped_data_path / self._season / "germany" / self._prior_type
            )
            time_res_string = (
                "" if self._time_resolution is None else f"_{self._time_resolution}H"
            )

            footprints_city = self._open_and_prepare(
                city_folder
                / f"remapped_footprints{self._city_suffixes[0]}{time_res_string}.nc",
            )[self.BIO_SECTOR_KEY]

            footprints_germany = self._open_and_prepare(
                germany_folder
                / (
                    "remapped_footprints"
                    + f"{self._germany_suffixes[0]}{time_res_string}.nc"
                ),
            )[self.BIO_SECTOR_KEY]

            footprints_sums_city = self._open_and_prepare(
                city_folder
                / f"remapped_footprints{self._city_suffixes[1]}{time_res_string}.nc",
            )[self.ANTH_SECTOR_KEY]
            footprints_sums_germany = self._open_and_prepare(
                germany_folder
                / (
                    "remapped_footprints"
                    + f"{self._germany_suffixes[1]}{time_res_string}.nc"
                )
            )[self.ANTH_SECTOR_KEY]

            bio_footprints = (
                xr.concat(
                    [
                        footprints_city.assign_coords(
                            subsector=footprints_city.group.values
                        ),
                        footprints_germany.assign_coords(
                            subsector=footprints_germany.group.values
                        ),
                    ],
                    dim="subsector",
                )
                .expand_dims(sector=[footprints_city.name])
                .sortby("subsector")
            )

            anth_footprints = (
                xr.concat(
                    [
                        footprints_sums_city.assign_coords(
                            subsector=footprints_city.group.values
                        ),
                        footprints_sums_germany.assign_coords(
                            subsector=footprints_germany.group.values
                        ),
                    ],
                    dim="subsector",
                )
                .expand_dims(sector=[footprints_sums_city.name])
                .sortby("subsector")
            )

            self._footprint = (
                xr.concat([bio_footprints, anth_footprints], dim="sector")
                .stack(state=self.STATE_DIMS, measurement=self.MEASUREMENT_DIMS)
                .astype(np.float32)
                .compute()
            )
        return self._footprint

    def load_timeframe(
        self,
        start_time: np.datetime64,
        end_time: np.datetime64,
        start_mtime: np.datetime64,
        end_mtime: np.datetime64,
    ) -> xr.DataArray:
        return (
            self.footprint.unstack()
            .sel(Time=slice(start_time, end_time), MTime=slice(start_mtime, end_mtime))
            .stack(state=self.STATE_DIMS, measurement=self.MEASUREMENT_DIMS)
        )


class LoadFootprintAnthBioCO(FootprintLoader):
    ANTH_SECTOR_KEY = "CO2_ANT_TOTAL"
    BIO_SECTOR_KEY = "E_CO2_VPRM"
    CO_SECTOR_KEY = "E_CO"
    STATE_DIMS = ["subsector", "Time", "sector"]
    MEASUREMENT_DIMS = ["MTime", "MPlace", "species"]

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
            germany_suffixes if germany_suffixes is not None else ["_vprm_co", "_sums"]
        )
        self._time_resolution = time_resolution

        self._footprint = None

    @property
    def footprint(self):
        if self._footprint is None:
            city_folder = (
                self._remapped_data_path / self._season / self._city / self._prior_type
            )
            germany_folder = (
                self._remapped_data_path / self._season / "germany" / self._prior_type
            )
            time_res_string = (
                "" if self._time_resolution is None else f"_{self._time_resolution}H"
            )

            footprints_city = self._open_and_prepare(
                city_folder
                / f"remapped_footprints{self._city_suffixes[0]}{time_res_string}.nc",
            )[[self.BIO_SECTOR_KEY, self.CO_SECTOR_KEY]]

            footprints_germany = self._open_and_prepare(
                germany_folder
                / (
                    "remapped_footprints"
                    + f"{self._germany_suffixes[0]}{time_res_string}.nc"
                ),
            )[[self.BIO_SECTOR_KEY, self.CO_SECTOR_KEY]]

            footprints_sums_city = self._open_and_prepare(
                city_folder
                / f"remapped_footprints{self._city_suffixes[1]}{time_res_string}.nc",
            )[self.ANTH_SECTOR_KEY]
            footprints_sums_germany = self._open_and_prepare(
                germany_folder
                / (
                    "remapped_footprints"
                    + f"{self._germany_suffixes[1]}{time_res_string}.nc"
                )
            )[self.ANTH_SECTOR_KEY]

            bio_footprints = (
                xr.concat(
                    [
                        footprints_city[self.BIO_SECTOR_KEY].assign_coords(
                            subsector=footprints_city.group.values
                        ),
                        footprints_germany[self.BIO_SECTOR_KEY].assign_coords(
                            subsector=footprints_germany.group.values
                        ),
                    ],
                    dim="subsector",
                )
                .expand_dims(sector=[self.BIO_SECTOR_KEY], species=["CO2"])
                .sortby("subsector")
            )

            co_footprints = (
                xr.concat(
                    [
                        footprints_city[self.CO_SECTOR_KEY].assign_coords(
                            subsector=footprints_city.group.values
                        ),
                        footprints_germany[self.CO_SECTOR_KEY].assign_coords(
                            subsector=footprints_germany.group.values
                        ),
                    ],
                    dim="subsector",
                )
                .expand_dims(sector=[self.CO_SECTOR_KEY], species=["CO"])
                .sortby("subsector")
            )

            anth_footprints = (
                xr.concat(
                    [
                        footprints_sums_city.assign_coords(
                            subsector=footprints_city.group.values
                        ),
                        footprints_sums_germany.assign_coords(
                            subsector=footprints_germany.group.values
                        ),
                    ],
                    dim="subsector",
                )
                .expand_dims(sector=[footprints_sums_city.name], species=["CO2"])
                .sortby("subsector")
            )

            self._footprint = (
                xr.concat(
                    [bio_footprints, anth_footprints, co_footprints], dim="sector"
                )
                .sortby("species")
                .stack(state=self.STATE_DIMS, measurement=self.MEASUREMENT_DIMS)
                .astype(np.float32)
                .compute()
            )
        return self._footprint.fillna(0)

    def load_timeframe(
        self,
        start_time: np.datetime64,
        end_time: np.datetime64,
        start_mtime: np.datetime64,
        end_mtime: np.datetime64,
    ) -> xr.DataArray:
        return (
            self.footprint.unstack()
            .sel(Time=slice(start_time, end_time), MTime=slice(start_mtime, end_mtime))
            .stack(state=self.STATE_DIMS, measurement=self.MEASUREMENT_DIMS)
        )


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
    ):
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
        self._footprint = None

    @property
    def footprint(self):
        if self._footprint is None:
            footprints_city = self._open_and_prepare(self._footprint_file_city)[
                self.TOTAL_EMISSION_KEY
            ]
            footprints_germany = self._open_and_prepare(self._footprint_file_germany)[
                self.TOTAL_EMISSION_KEY
            ]

            self._footprint = xr.concat(
                [
                    footprints_city.assign_coords(
                        subsector=footprints_city.group.values
                    ),
                    footprints_germany.assign_coords(
                        subsector=footprints_germany.group.values
                    ),
                ],
                dim="subsector",
            )
            if self._leave_out is not None:
                self._footprint = self._footprint.isel(
                    MPlace=~np.isin(self._footprint.MPlace.values, self._leave_out)
                )
            if self._keep_only is not None:
                self._footprint = self._footprint.sel(MPlace=self._keep_only)

            self._footprint = (
                self._footprint.sortby("subsector")
                .stack(state=self.STATE_DIMS, measurement=self.MEASUREMENT_DIMS)
                .astype(np.float32)
                .compute()
            )
        return self._footprint

    def load_timeframe(
        self,
        start_time: np.datetime64,
        end_time: np.datetime64,
        start_mtime: np.datetime64,
        end_mtime: np.datetime64,
    ) -> xr.DataArray:
        return (
            self.footprint.unstack()
            .sel(Time=slice(start_time, end_time), MTime=slice(start_mtime, end_mtime))
            .stack(state=self.STATE_DIMS, measurement=self.MEASUREMENT_DIMS)
        )
