""" This module descibes classes that can be used to load prior data for an inversion."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import xarray as xr

from flexwrfinversion.loaders.target import (
    FlexibleTargetLoaderTotal,
    TargetLoader,
    TargetLoaderTotalInCity,
)


class PriorLoader(ABC):
    @abstractmethod
    def __init__(self, target_loader: TargetLoader, *args, **kwargs):
        self.target_loader = target_loader

    @property
    @abstractmethod
    def prior(self, *args, **kwargs) -> xr.DataArray:
        """Load the prior data
        Returns:
            xr.DataArray: The prior data as 1D array. Coordinates should be stacked
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


class ShiftToBiospheric(PriorLoader):
    """Class to build prior that reduces anthropogenic emissions by 50% and adds biogenic
    emissions."""

    ANTHROPOGENIC_EMISSION_KEY = "CO2_ANT_TOTAL"
    BIOGENIC_EMISSION_KEY = "E_CO2_VPRM"
    TOTAL_EMISSION_KEY = "CO2_TOTAL"

    def __init__(
        self,
        target_loader: TargetLoaderTotalInCity,
    ):
        super().__init__(target_loader)
        self._prior = None

    @property
    def prior(self) -> xr.DataArray:
        if self._prior is None:
            target_emissions = self.target_loader.target_with_additional_sectors
            self._prior = (
                (
                    target_emissions[self.ANTHROPOGENIC_EMISSION_KEY] / 2
                    + target_emissions[self.BIOGENIC_EMISSION_KEY]
                    + np.abs(target_emissions[self.BIOGENIC_EMISSION_KEY] / 2)
                )
                .rename(self.TOTAL_EMISSION_KEY)
                .astype(np.float32)
                .compute()
            )
        return self._prior

    def load_timeframe(
        self, start_time: np.datetime64, end_time: np.datetime64
    ) -> xr.DataArray:
        return (
            self.prior.unstack()
            .sel(Time=slice(start_time, end_time))
            .stack(state=self.target_loader.STATE_DIMS)
        )


class FlatPrior(PriorLoader):
    """Class to build a flat prior."""

    def __init__(
        self,
        target_loader: TargetLoader,
        value: float = 0,
    ):
        super().__init__(target_loader)
        self._value = value
        self._prior = None

    @property
    def prior(self) -> xr.DataArray:
        if self._prior is None:
            self._prior = xr.full_like(
                self.target_loader.target, self._value, dtype=np.float32
            ).compute()
        return self._prior

    def load_timeframe(
        self, start_time: np.datetime64, end_time: np.datetime64
    ) -> xr.DataArray:
        return (
            self.prior.unstack()
            .sel(Time=slice(start_time, end_time))
            .stack(state=self.target_loader.STATE_DIMS)
        )


class FlexiblePriorLoaderTotal_ShiftToBiospheric(PriorLoader):
    ANTH_SECTOR_KEY = "CO2_ANT_TOTAL"
    BIO_SECTOR_KEY = "E_CO2_VPRM"

    def __init__(
        self,
        target_loader: FlexibleTargetLoaderTotal,
        anth_emission_file_city: str | Path,
        anth_emission_file_germany: str | Path,
        bio_emission_file_city: str | Path,
        bio_emission_file_germany: str | Path,
    ):
        super().__init__(target_loader)
        self._anth_emission_file_city = anth_emission_file_city
        self._anth_emission_file_germany = anth_emission_file_germany
        self._bio_emission_file_city = bio_emission_file_city
        self._bio_emission_file_germany = bio_emission_file_germany
        self._prior = None

    @property
    def prior(self):
        if self._prior is None:
            anth_emission_city = xr.open_dataset(self._anth_emission_file_city)[
                [self.ANTH_SECTOR_KEY]
            ]
            anth_emission_germany = xr.open_dataset(self._anth_emission_file_germany)[
                [self.ANTH_SECTOR_KEY]
            ]
            bio_emission_city = xr.open_dataset(self._bio_emission_file_city)[
                [self.BIO_SECTOR_KEY]
            ]
            bio_emission_germany = xr.open_dataset(self._bio_emission_file_germany)[
                [self.BIO_SECTOR_KEY]
            ]

            anth_emissions = xr.concat(
                [
                    anth_emission_city.assign_coords(
                        subsector=anth_emission_city.group.values
                    ),
                    anth_emission_germany.assign_coords(
                        subsector=anth_emission_germany.group.values
                    ),
                ],
                dim="subsector",
            )
            bio_emissions = xr.concat(
                [
                    bio_emission_city.assign_coords(
                        subsector=bio_emission_city.group.values
                    ),
                    bio_emission_germany.assign_coords(
                        subsector=bio_emission_germany.group.values
                    ),
                ],
                dim="subsector",
            )
            self._prior = (
                (
                    anth_emissions[self.ANTH_SECTOR_KEY] / 2
                    + bio_emissions[self.BIO_SECTOR_KEY]
                    + np.abs(bio_emissions[self.BIO_SECTOR_KEY] / 2)
                )
                .rename(self.target_loader.TOTAL_EMISSION_KEY)
                .stack(state=self.target_loader.STATE_DIMS)
                .astype(np.float32)
                .compute()
            )
        return self._prior

    def load_timeframe(
        self, start_time: np.datetime64, end_time: np.datetime64
    ) -> xr.DataArray:
        return (
            self.prior.unstack()
            .sel(Time=slice(start_time, end_time))
            .stack(state=self.target_loader.STATE_DIMS)
        )
