""" This module descibes classes that can be used to load prior data for an inversion."""

from abc import ABC, abstractmethod

import numpy as np
import xarray as xr

from flexwrfinversion.loaders.target import TargetLoader, TargetLoaderTotalInCity


class PriorLoader(ABC):
    @abstractmethod
    def __init__(self, target_loader: TargetLoader, *args, **kwargs):
        pass

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
        self.target_loader = target_loader

    @property
    def prior(self) -> xr.DataArray:
        target_emissions = self.target_loader.target_with_additional_sectors
        return (
            target_emissions[self.ANTHROPOGENIC_EMISSION_KEY] / 2
            + target_emissions[self.BIOGENIC_EMISSION_KEY]
            + np.abs(target_emissions[self.BIOGENIC_EMISSION_KEY] / 2)
        ).rename(self.TOTAL_EMISSION_KEY)

    def load_timeframe(
        self, start_time: np.datetime64, end_time: np.datetime64
    ) -> xr.DataArray:
        return (
            self.prior.unstack()
            .sel(Time=slice(start_time, end_time))
            .stack(state=self.target_loader.STATE_DIMS)
        )
