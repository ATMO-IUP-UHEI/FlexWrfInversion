""" This module descibes classes that can be used to load prior data for an inversion."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import xarray as xr

from flexwrfinversion.loaders.target import FlexibleTargetLoaderTotal, TargetLoader

FLOAT_PRECISION = np.float32


class PriorLoader(ABC):
    @abstractmethod
    def __init__(self, target_loader: TargetLoader, *args, **kwargs):
        self.target_loader = target_loader

    @property
    @abstractmethod
    def prior(self, *args, **kwargs) -> xr.DataArray:
        """Load the prior data.
        Returns:
            xr.DataArray: The prior data as 1D array. Coordinates should be stacked
                beforehand.
        """
        pass

    @abstractmethod
    def load_timeframe(
        self, start_time: np.datetime64, end_time: np.datetime64
    ) -> xr.DataArray:
        """Load the prior data.

        Args:
            start_time (np.datetime64): Start time of the emission timeframe
            end_time (np.datetime64): End time of the emissions timeframe
        Returns:
            xr.DataArray: The prior data as 1D array. Coordinates should be stacked
                beforehand.
        """
        pass

    @staticmethod
    def _get_emissions(
        emission_file_city: str | Path,
        emission_file_germany: str | Path,
        sector_key: str,
    ) -> xr.Dataset:
        emissions_city = xr.open_dataset(emission_file_city)[[sector_key]]
        emissions_germany = xr.open_dataset(emission_file_germany)[[sector_key]]
        return xr.concat(
            [
                emissions_city.assign_coords(subsector=emissions_city.group.values),
                emissions_germany.assign_coords(
                    subsector=emissions_germany.group.values
                ),
            ],
            dim="subsector",
        )


class FlatPrior(PriorLoader):
    """Class to build a flat prior."""

    def __init__(
        self,
        target_loader: TargetLoader,
        value: float = 0,
    ):
        """Prior with flat prior emissions.

        Args:
            target_loader (TargetLoader): Target loader used in the inversions
            value (float, optional): Value to use for prior in mol/m2/s. Defaults to 0.
        """
        super().__init__(target_loader)
        self._value = value
        self._prior = None

    @property
    def prior(self) -> xr.DataArray:
        if self._prior is None:
            self._prior = xr.full_like(
                self.target_loader.target, self._value, dtype=FLOAT_PRECISION
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


class PriorIsTarget(PriorLoader):
    """Class to build a prior from the target data."""

    def __init__(self, target_loader: TargetLoader):
        """Prior that is the same as the target data.

        Args:
            target_loader (TargetLoader): Target loader used in the inversions
        """
        super().__init__(target_loader)
        self._prior = None

    @property
    def prior(self) -> xr.DataArray:
        if self._prior is None:
            self._prior = self.target_loader.target.compute()
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
    def __init__(
        self,
        target_loader: FlexibleTargetLoaderTotal,
        anth_emission_file_city: str | Path,
        anth_emission_file_germany: str | Path,
        bio_emission_file_city: str | Path,
        bio_emission_file_germany: str | Path,
        ant_sector_key: str = "CO2_ANT_TOTAL",
        bio_sector_key: str = "E_CO2_VPRM",
    ):
        """Flexible implementation of prior that reduces anthropogenic emissions by 50%
        and adds biogenic emissions.

        Args:
            target_loader (FlexibleTargetLoaderTotal): Target loader used in the
                 inversion.
            anth_emission_file_city (str | Path): Emission file for the city that contains
                 `CO2_ANT_TOTAL`
            anth_emission_file_germany (str | Path): Emission file for germany that
                 contains `CO2_ANT_TOTAL`
            bio_emission_file_city (str | Path): Emission file for the city that contains
                 `E_CO2_VPRM`
            bio_emission_file_germany (str | Path): Emission file for germany that
                 contains `E_CO2_VPRM`
            ant_sector_key (str, optional): Sector key for anthropogenic emissions.
                 Defaults to "CO2_ANT_TOTAL".
            bio_sector_key (str, optional): Sector key for biogenic emissions. Defaults to
                 "E_CO2_VPRM".
        """
        super().__init__(target_loader)
        self._anth_emission_file_city = anth_emission_file_city
        self._anth_emission_file_germany = anth_emission_file_germany
        self._bio_emission_file_city = bio_emission_file_city
        self._bio_emission_file_germany = bio_emission_file_germany
        self.ant_sector_key = ant_sector_key
        self.bio_sector_key = bio_sector_key
        self._prior = None

    @property
    def prior(self):
        if self._prior is None:
            anth_emissions = self._get_emissions(
                self._anth_emission_file_city,
                self._anth_emission_file_germany,
                self.ant_sector_key,
            )
            bio_emissions = self._get_emissions(
                self._bio_emission_file_city,
                self._bio_emission_file_germany,
                self.bio_sector_key,
            )
            self._prior = (
                (
                    anth_emissions[self.ant_sector_key] / 2
                    + bio_emissions[self.bio_sector_key]
                    + np.abs(bio_emissions[self.bio_sector_key] / 2)
                )
                .rename("prior_emissions")
                .stack(state=self.target_loader.STATE_DIMS)
                .astype(FLOAT_PRECISION)
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


class PriorLoaderAnthBio_RelativeError_PointExtra(PriorLoader):
    def __init__(
        self,
        target_loader: FlexibleTargetLoaderTotal,
        anth_emission_file_city: str | Path,
        anth_emission_file_germany: str | Path,
        bio_emission_file_city: str | Path,
        bio_emission_file_germany: str | Path,
        point_emission_file_city: str | Path,
        point_emission_file_germany: str | Path,
        anth_emission_error: float,
        bio_emission_error: float,
        point_emission_error: float,
        ant_sector_key: str = "CO2_ANT_TOTAL",
        bio_sector_key: str = "E_CO2_VPRM",
        point_sector_key: str = "E_CO2TST",
    ):
        super().__init__(target_loader)
        self._anth_emission_file_city = anth_emission_file_city
        self._anth_emission_file_germany = anth_emission_file_germany
        self._bio_emission_file_city = bio_emission_file_city
        self._bio_emission_file_germany = bio_emission_file_germany
        self._point_emission_file_city = point_emission_file_city
        self._point_emission_file_germany = point_emission_file_germany
        self._anth_emission_error = anth_emission_error
        self._bio_emission_error = bio_emission_error
        self._point_emission_error = point_emission_error
        self.ant_sector_key = ant_sector_key
        self.bio_sector_key = bio_sector_key
        self.point_sector_key = point_sector_key
        self._prior = None

    @property
    def prior(self):
        if self._prior is None:
            anth_emissions = self._get_emissions(
                self._anth_emission_file_city,
                self._anth_emission_file_germany,
                self.ant_sector_key,
            )
            bio_emissions = self._get_emissions(
                self._bio_emission_file_city,
                self._bio_emission_file_germany,
                self.bio_sector_key,
            )
            point_emissions = self._get_emissions(
                self._point_emission_file_city,
                self._point_emission_file_germany,
                self.point_sector_key,
            )

            reduced_anth_emissions = (
                anth_emissions[self.ant_sector_key]
                - point_emissions[self.point_sector_key]
            )

            anth_emissions = (
                reduced_anth_emissions
                + self._anth_emission_error * np.abs(reduced_anth_emissions)
                + point_emissions[self.point_sector_key]
                + self._point_emission_error
                * np.abs(point_emissions[self.point_sector_key])
            ).expand_dims(sector=[self.ant_sector_key])
            bio_emissions = (
                bio_emissions[self.bio_sector_key]
                + self._bio_emission_error * np.abs(bio_emissions[self.bio_sector_key])
            ).expand_dims(sector=[self.bio_sector_key])
            self._prior = (
                xr.concat([anth_emissions, bio_emissions], dim="sector")
                .rename("prior_emissions")
                .sortby("sector")
                .sortby("subsector")
                .stack(state=self.target_loader.STATE_DIMS)
                .astype(FLOAT_PRECISION)
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


class PriorLoaderAnthBioCo_RelativeError_PointExtra(PriorLoader):
    def __init__(
        self,
        target_loader: FlexibleTargetLoaderTotal,
        anth_emission_file_city: str | Path,
        anth_emission_file_germany: str | Path,
        bio_emission_file_city: str | Path,
        bio_emission_file_germany: str | Path,
        point_emission_file_city: str | Path,
        point_emission_file_germany: str | Path,
        co_emission_file_city: str | Path,
        co_emission_file_germany: str | Path,
        anth_emission_error: float,
        bio_emission_error: float,
        point_emission_error: float,
        co_emission_error: float,
        ant_sector_key: str = "CO2_ANT_TOTAL",
        bio_sector_key: str = "E_CO2_VPRM",
        co_sector_key: str = "E_CO",
        point_sector_key: str = "E_CO2TST",
    ):
        super().__init__(target_loader)
        self._anth_emission_file_city = anth_emission_file_city
        self._anth_emission_file_germany = anth_emission_file_germany
        self._bio_emission_file_city = bio_emission_file_city
        self._bio_emission_file_germany = bio_emission_file_germany
        self._point_emission_file_city = point_emission_file_city
        self._point_emission_file_germany = point_emission_file_germany
        self._co_emission_file_city = co_emission_file_city
        self._co_emission_file_germany = co_emission_file_germany
        self._anth_emission_error = anth_emission_error
        self._bio_emission_error = bio_emission_error
        self._point_emission_error = point_emission_error
        self._co_emission_error = co_emission_error
        self.ant_sector_key = ant_sector_key
        self.bio_sector_key = bio_sector_key
        self.co_sector_key = co_sector_key
        self.point_sector_key = point_sector_key
        self._prior = None

    @property
    def prior(self):
        if self._prior is None:
            anth_emissions = self._get_emissions(
                self._anth_emission_file_city,
                self._anth_emission_file_germany,
                self.ant_sector_key,
            )
            bio_emissions = self._get_emissions(
                self._bio_emission_file_city,
                self._bio_emission_file_germany,
                self.bio_sector_key,
            )
            co_emissions = self._get_emissions(
                self._co_emission_file_city,
                self._co_emission_file_germany,
                self.co_sector_key,
            )
            point_emissions = self._get_emissions(
                self._point_emission_file_city,
                self._point_emission_file_germany,
                self.point_sector_key,
            )

            reduced_anth_emissions = (
                anth_emissions[self.ant_sector_key]
                - point_emissions[self.point_sector_key]
            )

            anth_emissions = (
                reduced_anth_emissions
                + self._anth_emission_error * np.abs(reduced_anth_emissions)
                + point_emissions[self.point_sector_key]
                + self._point_emission_error
                * np.abs(point_emissions[self.point_sector_key])
            ).expand_dims(sector=[self.ant_sector_key])

            bio_emissions = (
                bio_emissions[self.bio_sector_key]
                + self._bio_emission_error * np.abs(bio_emissions[self.bio_sector_key])
            ).expand_dims(sector=[self.bio_sector_key])

            co_emissions = (
                co_emissions[self.co_sector_key]
                + self._co_emission_error * np.abs(co_emissions[self.co_sector_key])
            ).expand_dims(sector=[self.co_sector_key])

            self._prior = (
                xr.concat([anth_emissions, bio_emissions, co_emissions], dim="sector")
                .rename("prior_emissions")
                .sortby("sector")
                .sortby("subsector")
                .stack(state=self.target_loader.STATE_DIMS)
                .astype(FLOAT_PRECISION)
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
