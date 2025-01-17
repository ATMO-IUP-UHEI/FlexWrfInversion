""" This module descibes classes that can be used to load/build prior covariance data for
     an inversion."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import xarray as xr

from flexwrfinversion.loaders.prior import (
    FlexiblePriorLoaderTotal_ShiftToBiospheric,
    PriorLoader,
)

FLOAT_PRECISION = np.float32


class TargetAsErrorNoCorrelation:
    def __init__(self, *args, **kwargs):
        raise ValueError("`TargetAsErrorNoCorrelation` is now called `TargetAsError`")


class RelativeErrorWithSpatialCorrelation:
    def __init__(self, *args, **kwargs):
        raise ValueError(
            "`RelativeErrorWithSpatialCorrelation` is now called `RelativeError`"
        )


class PriorCovarianceLoader(ABC):
    @abstractmethod
    def __init__(self, prior_loader: PriorLoader, *args, **kwargs):
        self.prior_loader = prior_loader
        self.spatial_correlation_path = None
        self._sector_correlation = None
        self._spatial_correlation = None
        self._temporal_correlation = None

    @property
    @abstractmethod
    def prior_std(self) -> xr.DataArray:
        """Property of prior standard deviation.
        Returns:
            xr.DataArray: The prior standard deviation data as 1D array. Unstacked
                coordinates are supposed to be passed.
        """
        pass

    @staticmethod
    def _to_two_dimensions(one_d_dataarray: xr.DataArray):
        """Convert a vector to two dimensions by renaming the dimensions.

        Args:
            one_d_dataarray (xr.DataArray): Vector to convert.

        Returns:
            xr.DataArray: Outer product of vector with itself.
        """
        vector0 = one_d_dataarray.rename(
            dict([(dim, f"{dim}0") for dim in one_d_dataarray.coords.keys()])
        )
        vector1 = one_d_dataarray.rename(
            dict([(dim, f"{dim}1") for dim in one_d_dataarray.coords.keys()])
        )
        return vector0 * vector1

    @property
    def spatial_correlation(self) -> xr.DataArray:
        """Full spatial correlation matrix (no temporal part included).

        Returns:
            xr.DataArray: Spatial correlation matrix.
        """
        if self._spatial_correlation is None:
            if self._spatial_correlation_path is None:
                spatial_coordinate_values = (
                    self.prior_loader.prior.unstack().subsector.values
                )
                self._spatial_correlation = xr.DataArray(
                    np.eye(
                        len(spatial_coordinate_values),
                        dtype=self.prior_loader.prior.dtype,
                    ),
                    coords=[
                        ("subsector0", spatial_coordinate_values),
                        ("subsector1", spatial_coordinate_values),
                    ],
                ).compute()
            else:
                self._spatial_correlation = xr.open_dataarray(
                    self._spatial_correlation_path
                ).compute()
            self._spatial_correlation = self._spatial_correlation.sortby(
                "subsector0"
            ).sortby("subsector1")
        return self._spatial_correlation

    @property
    def temporal_correlation(self) -> xr.DataArray:
        """Temporal correlation matrix (no spatial part included).

        Returns:
            xr.DataArray: Temporal correlation matrix.
        """
        if self._temporal_correlation is None:
            time_values = self.prior_loader.prior.unstack().Time.values
        return xr.DataArray(
            np.eye(
                len(time_values),
                dtype=self.prior_loader.prior.dtype,
            ),
            coords=[("Time0", time_values), ("Time1", time_values)],
        )

    @property
    def sector_correlation(self) -> xr.DataArray:
        if "sector" in self.prior_loader.prior.unstack().coords:
            if self._sector_correlation is None:
                correlation = self._to_two_dimensions(
                    xr.zeros_like(
                        self.prior_loader.prior.unstack().isel(Time=0, subsector=0)
                    )
                )
                correlation = correlation + np.eye(correlation.shape[0])
                self._sector_correlation = correlation.sortby("sector0").sortby(
                    "sector1"
                )
            return self._sector_correlation
        else:
            return None

    def compute_stacked_correlation(self, *correlations: xr.DataArray):
        """Compute the correlation of the given correlation matrices.

        Args:
            *correlations (xr.DataArray): Correlation matrices to multiply.

        Returns:
            xr.DataArray: The product of the correlation matrices.
        """
        correlation = 1
        for corr in correlations:
            if corr is None:
                continue
            correlation = correlation * corr

        return correlation.stack(
            state0=[
                state_dim + "0"
                for state_dim in self.prior_loader.target_loader.STATE_DIMS
            ],
            state1=[
                state_dim + "1"
                for state_dim in self.prior_loader.target_loader.STATE_DIMS
            ],
        )

    def load_timeframe(
        self, start_time: np.datetime64, end_time: np.datetime64
    ) -> xr.DataArray:
        unstacked_prior_std = self.prior_std.unstack()
        time_values = unstacked_prior_std.Time.sel(
            Time=slice(start_time, end_time)
        ).values
        correlation = self.compute_stacked_correlation(
            self.spatial_correlation,
            self.temporal_correlation.sel(Time0=time_values, Time1=time_values),
            self.sector_correlation,
        )

        return (
            (
                self._to_two_dimensions(
                    unstacked_prior_std.sel(Time=slice(start_time, end_time)).stack(
                        state=self.prior_loader.target_loader.STATE_DIMS
                    )
                )
                * correlation
            )
            .astype(FLOAT_PRECISION)
            .compute()
        )


class RelativeError(PriorCovarianceLoader):
    def __init__(
        self,
        prior_loader: FlexiblePriorLoaderTotal_ShiftToBiospheric,
        spatial_correlation_path: str | Path = None,
        relative_error: float = 1,
    ):
        """Prior covariance loader with standard deviation that is given as relative
        error to the prior. Spatial correlation can be used if given in proper format.

        Args:
            prior_loader (FlexiblePriorLoaderTotal_ShiftToBiospheric): Prior loader used
                 in the Inversion.
            spatial_correlation_path (str | Path, optional): Path to file with spatial
                 correlations. Defaults to None.
            relative_error (float, optional): Relative error to use. `1` corresponds.
                 to a 100% error. Defaults to 1.
        """
        super().__init__(prior_loader)
        self._spatial_correlation_path = spatial_correlation_path
        self._relative_error = relative_error
        self._prior_std = None

    @property
    def prior_std(self) -> xr.DataArray:
        if self._prior_std is None:
            self._prior_std = np.abs(self.prior_loader.prior) * self._relative_error
        return self._prior_std


class TargetAsError(PriorCovarianceLoader):
    def __init__(
        self,
        prior_loader: PriorLoader,
        minimum_error: float = None,
        spatial_correlation_path: str | Path = None,
    ):
        """Use target values as std for the covariance. Cannot use correlation.

        Args:
            prior_loader (PriorLoader): Prior loader of the inversion.
            minimum_error (float, optional): Minimal error to allow. Defaults to None.
        """
        super().__init__(prior_loader)
        self._prior_std = None
        self._minimum_error = minimum_error
        self._spatial_correlation_path = spatial_correlation_path

    @property
    def prior_std(self) -> xr.DataArray:
        if self._prior_std is None:
            self._prior_std = np.abs(self.prior_loader.target_loader.target)
            if self._minimum_error is not None:
                self._prior_std = xr.where(
                    self._prior_std < self._minimum_error,
                    self._minimum_error,
                    self._prior_std,
                )
            try:
                self._prior_std = self._prior_std.sortby("sector")
            except KeyError:
                pass
            try:
                self._prior_std = self._prior_std.sortby("subsector")
            except KeyError:
                pass
        return self._prior_std


class DifferenceOfPriorToTarget(TargetAsError):
    @property
    def prior_std(self) -> xr.DataArray:
        if self._prior_std is None:
            self._prior_std = np.abs(
                self.prior_loader.prior - self.prior_loader.target_loader.target
            )
            if self._minimum_error is not None:
                self._prior_std = xr.where(
                    self._prior_std < self._minimum_error,
                    self._minimum_error,
                    self._prior_std,
                )

        return self._prior_std


class TargetAsErrorWithCO_Correlation(PriorCovarianceLoader):
    def __init__(
        self,
        prior_loader: PriorLoader,
        anth_co_correlation: float = 0,
        spatial_correlation_path: str | Path = None,
        co2_minimum_error: float = None,
        co_minimum_error: float = None,
        ant_sector_key: str = "CO2_ANT_TOTAL",
        bio_sector_key: str = "E_CO2_VPRM",
        co_sector_key: str = "E_CO",
    ):
        """Prior covariance loader for anthropogenic and biogenic CO2 together with CO.
        A correlation is built into the covariace between CO2_anth and CO.

        Args:
            prior_loader (PriorLoader): Prior loader used in the inversion
            anth_co_correlation (float, optional): Correlation between anthropogenic CO2
                 and CO emissions. Defaults to 0.
            spatial_correlation_path (str | Path, optional): Path to the spatial
                 correlation to be used. Defaults to None.
            co2_minimum_error (float, optional): Minimum error for CO2. Defaults to None.
            co_minimum_error (float, optional): Minimum error for CO. Defaults to None.
            ant_sector_key (str, optional): Sector key for anthropogenic CO2. Defaults
                 to "CO2_ANT_TOTAL".
            bio_sector_key (str, optional): Sector key for biogenic CO2. Defaults to
                 "E_CO2_VPRM".
            co_sector_key (str, optional): Sector key for CO. Defaults to "E_CO".
        """
        super().__init__(prior_loader)
        self._anth_co_correlation = anth_co_correlation
        self._spatial_correlation_path = spatial_correlation_path
        self._co2_minimum_error = co2_minimum_error
        self._co_minimum_error = co_minimum_error
        self.ant_sector_key = ant_sector_key
        self.bio_sector_key = bio_sector_key
        self.co_sector_key = co_sector_key
        self._prior_std = None

    @property
    def prior_std(self) -> xr.DataArray:
        if self._prior_std is None:
            self._prior_std = np.abs(self.prior_loader.target_loader.target)
            if self._co2_minimum_error is not None:
                self._prior_std = xr.where(
                    (
                        self.prior_loader.target_loader.target.sector
                        == self.ant_sector_key
                    )
                    & (self._prior_std < self._co2_minimum_error),
                    self._co2_minimum_error,
                    self._prior_std,
                )
                self._prior_std = xr.where(
                    (
                        self.prior_loader.target_loader.target.sector
                        == self.bio_sector_key
                    )
                    & (self._prior_std < self._co2_minimum_error),
                    self._co2_minimum_error,
                    self._prior_std,
                )
            if self._co_minimum_error is not None:
                self._prior_std = xr.where(
                    (
                        self.prior_loader.target_loader.target.sector
                        == self.co_sector_key
                    )
                    & (self._prior_std < self._co_minimum_error),
                    self._co_minimum_error,
                    self._prior_std,
                )
                self._prior_std = self._prior_std.sortby("sector")
                self._prior_std = self._prior_std.sortby("subsector")
        return self._prior_std

    @property
    def sector_correlation(self) -> xr.DataArray:
        """Sector part of the correlation."""
        if self._sector_correlation is None:
            correlation = self._to_two_dimensions(
                xr.zeros_like(
                    self.prior_loader.prior.unstack().isel(Time=0, subsector=0)
                )
            )
            correlation = correlation + np.eye(correlation.shape[0])
            anth_index = np.argwhere(
                self.prior_loader.prior.unstack().sector.values
                == self.prior_loader.target_loader.ant_sector_key
            ).item()
            co_index = np.argwhere(
                self.prior_loader.prior.unstack().sector.values
                == self.prior_loader.target_loader.co_sector_key
            ).item()
            correlation[anth_index, co_index] = self._anth_co_correlation
            correlation[co_index, anth_index] = self._anth_co_correlation
            self._sector_correlation = correlation.sortby("sector0").sortby("sector1")
        return self._sector_correlation


class DifferenceOfPriorToTargetWithCO_Correlation(TargetAsErrorWithCO_Correlation):
    @property
    def prior_std(self) -> xr.DataArray:
        if self._prior_std is None:
            self._prior_std = np.abs(
                self.prior_loader.prior - self.prior_loader.target_loader.target
            )
            if self._co2_minimum_error is not None:
                self._prior_std = xr.where(
                    (
                        (
                            self.prior_loader.target_loader.target.sector
                            == self.ant_sector_key
                        )
                        & (self._prior_std < self._co2_minimum_error)
                    ),
                    self._co2_minimum_error,
                    self._prior_std,
                )
                self._prior_std = xr.where(
                    (
                        self.prior_loader.target_loader.target.sector
                        == self.bio_sector_key
                    )
                    & (self._prior_std < self._co2_minimum_error),
                    self._co2_minimum_error,
                    self._prior_std,
                )
            if self._co_minimum_error is not None:
                self._prior_std = xr.where(
                    (
                        self.prior_loader.target_loader.target.sector
                        == self.co_sector_key
                    )
                    & (self._prior_std < self._co_minimum_error),
                    self._co_minimum_error,
                    self._prior_std,
                )
        return self._prior_std
