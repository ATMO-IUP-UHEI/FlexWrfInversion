import numpy as np
import pytest
import xarray as xr

from flexwrfinversion.loaders.prior_covariance import (
    DifferenceOfPriorToTargetMinimumFromFile,
    DifferenceOfPriorToTargetMinimumFromFile_Scalable,
    DifferenceOfPriorToTargetMinimumFromFile_ScalableFromMean,
    RelativeError,
)


class Test_RelativeError:
    @pytest.mark.parametrize("relative_error", [0.5, 1])
    def test_prior_std(
        self, flexible_prior_loader_total_shift_to_biospheric, relative_error
    ):
        prior_covariance_loader = RelativeError(
            prior_loader=flexible_prior_loader_total_shift_to_biospheric,
            relative_error=relative_error,
        )
        assert prior_covariance_loader.prior_std is not None
        assert isinstance(prior_covariance_loader.prior_std, xr.DataArray)
        assert len(prior_covariance_loader.prior_std.dims) == 1
        assert (prior_covariance_loader.prior_std >= 0).all()
        assert np.allclose(
            prior_covariance_loader.prior_std,
            np.abs(flexible_prior_loader_total_shift_to_biospheric.prior)
            * relative_error,
        )

    def test_load_timeframe(self, relative_error):
        start_time = relative_error.prior_std.Time.values[0]
        end_time = relative_error.prior_std.Time.values[4]
        prior_covariance = relative_error.load_timeframe(start_time, end_time)
        assert prior_covariance is not None
        assert isinstance(prior_covariance, xr.DataArray)
        assert len(prior_covariance.dims) == 2
        assert ((prior_covariance != 0) == np.eye(prior_covariance.shape[0])).all()
        assert set(prior_covariance.dims) == {"state0", "state1"}

    def test_spatial_correlation(self, relative_error):
        assert relative_error.spatial_correlation is not None
        assert isinstance(relative_error.spatial_correlation, xr.DataArray)
        assert len(relative_error.spatial_correlation.dims) == 2
        assert (
            relative_error.spatial_correlation
            == np.eye(relative_error.spatial_correlation.shape[0])
        ).all()
        assert set(relative_error.spatial_correlation.dims) == {
            "subsector0",
            "subsector1",
        }


class Test_TargetAsError:
    def test_prior_std(self, target_as_error):
        prior_std = target_as_error.prior_std
        assert prior_std is not None
        assert isinstance(prior_std, xr.DataArray)
        assert len(prior_std.dims) == 1
        assert (prior_std >= 0).all()
        assert np.allclose(
            prior_std - np.abs(target_as_error.prior_loader.target_loader.target),
            0,
        )

    def test_prior_std_with_minimum(self, target_as_error_with_minimum):
        prior_std = target_as_error_with_minimum.prior_std
        assert prior_std is not None
        assert isinstance(prior_std, xr.DataArray)
        assert len(prior_std.dims) == 1
        assert (prior_std >= target_as_error_with_minimum._minimum_error).all()
        assert np.allclose(
            prior_std.where(
                prior_std > target_as_error_with_minimum._minimum_error,
                drop=True,
            )
            - np.abs(
                target_as_error_with_minimum.prior_loader.target_loader.target  # noqa
            ).where(
                np.abs(
                    target_as_error_with_minimum.prior_loader.target_loader.target  # noqa
                )
                > target_as_error_with_minimum._minimum_error,
                drop=True,
            ),
            0,
        )

    def test_load_timeframe(self, target_as_error):
        start_time = target_as_error.prior_std.Time.values[0]
        end_time = target_as_error.prior_std.Time.values[4]
        prior_covariance = target_as_error.load_timeframe(start_time, end_time)
        assert prior_covariance is not None
        assert isinstance(prior_covariance, xr.DataArray)
        assert len(prior_covariance.dims) == 2
        assert ((prior_covariance != 0) == np.eye(prior_covariance.shape[0])).all()
        assert set(prior_covariance.dims) == {"state0", "state1"}


class Test_DifferenceOfPriorToTarget:
    def test_prior_std(self, difference_of_prior_to_target):
        prior_std = difference_of_prior_to_target.prior_std
        assert prior_std is not None
        assert isinstance(prior_std, xr.DataArray)
        assert len(prior_std.dims) == 1
        assert (prior_std >= 0).all()
        assert (
            prior_std
            == np.abs(
                difference_of_prior_to_target.prior_loader.prior
                - difference_of_prior_to_target.prior_loader.target_loader.target
            )
        ).all()


class Test_TargetAsErrorWithCO_Correlation:
    def test_prior_std(self, target_as_error_with_co_correlation):
        prior_std = target_as_error_with_co_correlation.prior_std
        assert prior_std is not None
        assert isinstance(prior_std, xr.DataArray)
        assert len(prior_std.dims) == 1
        assert (prior_std >= 0).all()
        assert np.allclose(
            prior_std,
            np.abs(
                target_as_error_with_co_correlation.prior_loader.target_loader.target
            ),
        )

    def test_spatial_correlation(self, target_as_error_with_co_correlation):
        spatial_correlation = target_as_error_with_co_correlation.spatial_correlation
        assert spatial_correlation is not None
        assert isinstance(spatial_correlation, xr.DataArray)
        assert len(spatial_correlation.dims) == 2
        assert (spatial_correlation == np.eye(spatial_correlation.shape[0])).all()
        assert set(spatial_correlation.dims) == {
            "subsector0",
            "subsector1",
        }

    def test_sector_correlation(self, target_as_error_with_co_correlation):
        sector_correlation = target_as_error_with_co_correlation.sector_correlation
        assert sector_correlation is not None
        assert isinstance(sector_correlation, xr.DataArray)
        assert len(sector_correlation.dims) == 2
        assert (np.diag(sector_correlation) == 1).all()
        assert set(sector_correlation.dims) == {
            "sector0",
            "sector1",
        }
        assert (
            sector_correlation.unstack().sel(sector0="CO2_ANT_TOTAL", sector1="E_CO")
            == 0.5
        )
        assert (
            sector_correlation.unstack().sel(sector0="E_CO", sector1="CO2_ANT_TOTAL")
            == 0.5
        )
        assert np.count_nonzero(sector_correlation == 0.5) == 2

    def test_load_timeframe(self, target_as_error_with_co_correlation):
        start_time = target_as_error_with_co_correlation.prior_std.Time.values[0]
        end_time = target_as_error_with_co_correlation.prior_std.Time.values[4]
        prior_covariance = target_as_error_with_co_correlation.load_timeframe(
            start_time, end_time
        )
        assert prior_covariance is not None
        assert isinstance(prior_covariance, xr.DataArray)
        assert len(prior_covariance.dims) == 2
        assert set(prior_covariance.dims) == {"state0", "state1"}

    def test_load_timeframe_with_minimum(
        self, target_as_error_with_co_correlation_with_minimum
    ):
        start_time = (
            target_as_error_with_co_correlation_with_minimum.prior_std.Time.values[0]
        )
        end_time = (
            target_as_error_with_co_correlation_with_minimum.prior_std.Time.values[4]
        )
        prior_covariance = (
            target_as_error_with_co_correlation_with_minimum.load_timeframe(
                start_time, end_time
            )
        )
        assert prior_covariance is not None
        assert isinstance(prior_covariance, xr.DataArray)
        assert len(prior_covariance.dims) == 2
        assert set(prior_covariance.dims) == {"state0", "state1"}

        assert np.isclose(
            prior_covariance.where(
                (prior_covariance.sector0 == "CO2_ANT_TOTAL")
                & (prior_covariance.sector1 == "CO2_ANT_TOTAL")
                & (prior_covariance > 0),
                drop=True,
            )
            .min()
            .item(),
            1e-12,
            atol=0,
            rtol=1e-3,
        )
        assert np.isclose(
            prior_covariance.where(
                (prior_covariance.sector0 == "E_CO2_VPRM")
                & (prior_covariance.sector1 == "E_CO2_VPRM")
                & (prior_covariance > 0),
                drop=True,
            )
            .min()
            .item(),
            1e-12,
            atol=0,
            rtol=1e-3,
        )
        assert np.isclose(
            prior_covariance.where(
                (prior_covariance.sector0 == "E_CO")
                & (prior_covariance.sector1 == "E_CO")
                & (prior_covariance > 0),
                drop=True,
            )
            .min()
            .item(),
            4e-12,
            atol=0,
            rtol=1e-3,
        )


class Test_DifferenceOfPriorToTargetWithCO_Correlation:
    def test_prior_std(self, difference_of_prior_to_target_with_co_correlation):
        prior_std = difference_of_prior_to_target_with_co_correlation.prior_std
        assert prior_std is not None
        assert isinstance(prior_std, xr.DataArray)
        assert len(prior_std.dims) == 1
        assert (prior_std >= 0).all()
        assert (
            prior_std
            == np.abs(
                difference_of_prior_to_target_with_co_correlation.prior_loader.prior
                - difference_of_prior_to_target_with_co_correlation.prior_loader.target_loader.target  # noqa
            )
        ).all()


class Test_DifferenceOfPriorToTargetMinimumFromFile:
    def test_prior_std(self, difference_of_prior_to_target_minimum_from_file):
        prior_std = difference_of_prior_to_target_minimum_from_file.prior_std
        target = (
            difference_of_prior_to_target_minimum_from_file.prior_loader.target_loader.target  # noqa
        )
        prior = difference_of_prior_to_target_minimum_from_file.prior_loader.prior
        abs_diff = np.abs(prior - target)
        assert prior_std is not None
        assert isinstance(prior_std, xr.DataArray)
        assert len(prior_std.dims) == 1
        assert (prior_std >= 0).all()
        assert (prior_std >= abs_diff.mean()).all()
        assert prior_std.min() == abs_diff.mean()
        assert prior_std.dims == target.dims
        assert prior_std.shape == target.shape

    def test_minimum_error(self, difference_of_prior_to_target_minimum_from_file):
        minimum_error = difference_of_prior_to_target_minimum_from_file.minimum_error
        target = (
            difference_of_prior_to_target_minimum_from_file.prior_loader.target_loader.target  # noqa
        )
        prior = difference_of_prior_to_target_minimum_from_file.prior_loader.prior
        abs_diff = np.abs(prior - target)

        assert minimum_error is not None
        assert isinstance(minimum_error, xr.DataArray)
        assert (minimum_error == abs_diff.mean()).all()
        assert minimum_error.dims == target.dims
        assert minimum_error.shape == target.shape


class Test_DifferenceOfPriorToTargetMinimumFromFile_Scalable:
    def _write_minimum_error_file(self, tmp_path, flat_prior):
        minimum_file = tmp_path / "minimum_error.nc"
        target = flat_prior.target_loader.target.unstack()
        prior = flat_prior.prior.unstack()
        abs_diff = np.abs(target - prior)
        (xr.ones_like(target) * abs_diff.mean().item()).to_netcdf(minimum_file)
        return minimum_file

    def test_prior_std_scale_zero_matches_base(self, tmp_path, flat_prior):
        minimum_file = self._write_minimum_error_file(tmp_path, flat_prior)
        base = DifferenceOfPriorToTargetMinimumFromFile(
            prior_loader=flat_prior,
            minimum_error_file=minimum_file,
        )
        scalable = DifferenceOfPriorToTargetMinimumFromFile_Scalable(
            prior_loader=flat_prior,
            minimum_error_file=minimum_file,
            flat_component_scale=0.0,
        )
        assert np.allclose(scalable.prior_std, base.prior_std)

    def test_prior_std_scale_one_flattens_subsector(self, tmp_path, flat_prior):
        minimum_file = self._write_minimum_error_file(tmp_path, flat_prior)
        base = DifferenceOfPriorToTargetMinimumFromFile(
            prior_loader=flat_prior,
            minimum_error_file=minimum_file,
        )
        scalable = DifferenceOfPriorToTargetMinimumFromFile_Scalable(
            prior_loader=flat_prior,
            minimum_error_file=minimum_file,
            flat_component_scale=1.0,
        )

        base_unstacked = base.prior_std.unstack()
        scalable_unstacked = scalable.prior_std.unstack()

        assert "subsector" in scalable_unstacked.dims
        assert scalable_unstacked.std("subsector").fillna(0).max().item() == 0

        expected_mean = base_unstacked.mean("subsector")
        expected_flat = expected_mean.broadcast_like(base_unstacked)
        assert np.allclose(scalable_unstacked, expected_flat)

    def test_flat_subsectors_only_affects_subset(self, tmp_path, flat_prior):
        minimum_file = self._write_minimum_error_file(tmp_path, flat_prior)
        base = DifferenceOfPriorToTargetMinimumFromFile(
            prior_loader=flat_prior,
            minimum_error_file=minimum_file,
        )

        subsectors = list(base.prior_std.unstack().subsector.values)
        assert len(subsectors) >= 2
        subset = subsectors[:2]

        scalable = DifferenceOfPriorToTargetMinimumFromFile_Scalable(
            prior_loader=flat_prior,
            minimum_error_file=minimum_file,
            flat_component_scale=1.0,
            flat_subsectors=repr(subset),
        )

        base_unstacked = base.prior_std.unstack()
        scalable_unstacked = scalable.prior_std.unstack()

        expected_mean_subset = base_unstacked.sel(subsector=subset).mean("subsector")
        expected_flat_subset = expected_mean_subset.broadcast_like(
            base_unstacked.sel(subsector=subset)
        )

        assert np.allclose(
            scalable_unstacked.sel(subsector=subset),
            expected_flat_subset,
        )

        remaining = [s for s in subsectors if s not in subset]
        assert np.allclose(
            scalable_unstacked.sel(subsector=remaining),
            base_unstacked.sel(subsector=remaining),
        )


class Test_DifferenceOfPriorToTargetMinimumFromFile_ScalableFromMean:
    def _write_minimum_error_file(self, tmp_path, flat_prior):
        minimum_file = tmp_path / "minimum_error.nc"
        target = flat_prior.target_loader.target.unstack()
        prior = flat_prior.prior.unstack()
        abs_diff = np.abs(target - prior)
        (xr.ones_like(target) * abs_diff.mean().item()).to_netcdf(minimum_file)
        return minimum_file

    def test_prior_std_scale_zero_matches_base(self, tmp_path, flat_prior):
        minimum_file = self._write_minimum_error_file(tmp_path, flat_prior)
        base = DifferenceOfPriorToTargetMinimumFromFile(
            prior_loader=flat_prior,
            minimum_error_file=minimum_file,
        )
        scalable = DifferenceOfPriorToTargetMinimumFromFile_ScalableFromMean(
            prior_loader=flat_prior,
            minimum_error_file=minimum_file,
            flat_component_scale=0.0,
        )
        assert np.allclose(scalable.prior_std, base.prior_std)

    def test_prior_std_scale_one_flattens_sector(self, tmp_path, flat_prior):
        minimum_file = self._write_minimum_error_file(tmp_path, flat_prior)
        base = DifferenceOfPriorToTargetMinimumFromFile(
            prior_loader=flat_prior,
            minimum_error_file=minimum_file,
        )
        scalable = DifferenceOfPriorToTargetMinimumFromFile_ScalableFromMean(
            prior_loader=flat_prior,
            minimum_error_file=minimum_file,
            flat_component_scale=1.0,
        )

        base_unstacked = base.prior_std.unstack()
        scalable_unstacked = scalable.prior_std.unstack()

        assert "sector" in base_unstacked.dims
        assert "sector" in scalable_unstacked.dims

        expected_mean = base_unstacked.mean("sector")
        expected_flat = expected_mean.broadcast_like(base_unstacked)

        assert np.allclose(scalable_unstacked, expected_flat)
        assert scalable_unstacked.std("sector").fillna(0).max().item() == 0

    def test_flat_subsectors_only_affects_subset(self, tmp_path, flat_prior):
        minimum_file = self._write_minimum_error_file(tmp_path, flat_prior)
        base = DifferenceOfPriorToTargetMinimumFromFile(
            prior_loader=flat_prior,
            minimum_error_file=minimum_file,
        )

        base_unstacked = base.prior_std.unstack()
        subsectors = list(base_unstacked.subsector.values)
        assert len(subsectors) >= 2
        subset = subsectors[:2]

        scalable = DifferenceOfPriorToTargetMinimumFromFile_ScalableFromMean(
            prior_loader=flat_prior,
            minimum_error_file=minimum_file,
            flat_component_scale=1.0,
            flat_subsectors=repr(subset),
        )

        scalable_unstacked = scalable.prior_std.unstack()

        expected_mean_subset = base_unstacked.sel(subsector=subset).mean("sector")
        expected_flat_subset = expected_mean_subset.broadcast_like(
            base_unstacked.sel(subsector=subset)
        )
        assert np.allclose(
            scalable_unstacked.sel(subsector=subset),
            expected_flat_subset,
        )

        remaining = [s for s in subsectors if s not in subset]
        assert np.allclose(
            scalable_unstacked.sel(subsector=remaining),
            base_unstacked.sel(subsector=remaining),
        )
