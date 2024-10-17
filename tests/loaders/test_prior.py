import numpy as np
import pytest
import xarray as xr


class Test_FlatPrior:
    def test_prior(self, flat_prior, flat_prior_with_co):
        prior = flat_prior.prior
        assert prior is not None
        assert isinstance(prior, xr.DataArray)
        assert len(prior.dims) == 1
        assert set(prior.dims) == {"state"}
        assert prior.max().item() == pytest.approx(0.1, abs=1e-6)
        assert {"CO2_ANT_TOTAL", "E_CO2_VPRM"} == set(prior.sector.values)

        prior = flat_prior_with_co.prior
        assert prior is not None
        assert isinstance(prior, xr.DataArray)
        assert len(prior.dims) == 1
        assert set(prior.dims) == {"state"}
        assert prior.max().item() == pytest.approx(0.1, abs=1e-6)
        assert {"CO2_ANT_TOTAL", "E_CO2_VPRM", "E_CO"} == set(prior.sector.values)

    def test_load_timeframe(self, flat_prior, flat_prior_with_co):
        prior = flat_prior.prior
        start_time = prior["Time"].values[0]
        end_time = prior["Time"].values[3]

        prior_timeframe = flat_prior.load_timeframe(
            start_time=start_time,
            end_time=end_time,
        )

        assert prior_timeframe is not None
        assert isinstance(prior_timeframe, xr.DataArray)
        assert len(prior_timeframe.dims) == 1
        assert set(prior_timeframe.dims) == {"state"}
        assert (prior_timeframe == 0.1).all()
        assert {"CO2_ANT_TOTAL", "E_CO2_VPRM"} == set(prior.sector.values)

        prior_timeframe_with_co = flat_prior_with_co.load_timeframe(
            start_time=start_time,
            end_time=end_time,
        )
        assert prior_timeframe_with_co is not None
        assert isinstance(prior_timeframe_with_co, xr.DataArray)
        assert len(prior_timeframe_with_co.dims) == 1
        assert set(prior_timeframe_with_co.dims) == {"state"}
        assert (prior_timeframe_with_co == 0.1).all()
        assert {"CO2_ANT_TOTAL", "E_CO2_VPRM", "E_CO"} == set(
            prior_timeframe_with_co.sector.values
        )


class Test_FlexiblePriorLoaderTotal_ShiftToBiospheric:
    def test_prior(self, flexible_prior_loader_total_shift_to_biospheric):
        prior = flexible_prior_loader_total_shift_to_biospheric.prior
        assert prior is not None
        assert isinstance(prior, xr.DataArray)
        assert len(prior.dims) == 1
        assert set(prior.dims) == {"state"}

    def test_load_timeframe(self, flexible_prior_loader_total_shift_to_biospheric):
        prior = flexible_prior_loader_total_shift_to_biospheric.prior
        start_time = prior["Time"].values[0]
        end_time = prior["Time"].values[3]

        prior_timeframe = (
            flexible_prior_loader_total_shift_to_biospheric.load_timeframe(
                start_time=start_time,
                end_time=end_time,
            )
        )

        assert prior_timeframe is not None
        assert isinstance(prior_timeframe, xr.DataArray)
        assert len(prior_timeframe.dims) == 1
        assert set(prior_timeframe.dims) == {"state"}


class Test_PriorLoaderAnthBio_RelativeError_PointExtra:
    def test_prior(self, prior_loader_anth_bio_relative_error_point_extra):
        prior = prior_loader_anth_bio_relative_error_point_extra.prior
        target = prior_loader_anth_bio_relative_error_point_extra.target_loader.target
        anth_prior = prior.sel(sector="CO2_ANT_TOTAL")
        bio_prior = prior.sel(sector="E_CO2_VPRM")
        anth_target = target.sel(sector="CO2_ANT_TOTAL")
        bio_target = target.sel(sector="E_CO2_VPRM")

        rel_difference_anth = 1 - np.abs(anth_prior / anth_target)
        rel_difference_bio = np.abs(1 - np.abs(bio_prior / bio_target))

        assert prior is not None
        assert isinstance(prior, xr.DataArray)
        assert len(prior.dims) == 1
        assert set(prior.dims) == {"state"}
        assert not np.allclose(prior, target, atol=0, rtol=1e-3)
        assert ((rel_difference_anth >= 0.1) & (rel_difference_anth <= 0.5)).all()
        assert np.allclose(
            rel_difference_bio.where(~rel_difference_bio.isnull(), drop=True),
            0.3,
            atol=0,
            rtol=1e-3,
        )
