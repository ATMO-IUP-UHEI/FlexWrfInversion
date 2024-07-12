from pathlib import Path

import pytest
import xarray as xr

from flexwrfinversion.loaders.prior import (
    FlatPrior,
    FlexiblePriorLoaderTotal_ShiftToBiospheric,
    ShiftToBiospheric,
)
from flexwrfinversion.loaders.target import (
    FlexibleTargetLoaderTotal,
    TargetLoaderAnthAndBioSectors,
    TargetLoaderAnthBioCO,
    TargetLoaderTotalInCity,
)

EXAMPLE_DIRECTORY_0 = Path(__file__).parent.parent / "data" / "example_directory_0"


@pytest.fixture
def shift_to_biospheric():
    target_loader = TargetLoaderTotalInCity(
        remapped_data_path=EXAMPLE_DIRECTORY_0 / "remapped_data",
        season="spring",
        city="munich",
        prior_type="true",
        time_resolution=3,
    )
    return ShiftToBiospheric(target_loader=target_loader)


@pytest.fixture
def flat_prior():
    target_loader = TargetLoaderAnthAndBioSectors(
        remapped_data_path=EXAMPLE_DIRECTORY_0 / "remapped_data",
        season="spring",
        city="munich",
        prior_type="true",
        time_resolution=3,
    )
    return FlatPrior(target_loader=target_loader, value=0.1)


@pytest.fixture
def flat_prior_with_co():
    target_loader = TargetLoaderAnthBioCO(
        remapped_data_path=EXAMPLE_DIRECTORY_0 / "remapped_data",
        season="spring",
        city="munich",
        prior_type="true",
        time_resolution=3,
    )
    return FlatPrior(target_loader=target_loader, value=0.1)


@pytest.fixture
def flexible_prior_loader_total_shift_to_biospheric():
    target_loader = FlexibleTargetLoaderTotal(
        EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_true_emissions_sums_3H.nc",
        EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_true_emissions_sums_3H.nc",
    )
    return FlexiblePriorLoaderTotal_ShiftToBiospheric(
        target_loader=target_loader,
        anth_emission_file_city=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_true_emissions_sums_3H.nc",
        anth_emission_file_germany=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_true_emissions_sums_3H.nc",
        bio_emission_file_city=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "munich"
        / "true"
        / "remapped_true_emissions_3H.nc",
        bio_emission_file_germany=EXAMPLE_DIRECTORY_0
        / "remapped_data"
        / "spring"
        / "germany"
        / "true"
        / "remapped_true_emissions_vprm_co_3H.nc",
    )


class Test_ShiftToBiospheric:
    def test_prior(self, shift_to_biospheric):
        assert shift_to_biospheric.prior is not None
        assert isinstance(shift_to_biospheric.prior, xr.DataArray)
        assert len(shift_to_biospheric.prior.dims) == 1
        assert (
            shift_to_biospheric.prior != shift_to_biospheric.target_loader.target
        ).any()
        assert (
            (shift_to_biospheric.prior > 0)
            == (shift_to_biospheric.target_loader.target > 0)
        ).all()
        assert set(shift_to_biospheric.prior.dims) == {"state"}

    def test_load_timeframe(self, shift_to_biospheric):
        prior = shift_to_biospheric.prior
        start_time = prior["Time"].values[0]
        end_time = prior["Time"].values[3]

        prior_timeframe = shift_to_biospheric.load_timeframe(
            start_time=start_time,
            end_time=end_time,
        )

        assert prior_timeframe is not None
        assert isinstance(prior_timeframe, xr.DataArray)
        assert len(prior_timeframe.dims) == 1
        assert set(prior_timeframe.dims) == {"state"}


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
