"""
Independent pure-Python reference implementation of the Conley spatial HAC
meat matrix, used to validate the Rust backend (_conley_meat_rs).

This module provides a brute-force O(n^2) Python implementation of both the
"spherical" (haversine) and "triangular" (flat-earth) distance kernels, then
asserts that the Rust-backed _conley_meat() produces identical results.

The Python reference is intentionally simple and unoptimized to maximize
readability and minimize the chance of sharing bugs with the Rust code.
"""

import numpy as np
import pytest

from pyfixest.estimation.internals.vcov_utils import _conley_meat

# ---------------------------------------------------------------------------
# Pure-Python reference implementation (no Rust, no Numba)
# ---------------------------------------------------------------------------


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine great-circle distance in km between two points in degrees.

    Uses earth_diameter_km = 12752 to match the Rust backend constant.
    """
    EARTH_DIAMETER_KM = 12752.0  # Must match conley.rs
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    # Clamp for numerical safety
    a = min(a, 1.0)
    return EARTH_DIAMETER_KM * np.arcsin(np.sqrt(a))


def _triangular_within_cutoff(
    lat1: float, lon1: float, lat2: float, lon2: float, cutoff_km: float
) -> bool:
    """
    Flat-earth (triangular) distance check.

    Converts lat/lon differences to radians, scales longitude by cos(mean_lat),
    then checks if the squared planar distance is within (cutoff_km / 111)^2
    in radian units.
    """
    dlat_rad = np.radians(lat2 - lat1)
    # Longitude difference with wraparound
    dlon = lon2 - lon1
    dlon_rad = np.radians(dlon)
    # Handle wraparound: take the shorter arc
    if abs(dlon_rad) > np.pi:
        dlon_rad = dlon_rad - np.sign(dlon_rad) * 2 * np.pi

    cos_lat_mean = np.cos(np.radians((lat1 + lat2) / 2))
    scaled_lon = cos_lat_mean * dlon_rad
    dist_sq = dlat_rad**2 + scaled_lon**2
    cutoff_rad = np.radians(cutoff_km / 111.0)
    return dist_sq <= cutoff_rad**2


def _conley_meat_python_reference(
    scores: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    cutoff: float,
    distance: str = "spherical",
) -> np.ndarray:
    """
    Pure-Python brute-force Conley meat matrix (uniform kernel).

    This uses a uniform kernel (0/1 based on cutoff) which matches the Rust
    implementation. The Rust code does NOT apply a Bartlett taper — it simply
    includes or excludes pairs based on distance.

    Parameters
    ----------
    scores : np.ndarray, shape (n, k)
    lat, lon : np.ndarray, shape (n,)
        Coordinates in degrees (normalized to [-90,90] lat, [-180,180] lon).
    cutoff : float
        Distance cutoff in km.
    distance : str
        "spherical" (haversine) or "triangular" (flat-earth).

    Returns
    -------
    meat : np.ndarray, shape (k, k)
    """
    n, k = scores.shape
    meat = np.zeros((k, k))

    for i in range(n):
        for j in range(n):
            if distance == "spherical":
                d = _haversine_km(lat[i], lon[i], lat[j], lon[j])
                within = d <= cutoff
            else:
                within = _triangular_within_cutoff(
                    lat[i], lon[i], lat[j], lon[j], cutoff
                )

            if within:
                meat += np.outer(scores[i], scores[j])

    return meat


# ---------------------------------------------------------------------------
# Test class: validate Rust backend against Python reference
# ---------------------------------------------------------------------------


@pytest.mark.hac
class TestConleyRustVsPythonReference:
    """Validate the Rust Conley meat against an independent Python reference."""

    def test_spherical_small_dataset(self):
        """Spherical distance: Rust matches Python on a small random dataset."""
        rng = np.random.default_rng(42)
        n, k = 30, 3
        scores = rng.standard_normal((n, k))
        lat = rng.uniform(-30, 30, n)
        lon = rng.uniform(-60, 60, n)
        cutoff = 1000.0

        rust_meat = _conley_meat(
            scores=scores,
            lon_arr=lon,
            lat_arr=lat,
            cutoff=cutoff,
            distance="spherical",
            aggregate=False,
        )
        python_meat = _conley_meat_python_reference(
            scores, lat, lon, cutoff, distance="spherical"
        )

        np.testing.assert_allclose(rust_meat, python_meat, rtol=1e-6, atol=1e-10)

    def test_triangular_small_dataset(self):
        """Triangular distance: Rust matches Python on a small random dataset."""
        rng = np.random.default_rng(123)
        n, k = 25, 2
        scores = rng.standard_normal((n, k))
        lat = rng.uniform(-20, 20, n)
        lon = rng.uniform(-40, 40, n)
        cutoff = 500.0

        rust_meat = _conley_meat(
            scores=scores,
            lon_arr=lon,
            lat_arr=lat,
            cutoff=cutoff,
            distance="triangular",
            aggregate=False,
        )
        python_meat = _conley_meat_python_reference(
            scores, lat, lon, cutoff, distance="triangular"
        )

        np.testing.assert_allclose(rust_meat, python_meat, rtol=1e-6, atol=1e-10)

    def test_spherical_high_latitude(self):
        """Spherical at high latitudes where longitude convergence matters."""
        rng = np.random.default_rng(456)
        n, k = 20, 2
        scores = rng.standard_normal((n, k))
        lat = rng.uniform(60, 80, n)  # High latitude
        lon = rng.uniform(-180, 180, n)
        cutoff = 1500.0

        rust_meat = _conley_meat(
            scores=scores,
            lon_arr=lon,
            lat_arr=lat,
            cutoff=cutoff,
            distance="spherical",
            aggregate=False,
        )
        python_meat = _conley_meat_python_reference(
            scores, lat, lon, cutoff, distance="spherical"
        )

        np.testing.assert_allclose(rust_meat, python_meat, rtol=1e-6, atol=1e-10)

    def test_triangular_high_latitude(self):
        """Triangular at high latitudes."""
        rng = np.random.default_rng(789)
        n, k = 20, 2
        scores = rng.standard_normal((n, k))
        lat = rng.uniform(60, 80, n)
        lon = rng.uniform(-50, 50, n)
        cutoff = 800.0

        rust_meat = _conley_meat(
            scores=scores,
            lon_arr=lon,
            lat_arr=lat,
            cutoff=cutoff,
            distance="triangular",
            aggregate=False,
        )
        python_meat = _conley_meat_python_reference(
            scores, lat, lon, cutoff, distance="triangular"
        )

        np.testing.assert_allclose(rust_meat, python_meat, rtol=1e-6, atol=1e-10)

    def test_zero_cutoff_diagonal_only(self):
        """Zero cutoff: only self-pairs (i==i) should be included."""
        rng = np.random.default_rng(101)
        n, k = 15, 2
        scores = rng.standard_normal((n, k))
        # Distinct coordinates so no two points are at the same location
        lat = np.linspace(-10, 10, n)
        lon = np.linspace(-20, 20, n)
        cutoff = 0.0

        rust_meat = _conley_meat(
            scores=scores,
            lon_arr=lon,
            lat_arr=lat,
            cutoff=cutoff,
            distance="spherical",
            aggregate=False,
        )
        # With zero cutoff and distinct points, only diagonal survives
        expected = scores.T @ scores
        np.testing.assert_allclose(rust_meat, expected, rtol=1e-10, atol=1e-10)

    def test_huge_cutoff_all_pairs(self):
        """Huge cutoff: all pairs included, meat = (sum s_i)(sum s_i)^T."""
        rng = np.random.default_rng(202)
        n, k = 10, 3
        scores = rng.standard_normal((n, k))
        lat = rng.uniform(-30, 30, n)
        lon = rng.uniform(-60, 60, n)
        cutoff = 25000.0  # Larger than Earth's circumference

        rust_meat = _conley_meat(
            scores=scores,
            lon_arr=lon,
            lat_arr=lat,
            cutoff=cutoff,
            distance="spherical",
            aggregate=False,
        )
        score_sum = scores.sum(axis=0)
        expected = np.outer(score_sum, score_sum)
        np.testing.assert_allclose(rust_meat, expected, rtol=1e-6, atol=1e-10)

    def test_single_covariate(self):
        """Works correctly with k=1 (scalar scores)."""
        rng = np.random.default_rng(303)
        n = 20
        scores = rng.standard_normal((n, 1))
        lat = rng.uniform(-15, 15, n)
        lon = rng.uniform(-30, 30, n)
        cutoff = 600.0

        rust_meat = _conley_meat(
            scores=scores,
            lon_arr=lon,
            lat_arr=lat,
            cutoff=cutoff,
            distance="spherical",
            aggregate=False,
        )
        python_meat = _conley_meat_python_reference(
            scores, lat, lon, cutoff, distance="spherical"
        )

        np.testing.assert_allclose(rust_meat, python_meat, rtol=1e-6, atol=1e-10)

    def test_wraparound_longitude(self):
        """Points near the antimeridian (lon ~180/-180) are correctly paired."""
        scores = np.array([[1.0, 0.5], [2.0, 1.0], [0.5, 3.0]])
        lat = np.array([0.0, 0.0, 0.0])
        lon = np.array([179.0, -179.0, 0.0])
        # Points 0 and 1 are ~222 km apart at the equator
        cutoff = 300.0

        rust_meat = _conley_meat(
            scores=scores,
            lon_arr=lon,
            lat_arr=lat,
            cutoff=cutoff,
            distance="spherical",
            aggregate=False,
        )
        python_meat = _conley_meat_python_reference(
            scores, lat, lon, cutoff, distance="spherical"
        )

        np.testing.assert_allclose(rust_meat, python_meat, rtol=1e-6, atol=1e-10)

    @pytest.mark.parametrize("distance", ["spherical", "triangular"])
    def test_symmetry_matches_reference(self, distance):
        """Both Rust and Python produce symmetric matrices."""
        rng = np.random.default_rng(404)
        n, k = 25, 4
        scores = rng.standard_normal((n, k))
        lat = rng.uniform(-40, 40, n)
        lon = rng.uniform(-80, 80, n)
        cutoff = 700.0

        rust_meat = _conley_meat(
            scores=scores,
            lon_arr=lon,
            lat_arr=lat,
            cutoff=cutoff,
            distance=distance,
            aggregate=False,
        )
        python_meat = _conley_meat_python_reference(
            scores, lat, lon, cutoff, distance=distance
        )

        # Both should be symmetric
        np.testing.assert_allclose(rust_meat, rust_meat.T, atol=1e-12)
        np.testing.assert_allclose(python_meat, python_meat.T, atol=1e-12)
        # And match each other
        np.testing.assert_allclose(rust_meat, python_meat, rtol=1e-6, atol=1e-10)

    @pytest.mark.parametrize("distance", ["spherical", "triangular"])
    def test_many_covariates(self, distance):
        """Works with many covariates (k=10)."""
        rng = np.random.default_rng(505)
        n, k = 20, 10
        scores = rng.standard_normal((n, k))
        lat = rng.uniform(-25, 25, n)
        lon = rng.uniform(-50, 50, n)
        cutoff = 800.0

        rust_meat = _conley_meat(
            scores=scores,
            lon_arr=lon,
            lat_arr=lat,
            cutoff=cutoff,
            distance=distance,
            aggregate=False,
        )
        python_meat = _conley_meat_python_reference(
            scores, lat, lon, cutoff, distance=distance
        )

        np.testing.assert_allclose(rust_meat, python_meat, rtol=1e-5, atol=1e-9)
