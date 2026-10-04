"""Independent complex-field checks and controlled discretization sweeps."""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.physical_optics import ScalarField, gaussian_field

# All lengths are in mm; wavelength is the vacuum wavelength. No random input.
WAVELENGTH = 0.0006328
WAIST = 0.1
INDEX = 1.4
AMPLITUDE = 0.7 + 0.3j
WAVENUMBER = 2 * np.pi * INDEX / WAVELENGTH
RAYLEIGH_RANGE = WAVENUMBER * WAIST**2 / 2


def _gaussian_comparison(shape, dx, dy, rayleigh_distances):
    """Return carrier-removed numerical field and analytical paraxial reference.

    For E = exp(ikz) u, the paraxial equation is du/dz = i/(2k) laplacian(u).
    Substitution of u = A/q exp(-r^2/(w0^2 q)) gives q = 1 + iz/zR,
    zR = k w0^2/2. Thus 1/q supplies the amplitude and negative Gouy phase,
    while the imaginary part of the exponent supplies positive curvature.
    This reference uses no FFT, propagator helpers or fitted global phase.
    """
    distance = rayleigh_distances * RAYLEIGH_RANGE
    field = gaussian_field(
        shape=shape,
        dx=dx,
        dy=dy,
        wavelength=WAVELENGTH,
        waist_radius=WAIST,
        refractive_index=INDEX,
        amplitude=AMPLITUDE,
    )
    actual = be.to_numpy(field.propagate(distance).data)
    actual = actual * np.exp(-1j * WAVENUMBER * distance)
    ny, nx = shape
    x = (np.arange(nx) - (nx - 1) / 2) * dx
    y = (np.arange(ny) - (ny - 1) / 2) * dy
    radius_squared = y[:, None] ** 2 + x[None, :] ** 2
    q = 1 + 1j * rayleigh_distances
    reference = AMPLITUDE / q * np.exp(-radius_squared / (WAIST**2 * q))
    assert np.all(np.isfinite(actual))
    assert np.all(np.isfinite(reference))
    return actual, reference, radius_squared


@pytest.mark.parametrize("rayleigh_distances", [-2.0, -1.0, 0.5, 2.0])
def test_complex_gaussian_amplitude_curvature_and_gouy(
    set_test_backend, rayleigh_distances
):
    actual, reference, radius_squared = _gaussian_comparison(
        (225, 257), 0.006, 0.007, rayleigh_distances
    )
    # kw0 ~= 1390: omitted nonparaxial terms are O((kw0)^-2).
    # 2e-5 allows those terms and finite-window tails, not an intensity-only
    # comparison or an arbitrary phase alignment. Relative errors exclude tails.
    peak = np.max(np.abs(reference))
    assert np.max(np.abs(actual - reference)) / peak < 2e-5
    core = np.abs(reference) > 0.05 * peak
    assert np.max(np.abs(np.abs(actual[core]) / np.abs(reference[core]) - 1)) < 2e-5
    assert np.max(np.abs(np.angle(actual[core] / reference[core]))) < 2e-5

    center = (actual.shape[0] // 2, actual.shape[1] // 2)
    gouy = -np.arctan(rayleigh_distances)
    assert abs(np.angle(actual[center] / (AMPLITUDE * np.exp(1j * gouy)))) < 2e-5

    # Phase relative to the on-axis field isolates curvature from Gouy phase.
    # Within one propagated beam radius, |curvature phase| <= 2 < pi;
    # no unwrapping or phase-branch ambiguity is involved.
    width_squared = WAIST**2 * (1 + rayleigh_distances**2)
    curvature_core = radius_squared <= width_squared
    expected_curvature = rayleigh_distances * radius_squared / width_squared
    measured_curvature = np.angle(actual * np.conj(actual[center]))
    assert (
        np.max(np.abs((measured_curvature - expected_curvature)[curvature_core])) < 2e-5
    )


@pytest.mark.parametrize("sweep", ["window", "sampling"])
def test_gaussian_window_and_sampling_convergence(set_test_backend, sweep):
    if sweep == "window":
        # Fixed spacing: enlarge the DFT window, retaining common central nodes.
        grids = [(25, 0.01), (49, 0.01), (97, 0.01)]
    else:
        # Fixed DFT period N*dx = 1.2 mm: refine sampling independently of window.
        grids = [(size, 1.2 / size) for size in (9, 15, 25, 45)]

    errors = []
    for size, spacing in grids:
        actual, reference, radius_squared = _gaussian_comparison(
            (size, size), spacing, spacing, 1.0
        )
        # Same physical central disk on every grid; never compare different
        # windows' total power or interpolate one numerical answer onto another.
        core = radius_squared <= WAIST**2
        peak = abs(AMPLITUDE / (1 + 1j))
        errors.append(np.max(np.abs(actual[core] - reference[core])) / peak)

    # Deliberately coarse starting grids expose each discretization error.
    # Require clear improvement, stopping before the paraxial-reference floor.
    assert all(
        fine < coarse / 5 for coarse, fine in zip(errors, errors[1:], strict=False)
    )
    assert errors[-1] < 2e-5


def test_negative_distance_evanescent_decay_is_not_inverse(set_test_backend):
    # Exact Nyquist mode on an even grid: kx = ky = pi/dx > medium cutoff.
    size = 16
    dx = 0.1
    distance = 0.02
    mode = (-1.0) ** (np.arange(size)[:, None] + np.arange(size)[None, :])
    field = ScalarField(be.array(mode), dx=dx, wavelength=1.0)
    kappa = np.sqrt(2 * (np.pi / dx) ** 2 - (2 * np.pi) ** 2)
    attenuation = np.exp(-kappa * distance)
    for signed_distance in (-distance, distance):
        actual = be.to_numpy(field.propagate(signed_distance, evanescent="decay").data)
        np.testing.assert_allclose(actual, mode * attenuation, rtol=1e-12, atol=1e-12)

    round_trip = field.propagate(distance, evanescent="decay").propagate(
        -distance, evanescent="decay"
    )
    np.testing.assert_allclose(
        be.to_numpy(round_trip.data), mode * attenuation**2, rtol=1e-12, atol=1e-12
    )
    assert not np.allclose(be.to_numpy(round_trip.data), mode)
