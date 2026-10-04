.. _api_physical_optics:

Physical Optics
===============

Propagation conventions
-----------------------

All lengths must use the same unit. ``wavelength`` is the vacuum wavelength;
the homogeneous-medium wavenumber is :math:`k = 2\pi n/\lambda`.
Propagating spectral components acquire phase :math:`\exp(i k_z z)`.
With ``evanescent="decay"``, components above the propagating cutoff instead
acquire attenuation :math:`\exp(-\kappa |z|)`, where
:math:`\kappa = \sqrt{k_x^2 + k_y^2 - k^2}`. Negative distance therefore
reverses propagating phase but **does not invert evanescent attenuation**:
a forward/backward pair attenuates these components twice. This stable decay
policy does not reconstruct growing evanescent waves. ``"discard"`` removes
evanescent components even at zero distance; neither policy is generally
invertible on a field containing evanescent content.

The FFT assumes periodic boundaries, with no automatic padding or sampling
diagnostics. To assess discretization error, enlarge the computational window
at fixed sample spacing, then reduce the spacing at fixed physical window.
Compare the complex field on a common central region; intensity or conserved
discrete power alone cannot detect phase errors or periodic wraparound.

.. toctree::
   :maxdepth: 1

   Gaussian beam propagation example <../examples/gaussian_beam_propagation>

.. automodule:: optiland.physical_optics
   :members:
   :undoc-members:
   :show-inheritance:
