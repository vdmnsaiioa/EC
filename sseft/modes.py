"""
External field modes of a cell for the finite-k response (v0.0.27; notes/e4-job3-preregistration.md).

A longitudinal plane wave of the cell, per unit amplitude a,

    u(x) = k^ cos(k.x - p pi/2),      k = 2 pi n b_axis,

with b the reciprocal lattice vectors (a_i . b_j = delta_ij), n = 1, 2, ... its harmonic and p = 0 (cos) or 1 (sin) its
phase.  It is the field -grad phi of the external potential phi = -(1/|k|) sin(k.x - p pi/2): curl-free, as a static
field must be, so its gradient tensors are symmetric -- the read-out reads them so (bandfields: gE[a, b] = dE_a/dx_b,
ggE[a, b, c] = d^2 E_a / dx_b dx_c).  At the points x:

    u_a     =  k^_a cos(k.x - p pi/2)
    g_ab    =  du_a/dx_b           = -k^_a k_b sin(k.x - p pi/2)
    h_abc   =  d^2 u_a / dx_b dx_c = -k^_a k_b k_c cos(k.x - p pi/2)

in the units of the positions and the cell given (u per unit field, g per length, h per length squared).  A mode is
periodic in its cell, so it is the same at every image and in any replica of the cell.

The mode's susceptibility of a cell, chi_mode = -d^2 E / da^2 at a = 0, is what the cell's induced charge does in the
mode's field.  For a statistically homogeneous medium with polarisation answering the external (longitudinal) field
with chi_ext(k), chi_cos + chi_sin = chi_ext V, and the longitudinal dielectric function is
eps_L(k) = 1 / (1 - 4 pi chi_ext(k)).  At k = 0 under tin-foil boundary conditions the uniform field is the macroscopic
field itself and eps_inf = 1 + 4 pi chi_0 instead: the two limits differ by the screening of a longitudinal field, the
same in the model and in MBX (both sum the induced dipoles' fields by Ewald with the k = 0 term left out).
"""
import numpy as np

# the modes read by default: the three axes, the first two harmonics, both phases (12 a frame)
DEFAULT_MODES = tuple((axis, n, phase) for n in (1, 2) for axis in (0, 1, 2) for phase in (0, 1))


def wavevector(cell, axis, n):
    """k = 2 pi n b_axis (in the inverse units of the cell's), the cell's rows its lattice vectors."""
    B = np.linalg.inv(np.asarray(cell, float)).T
    return 2.0 * np.pi * float(n) * B[int(axis)]


def mode_field(points, cell, axis, n, phase):
    """(u (P, 3), g (P, 3, 3), h (P, 3, 3, 3)): the mode's field and its first and second derivatives at the points,
    per unit amplitude (module docstring)."""
    if int(n) == 0:
        raise ValueError("a mode's harmonic n must be >= 1 (n = 0 is the uniform field: model.induced_response)")
    if int(phase) not in (0, 1):
        raise ValueError("a mode's phase is 0 (cos) or 1 (sin)")
    x = np.asarray(points, float).reshape(-1, 3)
    k = wavevector(cell, axis, n)
    kh = k / np.linalg.norm(k)
    arg = x @ k - 0.5 * np.pi * int(phase)
    c, s = np.cos(arg), np.sin(arg)
    u = c[:, None] * kh[None, :]
    g = -s[:, None, None] * np.einsum("a,b->ab", kh, k)[None]
    h = -c[:, None, None, None] * np.einsum("a,b,c->abc", kh, k, k)[None]
    return u, g, h


def describe(cell, modes):
    """the modes as a record: axis, harmonic, phase and |k| (inverse units of the cell's)."""
    return [{"axis": int(a), "n": int(n), "phase": int(p), "k": float(np.linalg.norm(wavevector(cell, a, n)))}
            for a, n, p in modes]
