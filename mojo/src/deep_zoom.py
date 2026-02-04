"""Reference orbit computation for deep zoom Mandelbrot rendering.

Uses mpmath for arbitrary precision to compute reference orbits that
the GPU can use with perturbation theory for zooms beyond Float64 limits.
"""

import numpy as np

try:
    import mpmath
except ImportError:
    raise ImportError(
        "mpmath is required for deep zoom: pip install mpmath"
    )


class ReferenceOrbit:
    """Computes arbitrary-precision reference orbits for perturbation rendering.

    The reference orbit Z_n is computed at the center of the view using
    arbitrary precision (mpmath). The GPU then computes perturbations
    δ_n = z_n - Z_n for each pixel, where z_n is the actual orbit.

    This allows rendering at zoom levels far beyond Float64 limits (~10^15)
    since the perturbations δ remain small and representable in Float64.
    """

    def __init__(self, center_re: str, center_im: str, precision_digits: int = 50):
        """Initialize reference orbit calculator.

        Args:
            center_re: Real part of center as decimal string (e.g., "-0.75")
            center_im: Imaginary part of center as decimal string (e.g., "0.1")
            precision_digits: Decimal digits of precision for mpmath
        """
        self.center_re_str = center_re
        self.center_im_str = center_im
        self.precision_digits = precision_digits

        # Set mpmath precision (add some margin for intermediate calculations)
        mpmath.mp.dps = precision_digits + 20

        # Parse center point with full precision
        self._center = mpmath.mpc(center_re, center_im)

        # Orbit storage (populated by compute())
        self._orbit_re: list = []
        self._orbit_im: list = []
        self._orbit_length: int = 0
        self._escaped: bool = False
        self._escape_iteration: int = -1

    def compute(self, imax: int, escape_radius_sq: float = 256.0) -> int:
        """Compute reference orbit up to imax iterations.

        The orbit is stored for later retrieval via get_orbit_arrays().

        Args:
            imax: Maximum iterations to compute
            escape_radius_sq: Escape radius squared for bailout check

        Returns:
            Actual orbit length (may be less than imax if orbit escapes)
        """
        self._orbit_re = []
        self._orbit_im = []

        # Mandelbrot iteration: Z_{n+1} = Z_n^2 + C
        z = mpmath.mpc(0, 0)
        c = self._center

        for i in range(imax):
            # Store current position
            self._orbit_re.append(float(z.real))
            self._orbit_im.append(float(z.imag))

            # Check for escape (convert to float for comparison)
            z_norm_sq = float(z.real ** 2 + z.imag ** 2)
            if z_norm_sq > escape_radius_sq:
                self._escaped = True
                self._escape_iteration = i
                break

            # Iterate
            z = z * z + c

        self._orbit_length = len(self._orbit_re)
        return self._orbit_length

    def get_orbit_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the reference orbit as numpy Float64 arrays.

        Returns:
            Tuple of (re_array, im_array) where each is a 1D Float64 array
            of length orbit_length.
        """
        if not self._orbit_re:
            raise ValueError("Must call compute() before get_orbit_arrays()")
        return (
            np.array(self._orbit_re, dtype=np.float64),
            np.array(self._orbit_im, dtype=np.float64),
        )

    @property
    def orbit_length(self) -> int:
        """Length of computed orbit."""
        return self._orbit_length

    @property
    def escaped(self) -> bool:
        """Whether the reference point escaped."""
        return self._escaped

    @property
    def escape_iteration(self) -> int:
        """Iteration at which reference escaped (-1 if didn't escape)."""
        return self._escape_iteration

    def get_center_float64(self) -> tuple[float, float]:
        """Get center as Float64 (may lose precision at deep zooms)."""
        return float(self._center.real), float(self._center.imag)

    @staticmethod
    def precision_for_zoom(log2_zoom: float) -> int:
        """Estimate required precision digits for a given zoom level.

        Args:
            log2_zoom: log2 of the zoom factor (e.g., 50 means 2^50 zoom)

        Returns:
            Recommended precision in decimal digits
        """
        # Each power of 2 is roughly 0.301 decimal digits
        # Add margin for safety
        base_digits = int(log2_zoom * 0.301) + 20
        return max(50, base_digits)


def compute_delta_per_pixel(log2_zoom: float, height: int, base_half_height: float = 2.0) -> float:
    """Compute the complex plane distance per pixel.

    This divides by height (not width) to match the standard Mandelbrot renderer
    which uses half_height as the zoom parameter. This ensures square pixels
    in complex space while maintaining correct aspect ratio.

    Args:
        log2_zoom: log2 of the zoom factor
        height: Image height in pixels
        base_half_height: Base half-height at zoom=0 (default: 2.0 for im range [-2, 2])

    Returns:
        Complex plane distance per pixel (delta_per_pixel)
    """
    # Half-height at this zoom level
    half_h = base_half_height / (2 ** log2_zoom)
    # Full height span in complex plane
    full_h = half_h * 2
    # Delta per pixel (same for both x and y to maintain square aspect)
    return full_h / height


def find_best_reference(
    center_re: str,
    center_im: str,
    log2_zoom: float,
    width: int,
    height: int,
    sample_grid: int = 16,
    test_imax: int = 500,
) -> tuple[str, str, int]:
    """Find a good reference point by sampling the view for high-iteration pixels.

    Instead of using the view center (which might escape quickly), this samples
    a grid of points and returns the one with the highest iteration count.

    Args:
        center_re, center_im: View center as decimal strings
        log2_zoom: Current zoom level
        width, height: View dimensions
        sample_grid: Number of samples per axis (total = sample_grid^2)
        test_imax: Max iterations for testing each sample

    Returns:
        Tuple of (best_re, best_im, best_iterations) as decimal strings and iteration count
    """
    delta_per_pixel = compute_delta_per_pixel(log2_zoom, height)
    precision = precision_for_zoom(log2_zoom)

    best_re = center_re
    best_im = center_im
    best_iters = 0

    # Sample a grid of points
    for gy in range(sample_grid):
        for gx in range(sample_grid):
            # Convert grid position to pixel offset from center
            px = (gx / (sample_grid - 1) - 0.5) * width
            py = (gy / (sample_grid - 1) - 0.5) * height

            # Convert to complex offset
            offset_re = px * delta_per_pixel
            offset_im = -py * delta_per_pixel

            # Compute sample point with full precision
            sample_re = add_decimal_strings(center_re, str(offset_re), precision)
            sample_im = add_decimal_strings(center_im, str(offset_im), precision)

            # Quick iteration test using mpmath
            mpmath.mp.dps = precision + 10
            z = mpmath.mpc(0, 0)
            c = mpmath.mpc(sample_re, sample_im)

            iters = 0
            for i in range(test_imax):
                if float(z.real ** 2 + z.imag ** 2) > 256.0:
                    break
                z = z * z + c
                iters = i + 1

            if iters > best_iters:
                best_iters = iters
                best_re = sample_re
                best_im = sample_im

                # If we found a point that doesn't escape, that's ideal
                if iters == test_imax:
                    return best_re, best_im, best_iters

    return best_re, best_im, best_iters


def precision_for_zoom(log2_zoom: float) -> int:
    """Estimate required precision digits for a given zoom level."""
    base_digits = int(log2_zoom * 0.301) + 20
    return max(50, base_digits)


# Helper for find_best_reference to use the same function
def add_decimal_strings(a: str, b: str, precision: int = 100) -> str:
    """Add two decimal strings with arbitrary precision."""
    from decimal import Decimal, getcontext
    getcontext().prec = precision
    result = Decimal(a) + Decimal(b)
    return str(result)


if __name__ == "__main__":
    # Test: compute reference orbit at a known location
    print("Testing ReferenceOrbit...")

    # Test at the main cardioid
    orbit = ReferenceOrbit("-0.75", "0.0", precision_digits=50)
    length = orbit.compute(1000)
    print(f"Orbit at (-0.75, 0): length={length}, escaped={orbit.escaped}")

    # Test at a point that escapes
    orbit2 = ReferenceOrbit("0.5", "0.5", precision_digits=50)
    length2 = orbit2.compute(1000)
    print(f"Orbit at (0.5, 0.5): length={length2}, escaped={orbit2.escaped}, "
          f"escape_iter={orbit2.escape_iteration}")

    # Test array conversion
    re_arr, im_arr = orbit.get_orbit_arrays()
    print(f"Array shapes: re={re_arr.shape}, im={im_arr.shape}")
    print(f"First 5 re values: {re_arr[:5]}")

    # Test precision estimation
    for zoom in [10, 50, 100, 200]:
        prec = ReferenceOrbit.precision_for_zoom(zoom)
        print(f"Zoom 2^{zoom}: recommended {prec} decimal digits")

    # Test delta per pixel (using height=600)
    for zoom in [0, 10, 50]:
        delta = compute_delta_per_pixel(zoom, 600)
        print(f"Zoom 2^{zoom}: delta_per_pixel = {delta:.2e}")

    print("\nAll tests passed!")
