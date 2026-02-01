from python import Python, PythonObject
from complex import ComplexFloat64
from random import random_ui64
from math import sqrt
from moclap import cli_parse


# ============================================================================
# CLI Config
# ============================================================================

@fieldwise_init
struct Config(Copyable, Movable, Defaultable, Writable):
    """CLI configuration for Newton fractal generator."""
    var coefficients: String  # Comma-separated, e.g. "1,0,0,-1" for x³-1
    var out_file: String
    var tolerance: Float64
    var imax: Int
    var width: Int
    var height: Int
    var window_left: Float64
    var window_right: Float64
    var window_top: Float64
    var window_bottom: Float64

    fn __init__(out self):
        self.coefficients = "1,0,0,-1"
        self.out_file = "out.png"
        self.tolerance = 0.0001
        self.imax = 30
        self.width = 500
        self.height = 500
        self.window_left = -1.0
        self.window_right = 1.0
        self.window_top = 1.0
        self.window_bottom = -1.0


fn parse_coefficients(s: String) raises -> List[Float64]:
    """Parse comma-separated coefficients string into list of Float64."""
    var coeffs = List[Float64]()
    var parts = s.split(",")
    for i in range(len(parts)):
        var part = parts[i].strip()
        if len(part) > 0:
            coeffs.append(atof(part))
    return coeffs^


# ============================================================================
# Polynomial
# ============================================================================

@fieldwise_init
struct Polynomial(Stringable, Copyable, Movable):
    """Polynomial with coefficients in form a_n, ..., a_1, a_0."""
    var coefficients: List[Float64]

    fn __init__(out self, *coeffs: Float64):
        self.coefficients = List[Float64]()
        for c in coeffs:
            self.coefficients.append(c)

    fn degree(self) -> Int:
        return len(self.coefficients) - 1

    fn eval(self, x: ComplexFloat64) -> ComplexFloat64:
        """Evaluate polynomial at x using Horner's method."""
        var res = ComplexFloat64(0, 0)
        for i in range(len(self.coefficients)):
            res = res * x + ComplexFloat64(self.coefficients[i], 0)
        return res

    fn derivative(self) -> Polynomial:
        """Return derivative polynomial."""
        var derived = List[Float64]()
        var exp = len(self.coefficients) - 1
        for i in range(len(self.coefficients) - 1):
            derived.append(self.coefficients[i] * exp)
            exp -= 1
        return Polynomial(derived^)

    fn __str__(self) -> String:
        var res = String("")
        var degree = len(self.coefficients) - 1
        for i in range(len(self.coefficients)):
            var coeff = self.coefficients[i]
            var exp = degree - i
            if coeff == 0:
                continue

            # Sign
            if len(res) > 0:
                if coeff > 0:
                    res += " + "
                else:
                    res += " - "
                    coeff = -coeff
            elif coeff < 0:
                res += "-"
                coeff = -coeff

            # Coefficient
            if coeff != 1 or exp == 0:
                if coeff == Int(coeff):
                    res += String(Int(coeff))
                else:
                    res += String(coeff)

            # Variable
            if exp == 1:
                res += "x"
            elif exp > 1:
                res += "x^" + String(exp)

        if len(res) == 0:
            return "0"
        return res


# ============================================================================
# Helper functions
# ============================================================================

fn affine(floor_in: Float64, val: Float64, ceil_in: Float64,
          floor_out: Float64, ceil_out: Float64) -> Float64:
    """Map val from input range to output range."""
    return ((val - floor_in) / (ceil_in - floor_in)) * (ceil_out - floor_out) + floor_out


fn complex_abs(c: ComplexFloat64) -> Float64:
    """Compute magnitude of complex number."""
    return sqrt(c.re * c.re + c.im * c.im)


# ============================================================================
# Root tracking
# ============================================================================

@fieldwise_init
struct Root(Copyable, Movable):
    """A discovered root with its assigned color."""
    var re: Float64
    var im: Float64
    var r: Int
    var g: Int
    var b: Int

    fn __init__(out self, value: ComplexFloat64):
        self.re = value.re
        self.im = value.im
        # Random color in 0-150 range
        self.r = Int(random_ui64(0, 150))
        self.g = Int(random_ui64(0, 150))
        self.b = Int(random_ui64(0, 150))

    fn matches(self, val: ComplexFloat64, tolerance: Float64) -> Bool:
        """Check if val is close to this root."""
        var diff_re = self.re - val.re
        var diff_im = self.im - val.im
        var dist = sqrt(diff_re * diff_re + diff_im * diff_im)
        return dist < tolerance * 2


fn find_matching_root(roots: List[Root], val: ComplexFloat64, tolerance: Float64) -> Int:
    """Find index of root matching val, or -1 if none."""
    for i in range(len(roots)):
        if roots[i].matches(val, tolerance):
            return i
    return -1


# ============================================================================
# Main fractal generation
# ============================================================================

fn newton_fract(
    poly: Polynomial,
    img_name: String,
    tolerance: Float64,
    imax: Int,
    width: Int,
    height: Int,
    window_left: Float64,
    window_right: Float64,
    window_top: Float64,
    window_bottom: Float64,
) raises:
    # Import PIL
    var PIL = Python.import_module("PIL.Image")

    # Compute derivative
    var poly_prime = poly.derivative()

    # Track discovered roots
    var roots = List[Root]()

    # Accumulate pixel data as flat list of (r, g, b) tuples
    var pixels = Python.list()

    for j in range(height):
        var _j = affine(0, Float64(j), Float64(height), window_top, window_bottom)
        for i in range(width):
            var _i = affine(0, Float64(i), Float64(width), window_left, window_right)
            var val = ComplexFloat64(_i, _j)

            var converged = False
            var zero_div = False
            var final_count = imax

            for count in range(imax):
                var old = val
                var denom = poly_prime.eval(val)

                # Check for zero derivative
                if complex_abs(denom) < 1e-10:
                    zero_div = True
                    break

                var ratio = poly.eval(val) / denom
                val = val - ratio

                var diff = complex_abs(old - val)
                if diff < tolerance:
                    converged = True
                    final_count = count
                    break

            # Determine pixel color
            var r: Int = 150
            var g: Int = 150
            var b: Int = 150

            if zero_div:
                r = 128
                g = 128
                b = 128
            elif converged:
                # Find or add root
                var root_idx = find_matching_root(roots, val, tolerance)
                if root_idx < 0:
                    roots.append(Root(val))
                    root_idx = len(roots) - 1

                # Color based on root + iteration count for brightness
                var scaled_count = Int(affine(0, Float64(final_count), Float64(imax), 0, 105))
                r = roots[root_idx].r + scaled_count
                g = roots[root_idx].g + scaled_count
                b = roots[root_idx].b + scaled_count

            # Add pixel to list
            _ = pixels.append(Python.tuple(r, g, b))

    # Print root info
    print(String(len(roots)) + " roots found")

    # Create image and set all pixels at once
    var size = Python.tuple(width, height)
    var img = PIL.new("RGB", size)
    _ = img.putdata(pixels)

    # Save image
    _ = img.save(img_name)
    print("Saved to " + img_name)


fn main() raises:
    var config = cli_parse[Config]()

    # Parse coefficients
    var coeffs = parse_coefficients(config.coefficients)
    var poly = Polynomial(coeffs^)

    print("Newton fractal for: " + String(poly))
    print("dims: " + String(config.width) + "x" + String(config.height))
    print("tolerance: " + String(config.tolerance))
    print("imax: " + String(config.imax))
    print("window: [" + String(config.window_left) + ", " + String(config.window_right) + ", "
          + String(config.window_top) + ", " + String(config.window_bottom) + "]")

    newton_fract(
        poly,
        config.out_file,
        config.tolerance,
        config.imax,
        config.width,
        config.height,
        config.window_left,
        config.window_right,
        config.window_top,
        config.window_bottom,
    )
