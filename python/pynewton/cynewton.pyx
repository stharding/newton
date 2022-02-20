import random


class Polynomial:
    def __init__(self, *coefficients):
        """input: coefficients are in the form a_n, ...a_1, a_0"""
        self.coefficients = list(coefficients)  # tuple is turned into a list

    def __repr__(self):
        """
        method to return the canonical string representation
        of a polynomial.

        """
        return f"Polynomial({', '.join(map(str, self.coefficients))})"

    def __call__(self, x):
        res = 0
        for coeff in self.coefficients:
            res = res * x + coeff
        return res

    def degree(self):
        return len(self.coefficients)

    def derivative(self) -> "Polynomial":
        derived_coeffs = []
        exponent = len(self.coefficients) - 1
        for i in range(len(self.coefficients) - 1):
            derived_coeffs.append(self.coefficients[i] * exponent)
            exponent -= 1
        return Polynomial(*derived_coeffs)

    def __str__(self):
        def x_expr(degree):
            if degree == 0:
                res = ""
            elif degree == 1:
                res = "x"
            else:
                res = "x^" + str(degree)
            return res

        degree = len(self.coefficients) - 1
        res = ""

        for i in range(0, degree + 1):
            coeff = self.coefficients[i]
            # nothing has to be done if coeff is 0:
            if abs(coeff) == 1 and i < degree:
                # 1 in front of x shouldn't occur, e.g. x instead of 1x
                # but we need the plus or minus sign:
                res += f"{'+' if coeff>0 else '-'}{x_expr(degree-i)}"
            elif coeff != 0:
                res += f"{coeff:+g}{x_expr(degree-i)}"

        return res.lstrip("+")  # removing leading '+'0


def affine(floor_in, val, ceil_in, floor_out, ceil_out):
    """
    :param floor_in: the floor of the input range
    :param val: the position in the input range to affine
    :param ceil_in: the ceiling of the input range
    :param floor_out: the floor of the output range
    :param ceil_out: the ceiling of the output range
    :return: val affined onto the output range
    """

    floor_in = float(floor_in)
    val = float(val)
    ceil_in = float(ceil_in)
    floor_out = float(floor_out)
    ceil_out = float(ceil_out)

    return ((val - floor_in) / (ceil_in - floor_in)) * (
        ceil_out - floor_out
    ) + floor_out


def newton_fract(
    poly: Polynomial = None,
    tolerance=0.0001,
    imax=30,
    dims=(500, 500),
    window=(-1, 1, -1, 1),
):
    screen = pygame.display.set_mode(dims)
    colors = {}
    if poly is None:
        poly = Polynomial(1, 0, 0, -1)
    poly_prime = poly.derivative()
    roots = set()
    for i in range(dims[0]):
        _i = affine(0, i, dims[0], window[0], window[1])
        for j in range(dims[1]):
            _j = affine(0, j, dims[1], window[2], window[3])
            initial = complex(_i, _j)

            val = initial
            for count in range(imax):
                old = val
                try:
                    ratio = poly(val) / poly_prime(val)
                except ZeroDivisionError:
                    screen.set_at((i, j), pygame.Color(0, 0, 0))
                    break
                val = val - ratio
                diff = abs(old - val)
                if diff < tolerance:
                    represented = False
                    for root in roots:
                        if abs(val - root) < tolerance * 2:
                            represented = True
                    if not represented:
                        roots.add(val)
                        colors[val] = pygame.Color(
                            random.randint(0, 150),
                            random.randint(0, 150),
                            random.randint(0, 150),
                        )

                    scaled_count = int(affine(0, count, imax, 0, 105))
                    c_val = None
                    for root in roots:
                        if abs(val - root) < tolerance * 2:
                            c_val = root

                    color = pygame.Color(
                        colors[c_val].r + scaled_count,
                        colors[c_val].g + scaled_count,
                        colors[c_val].b + scaled_count,
                    )

                    screen.set_at((i, j), color)
                    break
            else:
                screen.set_at((i, j), pygame.Color(150, 150, 150))

    p_roots = {f"{root:5g}" for root in roots}
    print(f"{len(roots)} roots in {poly}: {p_roots}")
    pygame.display.flip()
