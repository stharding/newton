from PIL import Image, ImageDraw
import random


cdef class Polynomial:
    cdef list coefficients

    def __init__(self, *coefficients):
        """input: coefficients are in the form a_n, ...a_1, a_0"""
        self.coefficients = list(coefficients)  # tuple is turned into a list

    def __repr__(self):
        """
        method to return the canonical string representation
        of a polynomial.

        """
        return f"Polynomial({repr(self.coefficients).strip('[]')})"

    def __call__(self, complex x):
        return self._call(x)

    cdef complex _call(self, complex x):
        cdef complex res, coeff

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


cdef float affine(
    float floor_in,
    float val,
    float ceil_in,
    float floor_out,
    float ceil_out
):
    """
    :param floor_in: the floor of the input range
    :param val: the position in the input range to affine
    :param ceil_in: the ceiling of the input range
    :param floor_out: the floor of the output range
    :param ceil_out: the ceiling of the output range
    :return: val affined onto the output range
    """

    return ((val - floor_in) / (ceil_in - floor_in)) * (
        ceil_out - floor_out
    ) + floor_out


cpdef newton_fract(
    Polynomial poly = None,
    str img_name="out.png",
    float tolerance=0.0001,
    int imax=30,
    dims=(500, 500),
    window=(-1, 1, 1, -1),
):
    cdef int i, j, count, scaled_count
    cdef float _i, _j, diff
    cdef complex val, initial, root, old, ratio, c_val
    cdef set roots
    cdef Polynomial poly_prime

    img = Image.new(mode="RGB", size=dims)
    img_draw = ImageDraw.ImageDraw(img)
    colors = {}
    if poly is None:
        poly = Polynomial(1, 0, 0, -1)
    poly_prime = poly.derivative()
    roots = set()
    for i in range(dims[0]):
        _i = affine(0, i, dims[0], window[0], window[2])
        for j in range(dims[1]):
            _j = affine(0, j, dims[1], window[1], window[3])
            initial = complex(_i, _j)

            val = initial
            for count in range(imax):
                old = val
                try:
                    ratio = poly._call(val) / poly_prime._call(val)
                except ZeroDivisionError:
                    img_draw.point((i, j), (0, 0, 0))
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
                        colors[val] = (
                            random.randint(0, 100),
                            random.randint(0, 100),
                            random.randint(0, 100),
                        )

                    scaled_count = int(affine(0, count, imax, 0, 155))
                    c_val = 0
                    for root in roots:
                        if abs(val - root) < tolerance * 2:
                            c_val = root

                    img_draw.point(
                        (i, j),
                        (
                            colors[c_val][0] + scaled_count,
                            colors[c_val][1] + scaled_count,
                            colors[c_val][2] + scaled_count,
                        ),
                    )
                    break
            else:
                img_draw.point((i, j), (150, 150, 150))

    p_roots = {f"{root:5g}" for root in roots}
    print(f"{len(roots)} roots in {poly}: {p_roots}")
    img.save(img_name)
