import argparse


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-c",
        "--cython",
        action="store_true",
        help="use the cython version of the fractal generator",
    )
    parser.add_argument(
        "--coefficients",
        type=int,
        nargs="+",
        default=[1, 0, 0, -1],
        help="The coefficients of the polynomial for the newton fractal",
    )
    parser.add_argument(
        "-o",
        "--out-file",
        default="out.png",
        help="The output file to write to",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0001,
        help=(
            "the delta value to use to determin the iteration "
            "has converged to a root"
        ),
    )
    parser.add_argument(
        "--imax",
        type=int,
        default=30,
        help="the max iterations to perform",
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=[500, 500],
        nargs=2,
        help="The dimensions of the output image, in pixels",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=[-1, 1, 1, -1],
        nargs=4,
        help="The fractal space coordinates to render",
    )

    args = parser.parse_args()

    if args.cython:
        from .cynewton import Polynomial, newton_fract
    else:
        from .pynewton import Polynomial, newton_fract

    poly = Polynomial(*args.coefficients)
    print(f"Newton fractal for: {poly}")
    print(f"img_name: {args.out_file}")
    print(f"tolerance: {args.tolerance}")
    print(f"imax: {args.imax}")
    print(f"dims: {args.dims}")
    print(f"window: {args.window}")

    newton_fract(
        poly=poly,
        img_name=args.out_file,
        tolerance=args.tolerance,
        imax=args.imax,
        dims=args.dims,
        window=args.window,
    )


if __name__ == "__main__":
    main()
