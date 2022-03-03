// extern crate clap;
extern crate raster;

use num::Complex;
use rand::prelude::*;
use raster::Image;
use std::collections::HashMap;

fn affine(floor_in: f64, val: f64, ceil_in: f64, floor_out: f64, ceil_out: f64) -> f64 {
    ((val - floor_in) / (ceil_in - floor_in)) * (ceil_out - floor_out) + floor_out
}

struct Polynomial {
    coefficients: Vec<f64>,
}

impl Polynomial {
    fn value_at(&self, x: Complex<f64>) -> Complex<f64> {
        if x.re == 0.0 && x.im == 0.0 {
            return Complex::new(0.0, 0.0);
        }
        let mut result = Complex::new(0.0, 0.0);
        for coeff in &self.coefficients {
            result = result * x + coeff;
        }
        result
    }

    fn derivative(&self) -> Polynomial {
        let mut derived_coeffs: Vec<f64> = Vec::new();
        let coeff_len = self.coefficients.len() - 1;
        let mut exponent = coeff_len;
        for i in 0..coeff_len {
            derived_coeffs.push(self.coefficients[i] * exponent as f64);
            exponent -= 1;
        }
        Polynomial {
            coefficients: derived_coeffs,
        }
    }
}

// fn parse_args() {
//     // let args = clap::
// }

fn main() {
    let width = 2000;
    let height = 2000;
    let imax = 30;
    let window = vec![-1.0, 1.0, 1.0, -1.0];

    let mut colors = HashMap::new();
    let tolerance = 0.0001;
    let mut rng = rand::thread_rng();
    let mut roots: Vec<Complex<f64>> = Vec::new();
    let poly = Polynomial {
        coefficients: vec![1.0, 0.0, 0.0, -1.0],
    };
    let poly_prime = poly.derivative();

    let mut img = Image::blank(width, height);

    for x in 0..width {
        let _x = affine(0.0, x as f64, width as f64, window[0], window[2]);
        for i in 0..height {
            let _i = affine(0.0, i as f64, height as f64, window[1], window[3]);
            let mut val = Complex::new(_x, _i);
            let mut represented = false;
            for count in 0..imax {
                let old = val;
                let numerator = poly.value_at(val);
                let denominator = poly_prime.value_at(val);
                if denominator.re == 0.0 && denominator.im == 0.0 {
                    img.set_pixel(x, i, raster::Color::rgb(100, 100, 100))
                        .unwrap();
                } else {
                    let ratio = numerator / denominator;
                    val = val - ratio;
                    let diff = (old - val).norm();
                    if diff < tolerance {
                        for root in &roots {
                            if (val - *root).norm() < tolerance * 2.0 {
                                represented = true;
                            }
                        }
                        if !represented {
                            roots.push(val);
                            colors.insert(
                                format!("{val}"),
                                raster::Color::rgb(
                                    rng.gen_range(0..100),
                                    rng.gen_range(0..100),
                                    rng.gen_range(0..100),
                                ),
                            );
                        }

                        let scaled_count = affine(0.0, count as f64, imax as f64, 0., 155.) as u8;
                        let mut c_val = Complex::new(0.0, 0.0);
                        for root in &roots {
                            if (val - root).norm() < tolerance * 2.0 {
                                c_val = *root;
                            }
                        }
                        let base_color = colors.get(&format!("{c_val}")).unwrap();
                        let color = raster::Color::rgb(
                            base_color.r + scaled_count,
                            base_color.g + scaled_count,
                            base_color.b + scaled_count,
                        );
                        img.set_pixel(x, i, color).unwrap();
                        break;
                    }
                }
            }
            if !represented {
                img.set_pixel(x, i, raster::Color::rgb(100, 100, 100))
                    .unwrap();
            }
        }
    }

    raster::save(&img, "tst.png").unwrap();
    println!("Done, file at {}", "tst.png");
}
