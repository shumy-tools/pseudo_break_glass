use curve25519_dalek::ristretto::RistrettoPoint;
use curve25519_dalek::scalar::Scalar;

use rand_os::OsRng;

use core::ops::{Add, Mul, Sub};

fn base_coeficient<F>(size: usize, i: usize, xi: u32, get_xj: F) -> Scalar
    where F: Fn(usize) -> u32 {
    
    let sxi = Scalar::from(xi);
    let mut num = Scalar::one();
    let mut denum = Scalar::one();
    for j in 0..size {
        if j != i {
            let xj = Scalar::from(get_xj(j));
            num *= xj;
            denum *= xj - sxi;
        }
    }

    num * denum.invert()
}

#[derive(Copy, Clone)]
pub struct Share {
    pub i: u32,
    pub yi: Scalar
}

define_add_variants!(LHS = Share, RHS = Share, Output = Share);
impl<'a, 'b> Add<&'b Share> for &'a Share {
    type Output = Share;
    fn add(self, rhs: &'b Share) -> Share {
        assert!(self.i == rhs.i);
        Share { i: self.i, yi: self.yi + rhs.yi }
    }
}

define_add_variants!(LHS = Share, RHS = Scalar, Output = Share);
define_add_variants!(LHS = Scalar, RHS = Share, Output = Share);
define_comut_add!(LHS = Scalar, RHS = Share, Output = Share);
impl<'a, 'b> Add<&'b Scalar> for &'a Share {
    type Output = Share;
    fn add(self, rhs: &'b Scalar) -> Share {
        Share { i: self.i, yi: self.yi + rhs }
    }
}

define_sub_variants!(LHS = Share, RHS = Share, Output = Share);
impl<'a, 'b> Sub<&'b Share> for &'a Share {
    type Output = Share;
    fn sub(self, rhs: &'b Share) -> Share {
        assert!(self.i == rhs.i);
        Share { i: self.i, yi: self.yi + rhs.yi }
    }
}

define_sub_variants!(LHS = Share, RHS = Scalar, Output = Share);
define_sub_variants!(LHS = Scalar, RHS = Share, Output = Share);
define_comut_sub!(LHS = Scalar, RHS = Share, Output = Share);
impl<'a, 'b> Sub<&'b Scalar> for &'a Share {
    type Output = Share;
    fn sub(self, rhs: &'b Scalar) -> Share {
        Share { i: self.i, yi: self.yi + rhs }
    }
}

define_mul_variants!(LHS = Share, RHS = Scalar, Output = Share);
define_mul_variants!(LHS = Scalar, RHS = Share, Output = Share);
define_comut_mul!(LHS = Scalar, RHS = Share, Output = Share);
impl<'a, 'b> Mul<&'b Scalar> for &'a Share {
    type Output = Share;
    fn mul(self, rhs: &'b Scalar) -> Share {
        Share { i: self.i, yi: self.yi * rhs }
    }
}

define_mul_variants!(LHS = Share, RHS = RistrettoPoint, Output = RistrettoShare);
define_mul_variants!(LHS = RistrettoPoint, RHS = Share, Output = RistrettoShare);
define_comut_mul!(LHS = RistrettoPoint, RHS = Share, Output = RistrettoShare);
impl<'a, 'b> Mul<&'b RistrettoPoint> for &'a Share {
    type Output = RistrettoShare;
    fn mul(self, rhs: &'b RistrettoPoint) -> RistrettoShare {
        RistrettoShare { i: self.i, Yi: self.yi * rhs }
    }
}

#[allow(non_snake_case)]
#[derive(Copy, Clone)]
pub struct RistrettoShare {
    pub i: u32,
    pub Yi: RistrettoPoint
}

define_mul_variants!(LHS = RistrettoShare, RHS = Scalar, Output = RistrettoShare);
define_mul_variants!(LHS = Scalar, RHS = RistrettoShare, Output = RistrettoShare);
define_comut_mul!(LHS = Scalar, RHS = RistrettoShare, Output = RistrettoShare);
impl<'a, 'b> Mul<&'b Scalar> for &'a RistrettoShare {
    type Output = RistrettoShare;
    fn mul(self, rhs: &'b Scalar) -> RistrettoShare {
        RistrettoShare { i: self.i, Yi: self.Yi * rhs }
    }
}

pub trait Reconstruct<S> {
    type Output;
    fn reconstruct(&self, shares: &[S]) -> Self::Output;
}

pub struct SecretSharing {
    pub n: usize,
    pub t: usize
}

impl SecretSharing {
    pub fn share(&self, secret: Scalar) -> Vec<Share> {
        let poly = self.sample_polynomial(secret);
        self.evaluate_polynomial(&poly).iter().enumerate()
            .map(|(n, y)| Share{ i: (n + 1) as u32, yi: *y }).collect()
    }

    fn sample_polynomial(&self, zero_value: Scalar) -> Vec<Scalar> {
        let mut coefficients = vec![zero_value];

        let mut csprng: OsRng = OsRng::new().unwrap();
        let random_coefficients: Vec<Scalar> = (0..self.t).map(|_| Scalar::random(&mut csprng)).collect();

        coefficients.extend(random_coefficients);
        coefficients
    }

    fn evaluate_polynomial(&self, coefficients: &[Scalar]) -> Vec<Scalar> {
        (1..self.n + 1).map(|x| {
            let point = Scalar::from(x as u64);

            // evaluate using Horner's rule
            let mut rev = coefficients.iter().rev();
            let head = *rev.next().unwrap();
            
            rev.fold(head, |partial, coef| partial * point + coef)
        }).collect()
    }
}

impl Reconstruct<Share> for SecretSharing {
    type Output = Scalar;
    
    fn reconstruct(&self, shares: &[Share]) -> Scalar {
        assert!(shares.len() >= self.t + 1);
        let size = shares.len();

        let mut acc = Scalar::zero();
        for i in 0..size {
            let share = &shares[i];
            let coef = base_coeficient(size, i, share.i, |j| shares[j].i);

            acc += share.yi * coef;
        }

        acc
    }
}

impl Reconstruct<RistrettoShare> for SecretSharing {
    type Output = RistrettoPoint;

    #[allow(non_snake_case)]
    fn reconstruct(&self, shares: &[RistrettoShare]) -> RistrettoPoint {
        assert!(shares.len() >= self.t + 1);
        let size = shares.len();

        let mut res = Vec::<RistrettoPoint>::with_capacity(size);
        for i in 0..size {
            let share = &shares[i];
            let coef = base_coeficient(size, i, share.i, |j| shares[j].i);
            
            let Li = share.Yi * coef;
            res.push(Li);
        }

        let mut coefs = res.iter();
        let head = *coefs.next().unwrap();
        coefs.fold(head, |acc, coef| acc + coef)
    }
}
