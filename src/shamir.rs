use curve25519_dalek::ristretto::RistrettoPoint;
use curve25519_dalek::scalar::Scalar;

use rand_os::OsRng;

use core::ops::{Add, Mul, Sub};


//-----------------------------------------------------------------------------------------------------------
// Share
//-----------------------------------------------------------------------------------------------------------
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


//-----------------------------------------------------------------------------------------------------------
// RistrettoShare
//-----------------------------------------------------------------------------------------------------------
#[allow(non_snake_case)]
#[derive(Copy, Clone)]
pub struct RistrettoShare {
    pub i: u32,
    pub Yi: RistrettoPoint
}

define_add_variants!(LHS = RistrettoShare, RHS = RistrettoPoint, Output = RistrettoShare);
define_add_variants!(LHS = RistrettoPoint, RHS = RistrettoShare, Output = RistrettoShare);
define_comut_add!(LHS = RistrettoPoint, RHS = RistrettoShare, Output = RistrettoShare);
impl<'a, 'b> Add<&'b RistrettoPoint> for &'a RistrettoShare {
    type Output = RistrettoShare;
    fn add(self, rhs: &'b RistrettoPoint) -> RistrettoShare {
        RistrettoShare { i: self.i, Yi: self.Yi + rhs }
    }
}

define_sub_variants!(LHS = RistrettoShare, RHS = RistrettoPoint, Output = RistrettoShare);
define_sub_variants!(LHS = RistrettoPoint, RHS = RistrettoShare, Output = RistrettoShare);
define_comut_sub!(LHS = RistrettoPoint, RHS = RistrettoShare, Output = RistrettoShare);
impl<'a, 'b> Sub<&'b RistrettoPoint> for &'a RistrettoShare {
    type Output = RistrettoShare;
    fn sub(self, rhs: &'b RistrettoPoint) -> RistrettoShare {
        RistrettoShare { i: self.i, Yi: self.Yi - rhs }
    }
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


//-----------------------------------------------------------------------------------------------------------
// Shared traits and functions for Polynomial and RistrettoPolynomial
//-----------------------------------------------------------------------------------------------------------
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

pub trait Reconstruct<S> {
    type Output;
    fn reconstruct(shares: &[S], k: usize) -> Self::Output;
}

pub trait Evaluate {
    type Output;
    fn evaluate(&self, x: Scalar) -> Self::Output;
}


//-----------------------------------------------------------------------------------------------------------
// RistrettoPolynomial
//-----------------------------------------------------------------------------------------------------------
#[allow(non_snake_case)]
pub struct RistrettoPolynomial {
    pub k: usize,
    pub A: Vec<RistrettoPoint>
}

define_mul_variants!(LHS = Polynomial, RHS = RistrettoPoint, Output = RistrettoPolynomial);
define_mul_variants!(LHS = RistrettoPoint, RHS = Polynomial, Output = RistrettoPolynomial);
define_comut_mul!(LHS = RistrettoPoint, RHS = Polynomial, Output = RistrettoPolynomial);
impl<'a, 'b> Mul<&'b RistrettoPoint> for &'a Polynomial {
    type Output = RistrettoPolynomial;
    fn mul(self, rhs: &'b RistrettoPoint) -> RistrettoPolynomial {
        RistrettoPolynomial {
            k: self.k,
            A: self.a.iter().map(|ak| ak * rhs).collect::<Vec<RistrettoPoint>>()
        }
    }
}

impl RistrettoPolynomial {
    pub fn verify(&self, share: RistrettoShare) -> bool {
        let x = Scalar::from(share.i as u64);
        share.Yi == self.evaluate(x)
    }
}

impl Evaluate for RistrettoPolynomial {
    type Output = RistrettoPoint;
    
    fn evaluate(&self, x: Scalar) -> RistrettoPoint {
        // evaluate using Horner's rule
        let mut rev = self.A.iter().rev();
        let head = *rev.next().unwrap();
            
        rev.fold(head, |partial, coef| partial * x + coef)
    }
}


//-----------------------------------------------------------------------------------------------------------
// Polynomial
//-----------------------------------------------------------------------------------------------------------
pub struct Polynomial {
    pub k: usize,
    pub a: Vec<Scalar>
}

impl Polynomial {
    pub fn rnd(secret: Scalar, degree: usize) -> Self {
        let mut coefs = vec![secret];

        let mut csprng: OsRng = OsRng::new().unwrap();
        let rnd_coefs: Vec<Scalar> = (0..degree).map(|_| Scalar::random(&mut csprng)).collect();
        coefs.extend(rnd_coefs);
        
        Polynomial { k: degree, a: coefs }
    }

    pub fn shares(&self, n: usize) -> Vec<Share> {
        let mut shares = Vec::<Share>::new();
        for j in 1..n + 1 {
            let x = Scalar::from(j as u64);
            let share = Share { i: j as u32, yi: self.evaluate(x) };
            shares.push(share);
        }

        shares
    }
}

impl Evaluate for Polynomial {
    type Output = Scalar;
    
    fn evaluate(&self, x: Scalar) -> Scalar {
        // evaluate using Horner's rule
        let mut rev = self.a.iter().rev();
        let head = *rev.next().unwrap();
            
        rev.fold(head, |partial, coef| partial * x + coef)
    }
}

impl Reconstruct<Share> for Polynomial {
    type Output = Scalar;
    
    fn reconstruct(shares: &[Share], k: usize) -> Scalar {
        assert!(shares.len() >= k + 1);
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

impl Reconstruct<RistrettoShare> for Polynomial {
    type Output = RistrettoPoint;

    #[allow(non_snake_case)]
    fn reconstruct(shares: &[RistrettoShare], k: usize) -> RistrettoPoint {
        assert!(shares.len() >= k + 1);
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
