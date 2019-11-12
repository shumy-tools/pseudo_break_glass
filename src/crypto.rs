use curve25519_dalek::ristretto::{RistrettoPoint, CompressedRistretto};
use curve25519_dalek::scalar::Scalar;
use curve25519_dalek::constants::RISTRETTO_BASEPOINT_POINT;

use rand_os::OsRng;
use core::ops::{Add, Mul, Sub};

use sha2::{Sha512, Digest};

const G: RistrettoPoint = RISTRETTO_BASEPOINT_POINT;

//-----------------------------------------------------------------------------------------------------------
// Share
//-----------------------------------------------------------------------------------------------------------
#[derive(Debug, Copy, Clone)]
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
#[derive(Debug, Copy, Clone)]
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
fn cut_tail<Z>(v: &mut Vec::<Z>, elm: Z) where Z: Eq {
    if let Some(i) = v.iter().rev().rposition(|x| *x == elm) {
        v.truncate(i);
    }
}

fn short_mul(a: &mut Vec::<Scalar>, b: Scalar) {
    let mut prev = a[0];
    a[0] *= b;
    for i in 1..a.len() {
        let this = a[i];
        a[i] = prev + a[i] * b;
        prev = this;
    }
    a.push(Scalar::one());
}

fn lx_num_bar(range: &[Scalar], i: usize) -> (Vec<Scalar>, Scalar) {
    let mut num = vec![Scalar::one()];
    let mut denum = Scalar::one();
    for j in 0..range.len() {
        if j != i {
            short_mul(&mut num, -range[j]);
            denum *= range[i] - range[j];
        }
    }

    (num, denum.invert())
}

pub fn rnd_scalar() -> Scalar {
    let mut csprng: OsRng = OsRng::new().unwrap();
    Scalar::random(&mut csprng)
}

pub trait Interpolate<S> {
    type Output;
    fn interpolate(shares: &[S]) -> Self::Output;
}

pub trait Reconstruct<S> {
    type Output;
    fn reconstruct(shares: &[S]) -> Self::Output;
}

pub trait Evaluate {
    type Output;
    fn evaluate(&self, x: Scalar) -> Self::Output;
}

pub trait Degree {
    fn degree(&self) -> usize;
}

//-----------------------------------------------------------------------------------------------------------
// Polynomial
//-----------------------------------------------------------------------------------------------------------
#[derive(Debug, PartialEq, Eq)]
pub struct Polynomial {
    pub a: Vec<Scalar>
}

define_mul_variants!(LHS = Polynomial, RHS = Scalar, Output = Polynomial);
define_mul_variants!(LHS = Scalar, RHS = Polynomial, Output = Polynomial);
define_comut_mul!(LHS = Scalar, RHS = Polynomial, Output = Polynomial);
impl<'a, 'b> Mul<&'b Scalar> for &'a Polynomial {
    type Output = Polynomial;
    fn mul(self, rhs: &'b Scalar) -> Polynomial {
        Polynomial {
            a: self.a.iter().map(|ak| ak * rhs).collect::<Vec<Scalar>>()
        }
    }
}

define_mul_variants!(LHS = Polynomial, RHS = RistrettoPoint, Output = RistrettoPolynomial);
define_mul_variants!(LHS = RistrettoPoint, RHS = Polynomial, Output = RistrettoPolynomial);
define_comut_mul!(LHS = RistrettoPoint, RHS = Polynomial, Output = RistrettoPolynomial);
impl<'a, 'b> Mul<&'b RistrettoPoint> for &'a Polynomial {
    type Output = RistrettoPolynomial;
    fn mul(self, rhs: &'b RistrettoPoint) -> RistrettoPolynomial {
        RistrettoPolynomial {
            A: self.a.iter().map(|ak| ak * rhs).collect::<Vec<_>>()
        }
    }
}

impl Polynomial {
    pub fn rnd(secret: Scalar, degree: usize) -> Self {
        let mut coefs = vec![secret];

        let mut csprng: OsRng = OsRng::new().unwrap();
        let rnd_coefs: Vec<Scalar> = (0..degree).map(|_| Scalar::random(&mut csprng)).collect();
        coefs.extend(rnd_coefs);
        
        Polynomial { a: coefs }
    }

    pub fn l_i(range: &[Scalar], i: usize) -> Scalar {
        let mut num = Scalar::one();
        let mut denum = Scalar::one();
        for j in 0..range.len() {
            if j != i {
                num *= range[j];
                denum *= range[j] - range[i];
            }
        }

        num * denum.invert()
    }

    pub fn shares(&self, n: usize) -> Vec<Share> {
        let mut shares = Vec::<Share>::with_capacity(n);
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

impl Interpolate<Share> for Polynomial {
    type Output = Scalar;
    
    fn interpolate(shares: &[Share]) -> Scalar {
        let range = shares.iter().map(|s| Scalar::from(s.i)).collect::<Vec<_>>();

        let mut acc = Scalar::zero();
        for i in 0..shares.len() {
            acc += Polynomial::l_i(&range, i) * shares[i].yi;
        }

        acc
    }
}

impl Reconstruct<Share> for Polynomial {
    type Output = Polynomial;

    fn reconstruct(shares: &[Share]) -> Polynomial {
        let range = shares.iter().map(|s| Scalar::from(s.i)).collect::<Vec<_>>();

        let mut acc = vec![Scalar::zero(); range.len()];
        for i in 0..shares.len() {
            let (num, barycentric) = lx_num_bar(&range, i);
            for j in 0..num.len() {
                acc[j] += num[j] * barycentric * shares[i].yi;
            }
        }

        cut_tail(&mut acc, Scalar::zero());
        Polynomial { a: acc }
    }
}

impl Degree for Polynomial {
    fn degree(&self) -> usize {
        self.a.len() - 1
    }
}

//-----------------------------------------------------------------------------------------------------------
// RistrettoPolynomial
//-----------------------------------------------------------------------------------------------------------
#[allow(non_snake_case)]
#[derive(Debug, PartialEq, Eq)]
pub struct RistrettoPolynomial {
    pub A: Vec<RistrettoPoint>
}

define_mul_variants!(LHS = RistrettoPolynomial, RHS = Scalar, Output = RistrettoPolynomial);
define_mul_variants!(LHS = Scalar, RHS = RistrettoPolynomial, Output = RistrettoPolynomial);
define_comut_mul!(LHS = Scalar, RHS = RistrettoPolynomial, Output = RistrettoPolynomial);
impl<'a, 'b> Mul<&'b Scalar> for &'a RistrettoPolynomial {
    type Output = RistrettoPolynomial;

    #[allow(non_snake_case)]
    fn mul(self, rhs: &'b Scalar) -> RistrettoPolynomial {
        RistrettoPolynomial {
            A: self.A.iter().map(|Ak| Ak * rhs).collect::<Vec<_>>()
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

impl Interpolate<RistrettoShare> for RistrettoPolynomial {
    type Output = RistrettoPoint;

    #[allow(non_snake_case)]
    fn interpolate(shares: &[RistrettoShare]) -> RistrettoPoint {
        let range = shares.iter().map(|s| Scalar::from(s.i)).collect::<Vec<_>>();

        let mut acc = RistrettoPoint::default();
        for i in 0..shares.len() {
            acc += Polynomial::l_i(&range, i) * shares[i].Yi;
        }

        acc
    }
}

impl Reconstruct<RistrettoShare> for RistrettoPolynomial {
    type Output = RistrettoPolynomial;

    #[allow(non_snake_case)]
    fn reconstruct(shares: &[RistrettoShare]) -> RistrettoPolynomial {
        let range = shares.iter().map(|s| Scalar::from(s.i)).collect::<Vec<_>>();

        let mut acc = vec![RistrettoPoint::default(); range.len()];
        for i in 0..shares.len() {
            let (num, barycentric) = lx_num_bar(&range, i);
            for j in 0..num.len() {
                acc[j] += num[j] * barycentric * shares[i].Yi;
            }
        }

        cut_tail(&mut acc, RistrettoPoint::default());
        RistrettoPolynomial { A: acc }
    }
}

impl Degree for RistrettoPolynomial {
    fn degree(&self) -> usize {
        self.A.len() - 1
    }
}

//-----------------------------------------------------------------------------------------------------------
// Schnorr's signature
//-----------------------------------------------------------------------------------------------------------
pub struct Signature {
    pub c: Scalar,
    pub p: Scalar
}

impl Signature {
    #[allow(non_snake_case)]
    pub fn sign(s: Scalar, P: CompressedRistretto, data: &Vec<Box<[u8]>>) -> Self {
        let mut hasher = Sha512::new()
            .chain(s.as_bytes());
        
        for d in data {
            hasher.input(d);
        }

        let m = Scalar::from_hash(hasher); 
        let M = (m * G).compress();

        let mut hasher = Sha512::new()
            .chain(P.as_bytes())
            .chain(M.as_bytes());
        
        for d in data {
            hasher.input(d);
        }

        let _c = Scalar::from_hash(hasher);

        Signature { c: _c, p: m - _c * s }
    }

    #[allow(non_snake_case)]
    pub fn verify(&self, P: CompressedRistretto, data: &Vec<Box<[u8]>>) -> bool {
        let Ps = P.decompress();
        if Ps.is_none() {
            return false;
        }

        let M = self.c * Ps.unwrap() + self.p * G;

        let mut hasher = Sha512::new()
            .chain(P.as_bytes())
            .chain(M.compress().as_bytes());
        
        for d in data {
            hasher.input(d);
        }
        
        let c = Scalar::from_hash(hasher);

        c == self.c
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    use curve25519_dalek::constants::RISTRETTO_BASEPOINT_POINT;
    const G: RistrettoPoint = RISTRETTO_BASEPOINT_POINT;

    #[allow(non_snake_case)]
    #[test]
    fn test_reconstruct() {
        let threshold = 16;
        let parties = 3*threshold + 1;

        let s = rnd_scalar();

        let poly = Polynomial::rnd(s, threshold);
        let S_poly = &poly * G;

        let shares = poly.shares(parties);
        let S_shares = shares.iter().map(|s| s * G).collect::<Vec<_>>();

        let r_poly = Polynomial::reconstruct(&shares[0..2*threshold + 1]);
        assert!(poly == r_poly);

        let S_r_poly = RistrettoPolynomial::reconstruct(&S_shares[0..2*threshold + 1]);
        assert!(S_poly == S_r_poly);
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_signature() {
        let a = rnd_scalar();
        let Pa = (a * G).compress();

        let data: Vec<Box<[u8]>> = vec![Box::new(rnd_scalar().to_bytes()), Box::new(rnd_scalar().to_bytes())];

        let sig = Signature::sign(a, Pa, &data);
        assert!(sig.verify(Pa, &data) == true);
    }
}