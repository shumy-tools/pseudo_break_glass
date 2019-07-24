mod macros;
mod shamir;

use std::time::{Instant, Duration};

use clap::{Arg, App};

use curve25519_dalek::ristretto::{RistrettoPoint, CompressedRistretto};
use curve25519_dalek::scalar::Scalar;
use curve25519_dalek::constants::RISTRETTO_BASEPOINT_POINT;

use sha2::{Sha512, Digest};
use rand::prelude::*;

use shamir::{Reconstruct, Polynomial, RistrettoPolynomial, Share, RistrettoShare};

const RUNS: usize = 100;
const G: RistrettoPoint = RISTRETTO_BASEPOINT_POINT;

fn rnd_scalar() -> Scalar {
    let val = rand::thread_rng().gen::<[u8; 32]>();
    Scalar::hash_from_bytes::<Sha512>(&val)
}

fn t_1_slice(selected: &Vec<RistrettoShare>, threshold: usize) -> Vec<RistrettoShare> {
    let mut slice: Vec<RistrettoShare> = vec![];
    slice.extend(&selected[0..threshold]);
    slice.extend(&selected[threshold..threshold+1]);
    assert!(slice.len() == threshold+1);
    
    slice
}

#[allow(non_snake_case)]
fn multiparty_stats(parties: usize, threshold: usize) {
    println!("---Multiparty computation stats ({:?} runs)---", RUNS);

    let y = rnd_scalar();
    let Y = y * G;

    let poly = Polynomial::rnd(y, threshold);
    let y_shares = poly.shares(parties);

    let mut total = Duration::new(0, 0);
    for _ in 0..RUNS {
        // derive Pc from a fragment of public information
        let public_x = rand::thread_rng().gen::<[u8; 32]>();
        let Pc = RistrettoPoint::hash_from_bytes::<Sha512>(&public_x);
        
        // derive Pc + R = W and Yr
        let r = rnd_scalar();
        let R = r * G;
        let compressed_W = (Pc + R).compress();
        let compressed_Yr = (r * Y).compress();
        
        // client to parties transmission <W,Yr>
        let start = Instant::now();

            // Use CompressedRistretto::from_slice(..) to get a point from the input.
            // The CompressedRistretto::decompress() validates the point.
            let W = compressed_W.decompress().unwrap();
            let Yr = compressed_Yr.decompress().unwrap();

            // each party derives the partial pesudonym
            let Wy_shares = y_shares.iter().map(|yi| yi * W).collect::<Vec<_>>();

            // ...attack with same degree polynomial...
            let fake_y = rnd_scalar();
            let fake_poly = Polynomial::rnd(fake_y, threshold);
            let fake_y_shares = fake_poly.shares(threshold);
            let fake_Wy_shares = fake_y_shares.iter()
                .zip(&y_shares)
                .map(|(yi, fi)| (fi + yi) * W)
                .collect::<Vec<_>>();

            // select a subset (2t + 1) of shares
            let mut rng = rand::thread_rng();
            let mut selected: Vec<RistrettoShare> = vec![];
            selected.extend(&fake_Wy_shares[0..threshold]);              // t fake shares
            selected.extend(&Wy_shares[threshold..2*threshold+1]);       // t + 1 original shares
            selected.shuffle(&mut rng);                                  // shuffle fakes with originals
            assert!(selected.len() == 2*threshold+1);

            // derive the fake_Wy point
            let fake_Wy = RistrettoPolynomial::reconstruct(&selected);

                // ..detect attack...
                let slice = t_1_slice(&selected, threshold);
                let slice_Wy = RistrettoPolynomial::reconstruct(&slice);
                assert!(fake_Wy != slice_Wy); // fail if the attack is not detected

        let run = Instant::now() - start;
        total += run;

        // the next verification step is not contained in the total time because the process only runs one time.
            // derive the correct Wy point
            let mut rng = rand::thread_rng();
            let mut ok: Vec<RistrettoShare> = vec![];
            ok.extend(&Wy_shares[0..2*threshold+1]);
            ok.shuffle(&mut rng); 

            let Wy = RistrettoPolynomial::reconstruct(&ok);

                // ..verification...
                let slice = t_1_slice(&ok, threshold);
                let slice_Wy = RistrettoPolynomial::reconstruct(&slice);
                assert!(Wy == slice_Wy); // should not detect any attack

            // confirm the expected result from the original shares
            let PI = Wy - Yr;
            assert!(y * Pc == PI);
    }

    let avg = (total.as_micros() as f64) / (1000.0*RUNS as f64);
    println!("   Avg. per run: {:?}ms", avg);
}

#[allow(non_snake_case)]
fn master_key_stats(parties: usize, threshold: usize) {
    println!("---Master key setup stats---");

    // setup all party keys
    let mut p_si = Vec::<Scalar>::with_capacity(parties);
    let mut p_Pi = Vec::<RistrettoPoint>::with_capacity(parties);
    for _ in 0..parties {
        let si = rnd_scalar();
        p_si.push(si);

        let Pi = si * G;
        p_Pi.push(Pi);
    }

    let mut total = Duration::new(0, 0);

    // (Matrix Setup) setup the secret and public matrix
    let start = Instant::now();

    let mut s_matrix = Vec::<Vec::<Scalar>>::with_capacity(parties);
    let mut P_matrix = Vec::<Vec::<CompressedRistretto>>::with_capacity(parties);
    for i in 0..parties {
        let mut i_p = Vec::<Scalar>::with_capacity(parties);
        let mut i_P = Vec::<CompressedRistretto>::with_capacity(parties);
        for j in 0..parties {
            let dh_p = (p_si[i] * p_Pi[j]).compress(); // perform a Diffie-Hellman between parties (i, j)
            let mut hasher = Sha512::new();
            hasher.input(dh_p.as_bytes());
            let dh_s = Scalar::from_hash(hasher);
            i_p.push(dh_s);
            i_P.push((dh_s * G).compress());
        }

        s_matrix.push(i_p);
        P_matrix.push(i_P);
    }

    // is the public matrix symmetric?
    for i in 0..parties {
        for j in 0..parties {
            assert!(P_matrix[i][j] == P_matrix[j][i]);
        }
    }

    let run = Instant::now() - start;
    total += run;
    println!("   Matrix setup (size={:?}Kb, time={:?}ms)", ((P_matrix[0][0].as_bytes().len() * parties * parties) as f64) / 1024.0, (run.as_micros() as f64) / 1000.0);
    
    // (Step 1 & 2) commit to pre-defined polynomials and construct encrypted shares
    let start = Instant::now();
    
    let mut y = Scalar::zero();
    let mut i_Ak = Vec::<RistrettoPolynomial>::with_capacity(parties);
    let mut i_e_j = Vec::<Vec::<Share>>::with_capacity(parties);
    for i in 0..parties {
        let i_y = rnd_scalar();
        y += i_y;

        let i_ak = Polynomial::rnd(i_y, threshold);
        let i_y_shares = i_ak.shares(parties);

        // commit to a polynomial
        i_Ak.push(i_ak * G);

        // encrypt shares
        let mut e_j = Vec::<Share>::with_capacity(parties);
        for j in 0..parties {
            e_j.push( s_matrix[i][j] + i_y_shares[j] );
        }
        i_e_j.push(e_j);
    }

    let run = Instant::now() - start;
    total += run;
    println!("   Commit and encrypt: {:?}ms", (run.as_micros() as f64) / 1000.0);

    // (Step 3) verify if shares are correct
    let start = Instant::now();

    for i in 0..parties {
        for j in 0..parties {
            // The CompressedRistretto::decompress() validates the point.
            let i_Y_j = i_e_j[i][j] * G - P_matrix[i][j].decompress().unwrap();
            assert!(i_Ak[i].verify(i_Y_j) == true);
        }
    }

    let run = Instant::now() - start;
    total += run;
    println!("   Share verification: {:?}ms", (run.as_micros() as f64) / 1000.0);

    // reconstruct master key
    let start = Instant::now();

    let mut y_j = Vec::<Share>::with_capacity(parties);
    for j in 0..parties {
        // sum the shares of different sources
        let mut sum_j = Scalar::zero();
        for i in 0..parties {
            sum_j += i_e_j[i][j].yi - s_matrix[i][j];
        }
        y_j.push(Share { i: i_e_j[0][j].i, yi: sum_j });
    }

    let run = Instant::now() - start;
    total += run;

    // confirm the expected result
    let y_res = Polynomial::reconstruct(&y_j);
    assert!(y == y_res);

    println!("   Total: {:?}ms", (total.as_micros() as f64) / 1000.0);
}

fn main() {
    let matches = App::new("Statistics for FedPI")
        .version("1.0")
        .author("Micael Pedrosa <micaelpedrosa@ua.pt>")
        .about("Simulations and measurements for the FedPI protocol.")
        .arg(Arg::with_name("threshold")
            .help("Sets the threshold number (t). The number of parties are set automatically to 3t+1.")
            .required(true)
            .short("t")
            .long("threshold")
            .takes_value(true))
        .get_matches();
    
    let str_threshold = matches.value_of("threshold").unwrap();

    let threshold = str_threshold.parse::<usize>().unwrap();
    let parties = 3 * threshold + 1;
    println!("Setup: (t={}, 3t+1={})", threshold, parties);

    master_key_stats(parties, threshold);
    multiparty_stats(parties, threshold);
}

#[cfg(test)]
mod tests {
    use super::*;
    use shamir::l_i;

    #[test]
    fn attack() {
        // Verify that the equality holds:
        // \sum_{i=1}^{t} y_{i} \cdot (l_{i}^{[t+1]} - l_{i}^{[n]}) = \sum_{t+1}^{n} y_{i} \cdot l_{i}^{[n]} - y_{t+1} \cdot l_{t+1}^{[t+1]}
        // this can be used to attack t shares.

        let threshold = 4;
        let parties = 3*threshold + 1;

        let poly = Polynomial::rnd(rnd_scalar(), threshold);
        let shares = poly.shares(parties);

        let n = shares[0..shares.len()].iter().map(|s| Scalar::from(s.i)).collect::<Vec<_>>();
        let t_1 = shares[0..threshold+1].iter().map(|s| Scalar::from(s.i)).collect::<Vec<_>>();

        let mut acc1 = Scalar::zero();
        for i in 0..threshold {
            acc1 += shares[i].yi * (l_i(&t_1, i) - l_i(&n, i));
        }

        let mut acc2 = - shares[threshold].yi * l_i(&t_1, threshold);
        for i in threshold..shares.len() {
            acc2 += shares[i].yi * l_i(&n, i);
        }

        assert!(acc1 == acc2);
    }
}
