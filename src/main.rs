mod macros;
mod shamir;

use std::time::{Instant, Duration};

use clap::{Arg, App};

use curve25519_dalek::ristretto::{RistrettoPoint, CompressedRistretto};
use curve25519_dalek::scalar::Scalar;

use sha2::{Sha512, Digest};
use rand::prelude::*;

use shamir::{Reconstruct, Polynomial, RistrettoPolynomial, Share, RistrettoShare};

const RUNS: usize = 500;

fn rand_scalar() -> Scalar {
    let val = rand::thread_rng().gen::<[u8; 32]>();
    Scalar::hash_from_bytes::<Sha512>(&val)
}

#[allow(non_snake_case)]
fn multiparty_stats(parties: usize, threshold: usize) {
    println!("---Multiparty computation stats (500 runs)---");
    let G: RistrettoPoint = RistrettoPoint::default();

    let y = rand_scalar();
    let Y = y * G;

    let poly = Polynomial::rnd(y, threshold);
    let y_shares = poly.shares(parties);

    // piece of public information
    let public_x = rand::thread_rng().gen::<[u8; 32]>();

    let mut total = Duration::new(0, 0);
    for _ in 0..RUNS {
        let start = Instant::now();

        // derive Pc from a fragment of public information
        let Pc = RistrettoPoint::hash_from_bytes::<Sha512>(&public_x);
        
        // derive Pc + R = W and Yr
        let r = rand_scalar();
        let R = r * G;
        let W = Pc + R;
        let Yr = r * Y;
        
        // client to parties transmission simulation <W,Yr>
        //sleep(Duration::new(2, 0));

            // each party derives the partial pesudonym
            let mut pi_shares = y_shares.iter().map(|yi| yi * W).collect::<Vec<RistrettoShare>>();

            // select a subset (> t) of random shares
            let mut rng = rand::thread_rng();
            pi_shares.shuffle(&mut rng);
            let selected = &pi_shares[0..threshold + 1];

            // derive the pseudonym
            let PI = Polynomial::reconstruct(selected, threshold) - Yr;

        // party to client transmission simulation
        //sleep(Duration::new(2, 0));

        let run = Instant::now() - start;
        assert!(y * Pc == PI); // confirm the expected result
        total += run;
    }

    let avg = (total.as_micros() as f64) / (1000.0*RUNS as f64);
    println!("   Avg. per run: {:?}ms", avg);
}

#[allow(non_snake_case)]
fn master_key_stats(parties: usize, threshold: usize) {
    println!("---Master key setup stats---");
    let G: RistrettoPoint = RistrettoPoint::default();

    // setup all party keys
    let mut p_si = Vec::<Scalar>::new();
    let mut p_Pi = Vec::<RistrettoPoint>::new();
    for _ in 0..parties {
        let si = rand_scalar();
        p_si.push(si);

        let Pi = si * G;
        p_Pi.push(Pi);
    }

    let mut total = Duration::new(0, 0);

    // (Matrix Setup) setup the secret and public matrix
    let start = Instant::now();

    let mut s_matrix = Vec::<Vec::<Scalar>>::new();
    let mut P_matrix = Vec::<Vec::<CompressedRistretto>>::new();
    for i in 0..parties {
        let mut i_p = Vec::<Scalar>::new();
        let mut i_P = Vec::<CompressedRistretto>::new();
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
    let mut i_Ak = Vec::<RistrettoPolynomial>::new();
    let mut i_e_j = Vec::<Vec::<Share>>::new();
    for i in 0..parties {
        let i_y = rand_scalar();
        y += i_y;

        let i_ak = Polynomial::rnd(i_y, threshold);
        let i_y_shares = i_ak.shares(parties);

        // commit to a polynomial
        i_Ak.push(i_ak * G);

        // encrypt shares
        let mut e_j = Vec::<Share>::new();
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
            let i_Y_j = i_e_j[i][j] * G - P_matrix[i][j].decompress().unwrap();
            assert!(i_Ak[i].verify(i_Y_j) == true);
        }
    }

    let run = Instant::now() - start;
    total += run;
    println!("   Share verification: {:?}ms", (run.as_micros() as f64) / 1000.0);

    // reconstruct master key
    let start = Instant::now();

    let mut y_j = Vec::<Share>::new();
    for j in 0..parties {
        // sum the shares of different sources
        let mut sum_j = Scalar::zero();
        for i in 0..parties {
            sum_j += i_e_j[i][j].yi - s_matrix[i][j];
        }
        y_j.push(Share { i: i_e_j[0][j].i, yi: sum_j });
    }
    let y_res = Polynomial::reconstruct(&y_j, threshold);
    assert!(y == y_res);

    let run = Instant::now() - start;
    total += run;
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
