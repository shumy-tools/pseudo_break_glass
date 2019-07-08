mod macros;
mod shamir;

use std::time::{Instant, Duration};

use clap::{Arg, App};

use curve25519_dalek::ristretto::RistrettoPoint;
use curve25519_dalek::scalar::Scalar;

use sha2::Sha512;
use rand::prelude::*;

use shamir::{Reconstruct, SecretSharing, RistrettoShare};

fn rand_scalar() -> Scalar {
    let val = rand::thread_rng().gen::<[u8; 32]>();
    Scalar::hash_from_bytes::<Sha512>(&val)
}

#[allow(non_snake_case)]
fn main() {
    const RUNS: usize = 500;

    let matches = App::new("Statistics for P-ID")
        .version("1.0")
        .author("Micael Pedrosa <micaelpedrosa@ua.pt>")
        .about("Performs time measurements for runs of the PID function.")
        .arg(Arg::with_name("parties")
            .help("Sets the number of parties (n)")
            .required(true)
            .short("p")
            .long("parties")
            .takes_value(true))
        .arg(Arg::with_name("threshold")
            .help("Sets the threshold number (t)")
            .required(true)
            .short("t")
            .long("threshold")
            .takes_value(true))
        .get_matches();

    let str_parties = matches.value_of("parties").unwrap();
    let str_threshold = matches.value_of("threshold").unwrap();

    let parties = str_parties.parse::<usize>().unwrap();
    let threshold = str_threshold.parse::<usize>().unwrap();
    println!("Setup: (parties={}, threshold={})", parties, threshold);

    let tss = SecretSharing { n: parties, t: threshold };

    // setup the master key shares
    let y = rand_scalar();
    let y_shares = tss.share(y);

    let mut total = Duration::new(0, 0);
    for _ in 0..RUNS {
        let start = Instant::now();

        // select a random fragment of public information and derive PC
        let public_x = rand::thread_rng().gen::<[u8; 32]>();
        let Pc = RistrettoPoint::hash_from_bytes::<Sha512>(&public_x);
        
        // derive r * Pc => Rc
        let r = rand_scalar();
        let Rc = r * Pc;
        
        // client to parties transmission simulation (Rc)
        //sleep(Duration::new(2, 0));

            // each party derives the partial pesudonym
            let mut lambda_r_shares = y_shares.iter()
                .map(|yi| yi * Rc).collect::<Vec<RistrettoShare>>();

            // select a subset (> t) of random shares
            let mut rng = rand::thread_rng();
            lambda_r_shares.shuffle(&mut rng);
            let selected = &lambda_r_shares[0..threshold+1];

        // parties to client transmission simulation (selected)
        //sleep(Duration::new(2, 0));

        let lambda_r = tss.reconstruct(selected);
        let lambda = r.invert() * lambda_r;

        let run = Instant::now() - start;
        assert!(y * Pc == lambda); // confirm the expected result of Lagrange interpolation
        total += run;
    }

    let avg = (total.as_micros() as f64) / (1000.0*RUNS as f64);
    println!("Avg. per run ({:?} runs): {:?}ms", RUNS, avg);
}
