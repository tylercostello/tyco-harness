//! Prints the first 1000 prime numbers, optimized for speed.
//!
//! Strategy:
//! - Use the Sieve of Eratosthenes, which runs in O(n log log n) instead of
//!   the O(n sqrt(n)) of naive trial division.
//! - Derive a guaranteed upper bound for the 1000th prime so we don't need to
//!   guess a limit: for n >= 6, p_n < n * (ln n + ln ln n).
//! - Build the entire output in one `String` and emit it with a single `print!`,
//!   avoiding 1000 separate syscalls.

use std::io::{self, Write};

/// Guaranteed upper bound for the n-th prime (n >= 6).
///
/// By Rosser's theorem / Dusart's bound, p_n < n * (ln n + ln ln n) for n >= 6.
fn nth_prime_upper_bound(n: usize) -> usize {
    let nf = n as f64;
    let bound = nf * (nf.ln() + nf.ln().ln());
    // Add a small margin and ensure we can index up to `limit` inclusive.
    bound.ceil() as usize + 1
}

/// Classic Sieve of Eratosthenes. Returns a boolean vector where index i is
/// true iff i is prime.
fn sieve(limit: usize) -> Vec<bool> {
    let mut is_prime = vec![true; limit + 1];
    is_prime[0] = false;
    is_prime[1] = false;

    // Only need to mark multiples starting from i*i, and only for i <= sqrt(limit).
    let mut i = 2;
    while i * i <= limit {
        if is_prime[i] {
            let mut j = i * i;
            while j <= limit {
                is_prime[j] = false;
                j += i;
            }
        }
        i += 1;
    }
    is_prime
}

fn main() -> io::Result<()> {
    const COUNT: usize = 1000;

    let limit = nth_prime_upper_bound(COUNT);
    let is_prime = sieve(limit);

    // Collect the first COUNT primes.
    let mut out = String::with_capacity(COUNT * 5); // ~5 chars per number on average
    let mut found = 0usize;
    for (value, &prime) in is_prime.iter().enumerate() {
        if prime {
            out.push_str(&value.to_string());
            out.push(' ');
            found += 1;
            if found == COUNT {
                break;
            }
        }
    }

    let stdout = io::stdout();
    let mut handle = stdout.lock();
    handle.write_all(out.as_bytes())?;
    handle.write_all(b"\n")?;
    handle.flush()?;
    Ok(())
}
