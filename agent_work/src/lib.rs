/// Computes the sum of two integers.
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

/// Returns true if the number is even.
pub fn is_even(n: i32) -> bool {
    n % 2 == 0
}

/// Reverses a string.
pub fn reverse(s: &str) -> String {
    s.chars().rev().collect()
}

/// Returns the n-th Fibonacci number (0-indexed, fib(0) = 0, fib(1) = 1).
pub fn fib(n: u32) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => {
            let (mut a, mut b) = (0u64, 1u64);
            for _ in 2..=n {
                let next = a + b;
                a = b;
                b = next;
            }
            b
        }
    }
}

/// Returns the maximum value in a non-empty slice of integers.
///
/// # Panics
/// Panics if the slice is empty.
pub fn max_of(values: &[i32]) -> i32 {
    values
        .iter()
        .cloned()
        .max()
        .expect("max_of called with an empty slice")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 3), 5);
        assert_eq!(add(-1, 1), 0);
    }

    #[test]
    fn test_is_even() {
        assert!(is_even(4));
        assert!(!is_even(7));
    }

    #[test]
    fn test_reverse() {
        assert_eq!(reverse("abc"), "cba");
        assert_eq!(reverse(""), "");
    }

    #[test]
    fn test_fib() {
        assert_eq!(fib(0), 0);
        assert_eq!(fib(1), 1);
        assert_eq!(fib(2), 1);
        assert_eq!(fib(10), 55);
        assert_eq!(fib(20), 6765);
    }

    #[test]
    fn test_max_of() {
        assert_eq!(max_of(&[1, 5, 3]), 5);
        assert_eq!(max_of(&[-7, -2, -9]), -2);
        assert_eq!(max_of(&[42]), 42);
    }

    #[test]
    #[should_panic]
    fn test_max_of_empty_panics() {
        max_of(&[]);
    }
}
