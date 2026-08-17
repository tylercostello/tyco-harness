/// A small demo module with testable logic.
pub mod core {
    /// Returns the larger of two integers.
    pub fn max(a: i64, b: i64) -> i64 {
        if a > b { a } else { b }
    }

    /// Computes the factorial of `n` (0! == 1).
    pub fn factorial(n: u32) -> u64 {
        (1..=n).fold(1u64, |acc, i| acc * i as u64)
    }

    /// Checks whether `n` is a palindrome (reads the same forwards and backwards).
    pub fn is_palindrome(n: i64) -> bool {
        let s = n.to_string();
        let rev: String = s.chars().rev().collect();
        s == rev
    }

    /// Reverses a string.
    pub fn reverse_str(s: &str) -> String {
        s.chars().rev().collect()
    }

    /// Splits a list of numbers into evens and odds.
    pub fn partition_even_odd(nums: &[i64]) -> (Vec<i64>, Vec<i64>) {
        let mut evens = Vec::new();
        let mut odds = Vec::new();
        for &n in nums {
            if n % 2 == 0 {
                evens.push(n);
            } else {
                odds.push(n);
            }
        }
        (evens, odds)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_max() {
            assert_eq!(max(1, 2), 2);
            assert_eq!(max(5, 3), 5);
            assert_eq!(max(-7, -2), -2);
            assert_eq!(max(4, 4), 4);
        }

        #[test]
        fn test_factorial() {
            assert_eq!(factorial(0), 1);
            assert_eq!(factorial(1), 1);
            assert_eq!(factorial(5), 120);
            assert_eq!(factorial(6), 720);
        }

        #[test]
        fn test_is_palindrome() {
            assert!(is_palindrome(12321));
            assert!(!is_palindrome(12345));
            assert!(is_palindrome(7));
            assert!(!is_palindrome(-121)); // "-121" reversed is "121-"
            assert!(!is_palindrome(10));
        }

        #[test]
        fn test_reverse_str() {
            assert_eq!(reverse_str("hello"), "olleh");
            assert_eq!(reverse_str(""), "");
            assert_eq!(reverse_str("a"), "a");
        }

        #[test]
        fn test_partition_even_odd() {
            let (evens, odds) = partition_even_odd(&[1, 2, 3, 4, 5, 6]);
            assert_eq!(evens, vec![2, 4, 6]);
            assert_eq!(odds, vec![1, 3, 5]);

            let (evens, odds) = partition_even_odd(&[]);
            assert!(evens.is_empty());
            assert!(odds.is_empty());
        }
    }
}

fn main() {
    println!("max(3, 9) = {}", core::max(3, 9));
    println!("factorial(6) = {}", core::factorial(6));
    println!("12321 is palindrome: {}", core::is_palindrome(12321));
    println!("reversed 'rust': {}", core::reverse_str("rust"));
    println!("partition: {:?}", core::partition_even_odd(&[1, 2, 3, 4]));
}
