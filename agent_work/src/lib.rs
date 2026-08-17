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
}
