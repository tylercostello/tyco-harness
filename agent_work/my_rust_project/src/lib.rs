//! A small library with a few pure functions and their unit tests.

/// Adds two unsigned integers.
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

/// Returns the largest of two unsigned integers.
pub fn max_of_two(a: u64, b: u64) -> u64 {
    if a >= b {
        a
    } else {
        b
    }
}

/// Computes the factorial of a small unsigned integer.
pub fn factorial(n: u32) -> u64 {
    (1..=n).fold(1u64, |acc, i| acc * i as u64)
}

/// Returns all even numbers in the inclusive range `1..=n`.
pub fn even_numbers(n: u32) -> Vec<u32> {
    (1..=n).filter(|&x| x % 2 == 0).collect()
}

/// Prints all even numbers from 1 to 10, one per line.
pub fn print_even_numbers() {
    for n in even_numbers(10) {
        println!("{n}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
    fn add_handles_zero() {
        assert_eq!(add(0, 5), 5);
        assert_eq!(add(7, 0), 7);
    }

    #[test]
    fn max_of_two_picks_larger() {
        assert_eq!(max_of_two(3, 9), 9);
        assert_eq!(max_of_two(9, 3), 9);
        assert_eq!(max_of_two(4, 4), 4);
    }

    #[test]
    fn factorial_base_cases() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(5), 120);
    }

    #[test]
    fn even_numbers_from_1_to_10() {
        assert_eq!(even_numbers(10), vec![2, 4, 6, 8, 10]);
        assert_eq!(even_numbers(1), Vec::<u32>::new());
        assert_eq!(even_numbers(2), vec![2]);
    }
}
