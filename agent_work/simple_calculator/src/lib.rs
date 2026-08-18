/// A tiny calculator with basic arithmetic operations.

/// Adds two integers.
pub fn add(a: i64, b: i64) -> i64 {
    a.wrapping_add(b)
}

/// Subtracts `b` from `a`.
pub fn subtract(a: i64, b: i64) -> i64 {
    a.wrapping_sub(b)
}

/// Multiplies two integers.
pub fn multiply(a: i64, b: i64) -> i64 {
    a.wrapping_mul(b)
}

/// Integer division. Returns `None` if the divisor is zero.
pub fn divide(a: i64, b: i64) -> Option<i64> {
    if b == 0 {
        None
    } else {
        Some(a / b)
    }
}

/// Computes the remainder of `a % b`. Returns `None` if the divisor is zero.
pub fn remainder(a: i64, b: i64) -> Option<i64> {
    if b == 0 {
        None
    } else {
        Some(a % b)
    }
}

/// Evaluates a simple expression described as a tuple of (left, op, right).
///
/// Supported operators are `+`, `-`, `*`, `/`, and `%`.
/// Returns `None` for division by zero or an unsupported operator.
pub fn evaluate(left: i64, op: &str, right: i64) -> Option<i64> {
    match op {
        "+" => Some(add(left, right)),
        "-" => Some(subtract(left, right)),
        "*" => Some(multiply(left, right)),
        "/" => divide(left, right),
        "%" => remainder(left, right),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 3), 5);
        assert_eq!(add(-2, 2), 0);
        assert_eq!(add(-5, -3), -8);
    }

    #[test]
    fn test_subtract() {
        assert_eq!(subtract(5, 3), 2);
        assert_eq!(subtract(3, 5), -2);
        assert_eq!(subtract(0, 1), -1);
    }

    #[test]
    fn test_multiply() {
        assert_eq!(multiply(4, 5), 20);
        assert_eq!(multiply(-4, 5), -20);
        assert_eq!(multiply(0, 100), 0);
    }

    #[test]
    fn test_divide() {
        assert_eq!(divide(10, 2), Some(5));
        assert_eq!(divide(-10, 2), Some(-5));
        assert_eq!(divide(10, 3), Some(3));
        assert_eq!(divide(10, 0), None);
    }

    #[test]
    fn test_remainder() {
        assert_eq!(remainder(10, 3), Some(1));
        assert_eq!(remainder(10, 5), Some(0));
        assert_eq!(remainder(10, 0), None);
    }

    #[test]
    fn test_evaluate() {
        assert_eq!(evaluate(7, "+", 3), Some(10));
        assert_eq!(evaluate(7, "-", 3), Some(4));
        assert_eq!(evaluate(7, "*", 3), Some(21));
        assert_eq!(evaluate(7, "/", 2), Some(3));
        assert_eq!(evaluate(7, "%", 2), Some(1));
        assert_eq!(evaluate(7, "/", 0), None);
        assert_eq!(evaluate(7, "^", 2), None);
    }

    #[test]
    fn test_large_values_wrap() {
        // wrapping_add should not panic on overflow.
        assert_eq!(add(i64::MAX, 1), i64::MIN);
        assert_eq!(subtract(i64::MIN, 1), i64::MAX);
    }
}
