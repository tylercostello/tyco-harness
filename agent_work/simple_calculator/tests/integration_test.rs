use simple_calculator::{add, divide, evaluate, multiply, remainder, subtract};

#[test]
fn integration_add_and_sub() {
    assert_eq!(add(10, 20), 30);
    assert_eq!(subtract(20, 10), 10);
}

#[test]
fn integration_multiply_and_divide() {
    assert_eq!(multiply(6, 7), 42);
    assert_eq!(divide(42, 6), Some(7));
    assert_eq!(divide(42, 0), None);
}

#[test]
fn integration_evaluate_variants() {
    assert_eq!(evaluate(100, "-", 50), Some(50));
    assert_eq!(evaluate(100, "%", 7), Some(2));
    assert_eq!(evaluate(100, "/", 0), None);
    assert_eq!(evaluate(100, "?", 7), None);
}
