use my_app::{add, fib, is_even, max_of, reverse};

#[test]
fn integration_add() {
    assert_eq!(add(10, 20), 30);
}

#[test]
fn integration_is_even() {
    assert!(is_even(2));
    assert!(!is_even(3));
}

#[test]
fn integration_reverse() {
    assert_eq!(reverse("rust"), "tsur");
}

#[test]
fn integration_fib() {
    assert_eq!(fib(0), 0);
    assert_eq!(fib(5), 5);
    assert_eq!(fib(10), 55);
}

#[test]
fn integration_max_of() {
    assert_eq!(max_of(&[3, 9, 1]), 9);
    assert_eq!(max_of(&[-1, -5]), -1);
}
