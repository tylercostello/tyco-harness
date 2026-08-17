use my_app::{add, is_even, reverse};

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
