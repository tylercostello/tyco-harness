fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn main() {
    println!("Hello, World!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn test_add() {
        assert_eq!(add(2, 3), 5);
    }
}
