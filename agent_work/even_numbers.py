def main():
    """Print all even numbers from 1 to 100 (inclusive)."""
    for n in range(1, 101):
        if n % 2 == 0:
            print(n)


if __name__ == "__main__":
    main()
