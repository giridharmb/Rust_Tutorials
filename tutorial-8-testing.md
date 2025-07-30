# Tutorial 8: Testing & Documentation

## Unit Tests

Rust has built-in testing support. Tests are functions annotated with `#[test]`.

```rust
// src/lib.rs

// Function to test
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        Err("Cannot divide by zero".to_string())
    } else {
        Ok(a / b)
    }
}

// Test module
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_add() {
        assert_eq!(add(2, 2), 4);
        assert_eq!(add(-1, 1), 0);
        assert_eq!(add(0, 0), 0);
    }
    
    #[test]
    fn test_divide_success() {
        assert_eq!(divide(10.0, 2.0), Ok(5.0));
        assert_eq!(divide(-10.0, 2.0), Ok(-5.0));
    }
    
    #[test]
    fn test_divide_by_zero() {
        assert_eq!(divide(10.0, 0.0), Err("Cannot divide by zero".to_string()));
    }
    
    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_panic() {
        assert!(false, "This should panic");
    }
    
    #[test]
    #[ignore]
    fn expensive_test() {
        // This test is ignored by default
        // Run with: cargo test -- --ignored
        println!("This is an expensive test");
    }
}
```

## Test Organization

```rust
// src/lib.rs

#[derive(Debug, PartialEq)]
pub struct Rectangle {
    width: u32,
    height: u32,
}

impl Rectangle {
    pub fn new(width: u32, height: u32) -> Self {
        Rectangle { width, height }
    }
    
    pub fn area(&self) -> u32 {
        self.width * self.height
    }
    
    pub fn can_hold(&self, other: &Rectangle) -> bool {
        self.width > other.width && self.height > other.height
    }
    
    pub fn square(size: u32) -> Rectangle {
        Rectangle {
            width: size,
            height: size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Helper function for tests
    fn create_test_rectangles() -> (Rectangle, Rectangle, Rectangle) {
        let rect1 = Rectangle::new(8, 7);
        let rect2 = Rectangle::new(5, 1);
        let rect3 = Rectangle::new(3, 9);
        (rect1, rect2, rect3)
    }
    
    #[test]
    fn larger_can_hold_smaller() {
        let (rect1, rect2, _) = create_test_rectangles();
        assert!(rect1.can_hold(&rect2));
    }
    
    #[test]
    fn smaller_cannot_hold_larger() {
        let (rect1, rect2, _) = create_test_rectangles();
        assert!(!rect2.can_hold(&rect1));
    }
    
    #[test]
    fn rectangles_with_different_dimensions() {
        let (rect1, _, rect3) = create_test_rectangles();
        assert!(!rect1.can_hold(&rect3));
        assert!(!rect3.can_hold(&rect1));
    }
    
    #[test]
    fn test_area() {
        let rect = Rectangle::new(10, 20);
        assert_eq!(rect.area(), 200);
    }
    
    #[test]
    fn test_square() {
        let square = Rectangle::square(5);
        assert_eq!(square.width, 5);
        assert_eq!(square.height, 5);
        assert_eq!(square.area(), 25);
    }
}
```

## Integration Tests

Integration tests go in the `tests` directory and test the library as external code would use it.

```rust
// tests/integration_test.rs

// Import the library crate
use my_project;

#[test]
fn test_public_api() {
    let rect = my_project::Rectangle::new(10, 20);
    assert_eq!(rect.area(), 200);
}

#[test]
fn test_complex_scenario() {
    let square1 = my_project::Rectangle::square(5);
    let square2 = my_project::Rectangle::square(3);
    
    assert!(square1.can_hold(&square2));
    assert!(!square2.can_hold(&square1));
}

// tests/common/mod.rs
// Shared test utilities
pub fn setup() -> TestContext {
    TestContext {
        test_data: vec![1, 2, 3, 4, 5],
    }
}

pub struct TestContext {
    pub test_data: Vec<i32>,
}

// tests/another_integration_test.rs
mod common;

#[test]
fn test_with_common_setup() {
    let context = common::setup();
    assert_eq!(context.test_data.len(), 5);
}
```

## Advanced Testing Patterns

```rust
// src/lib.rs

use std::fmt::Debug;
use std::io::{self, Write};

// Testing with traits
pub trait Storage {
    fn save(&mut self, key: &str, value: &str) -> Result<(), String>;
    fn load(&self, key: &str) -> Result<String, String>;
}

pub struct FileStorage {
    base_path: String,
}

impl Storage for FileStorage {
    fn save(&mut self, key: &str, value: &str) -> Result<(), String> {
        // Implementation
        Ok(())
    }
    
    fn load(&self, key: &str) -> Result<String, String> {
        // Implementation
        Ok(String::from("value"))
    }
}

// Service that depends on Storage
pub struct DataService<S: Storage> {
    storage: S,
}

impl<S: Storage> DataService<S> {
    pub fn new(storage: S) -> Self {
        DataService { storage }
    }
    
    pub fn process_data(&mut self, key: &str, data: &str) -> Result<String, String> {
        let processed = format!("Processed: {}", data);
        self.storage.save(key, &processed)?;
        Ok(processed)
    }
}

// Testing with custom assertions
pub fn assert_close(a: f64, b: f64, epsilon: f64) {
    assert!(
        (a - b).abs() < epsilon,
        "Values not close enough: {} vs {}, epsilon: {}",
        a, b, epsilon
    );
}

// Parameterized tests using macros
macro_rules! test_operation {
    ($($name:ident: $value:expr,)*) => {
        $(
            #[test]
            fn $name() {
                let (input, expected) = $value;
                assert_eq!(process(input), expected);
            }
        )*
    }
}

fn process(x: i32) -> i32 {
    x * 2
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Mock implementation for testing
    struct MockStorage {
        data: std::collections::HashMap<String, String>,
        save_called: std::cell::RefCell<bool>,
    }
    
    impl MockStorage {
        fn new() -> Self {
            MockStorage {
                data: std::collections::HashMap::new(),
                save_called: std::cell::RefCell::new(false),
            }
        }
        
        fn was_save_called(&self) -> bool {
            *self.save_called.borrow()
        }
    }
    
    impl Storage for MockStorage {
        fn save(&mut self, key: &str, value: &str) -> Result<(), String> {
            *self.save_called.borrow_mut() = true;
            self.data.insert(key.to_string(), value.to_string());
            Ok(())
        }
        
        fn load(&self, key: &str) -> Result<String, String> {
            self.data.get(key)
                .cloned()
                .ok_or_else(|| "Key not found".to_string())
        }
    }
    
    #[test]
    fn test_data_service_with_mock() {
        let mock_storage = MockStorage::new();
        let mut service = DataService::new(mock_storage);
        
        let result = service.process_data("test_key", "test_data");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Processed: test_data");
    }
    
    #[test]
    fn test_custom_assertion() {
        assert_close(1.0, 1.0001, 0.001);
        // assert_close(1.0, 2.0, 0.001); // This would fail
    }
    
    // Using the parameterized test macro
    test_operation! {
        test_zero: (0, 0),
        test_one: (1, 2),
        test_negative: (-5, -10),
        test_large: (100, 200),
    }
}
```

## Documentation

Rust uses documentation comments to generate HTML documentation.

```rust
//! # My Library
//! 
//! `my_library` provides utilities for working with geometric shapes.
//! 
//! ## Quick Start
//! 
//! ```
//! use my_library::shapes::{Circle, Rectangle};
//! 
//! let circle = Circle::new(5.0);
//! let area = circle.area();
//! assert_eq!(area, std::f64::consts::PI * 25.0);
//! ```

/// A module containing shape definitions and operations.
pub mod shapes {
    use std::f64::consts::PI;
    
    /// Represents a circle with a given radius.
    /// 
    /// # Examples
    /// 
    /// ```
    /// use my_library::shapes::Circle;
    /// 
    /// let circle = Circle::new(10.0);
    /// assert_eq!(circle.radius(), 10.0);
    /// ```
    #[derive(Debug, Clone, Copy)]
    pub struct Circle {
        radius: f64,
    }
    
    impl Circle {
        /// Creates a new circle with the specified radius.
        /// 
        /// # Arguments
        /// 
        /// * `radius` - The radius of the circle. Must be positive.
        /// 
        /// # Panics
        /// 
        /// Panics if `radius` is negative.
        /// 
        /// # Examples
        /// 
        /// ```
        /// use my_library::shapes::Circle;
        /// 
        /// let circle = Circle::new(5.0);
        /// ```
        pub fn new(radius: f64) -> Self {
            assert!(radius >= 0.0, "Radius must be non-negative");
            Circle { radius }
        }
        
        /// Returns the radius of the circle.
        pub fn radius(&self) -> f64 {
            self.radius
        }
        
        /// Calculates the area of the circle.
        /// 
        /// # Returns
        /// 
        /// The area as a `f64`.
        /// 
        /// # Examples
        /// 
        /// ```
        /// use my_library::shapes::Circle;
        /// 
        /// let circle = Circle::new(2.0);
        /// let area = circle.area();
        /// assert!((area - 12.566370614359172).abs() < 0.0001);
        /// ```
        pub fn area(&self) -> f64 {
            PI * self.radius * self.radius
        }
        
        /// Calculates the circumference of the circle.
        pub fn circumference(&self) -> f64 {
            2.0 * PI * self.radius
        }
    }
    
    /// A rectangle with width and height.
    /// 
    /// Rectangles are created using the [`Rectangle::new`] method or
    /// the [`Rectangle::square`] convenience method for squares.
    /// 
    /// [`Rectangle::new`]: Rectangle::new
    /// [`Rectangle::square`]: Rectangle::square
    #[derive(Debug, Clone, Copy)]
    pub struct Rectangle {
        width: f64,
        height: f64,
    }
    
    impl Rectangle {
        /// Creates a new rectangle.
        /// 
        /// # Errors
        /// 
        /// Returns an error if width or height are negative.
        pub fn new(width: f64, height: f64) -> Result<Self, String> {
            if width < 0.0 || height < 0.0 {
                Err("Width and height must be non-negative".to_string())
            } else {
                Ok(Rectangle { width, height })
            }
        }
        
        /// Creates a square with equal width and height.
        pub fn square(size: f64) -> Result<Self, String> {
            Self::new(size, size)
        }
        
        /// Returns the area of the rectangle.
        pub fn area(&self) -> f64 {
            self.width * self.height
        }
    }
}

/// Utilities for mathematical operations.
pub mod math {
    /// Computes the factorial of a number.
    /// 
    /// # Arguments
    /// 
    /// * `n` - A non-negative integer
    /// 
    /// # Returns
    /// 
    /// Returns `Some(factorial)` if the computation doesn't overflow,
    /// otherwise returns `None`.
    /// 
    /// # Examples
    /// 
    /// Basic usage:
    /// 
    /// ```
    /// use my_library::math::factorial;
    /// 
    /// assert_eq!(factorial(5), Some(120));
    /// assert_eq!(factorial(0), Some(1));
    /// ```
    /// 
    /// Handling overflow:
    /// 
    /// ```
    /// use my_library::math::factorial;
    /// 
    /// // This will overflow for large numbers
    /// assert_eq!(factorial(100), None);
    /// ```
    pub fn factorial(n: u32) -> Option<u64> {
        match n {
            0 => Some(1),
            n => {
                let mut result = 1u64;
                for i in 1..=n {
                    result = result.checked_mul(i as u64)?;
                }
                Some(result)
            }
        }
    }
}
```

## Benchmarks

Benchmarks help measure performance. They require the nightly compiler and the `test` crate.

```rust
// benches/benchmark.rs
#![feature(test)]

extern crate test;

use test::Bencher;

#[bench]
fn bench_factorial(b: &mut Bencher) {
    b.iter(|| {
        my_library::math::factorial(20)
    });
}

#[bench]
fn bench_circle_area(b: &mut Bencher) {
    let circle = my_library::shapes::Circle::new(10.0);
    b.iter(|| {
        circle.area()
    });
}

// Benchmark with setup
#[bench]
fn bench_vec_operations(b: &mut Bencher) {
    let data: Vec<i32> = (0..1000).collect();
    
    b.iter(|| {
        let sum: i32 = data.iter().sum();
        test::black_box(sum); // Prevent optimization
    });
}
```

## Property-Based Testing

Using the `proptest` crate for property-based testing:

```toml
# Cargo.toml
[dev-dependencies]
proptest = "1.0"
```

```rust
// src/lib.rs

pub fn reverse<T: Clone>(list: &[T]) -> Vec<T> {
    let mut result = list.to_vec();
    result.reverse();
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    
    #[test]
    fn test_reverse_basic() {
        assert_eq!(reverse(&[1, 2, 3]), vec![3, 2, 1]);
        assert_eq!(reverse(&['a', 'b', 'c']), vec!['c', 'b', 'a']);
    }
    
    proptest! {
        #[test]
        fn test_reverse_twice_returns_original(vec: Vec<i32>) {
            let reversed_once = reverse(&vec);
            let reversed_twice = reverse(&reversed_once);
            prop_assert_eq!(vec, reversed_twice);
        }
        
        #[test]
        fn test_reverse_preserves_length(vec: Vec<i32>) {
            let reversed = reverse(&vec);
            prop_assert_eq!(vec.len(), reversed.len());
        }
        
        #[test]
        fn test_factorial_properties(n in 0u32..13) {
            if let Some(fact_n) = factorial(n) {
                if n > 0 {
                    if let Some(fact_n_minus_1) = factorial(n - 1) {
                        prop_assert_eq!(fact_n, fact_n_minus_1 * n as u64);
                    }
                }
            }
        }
    }
}

fn factorial(n: u32) -> Option<u64> {
    match n {
        0 => Some(1),
        n => factorial(n - 1).and_then(|f| f.checked_mul(n as u64)),
    }
}
```

## Test-Driven Development Example

```rust
// src/lib.rs

/// A simple calculator for demonstrating TDD
pub struct Calculator {
    memory: f64,
}

impl Calculator {
    pub fn new() -> Self {
        Calculator { memory: 0.0 }
    }
    
    pub fn add(&mut self, a: f64, b: f64) -> f64 {
        let result = a + b;
        self.memory = result;
        result
    }
    
    pub fn subtract(&mut self, a: f64, b: f64) -> f64 {
        let result = a - b;
        self.memory = result;
        result
    }
    
    pub fn memory_recall(&self) -> f64 {
        self.memory
    }
    
    pub fn memory_clear(&mut self) {
        self.memory = 0.0;
    }
}

#[cfg(test)]
mod calculator_tests {
    use super::*;
    
    #[test]
    fn test_new_calculator_has_zero_memory() {
        let calc = Calculator::new();
        assert_eq!(calc.memory_recall(), 0.0);
    }
    
    #[test]
    fn test_add_stores_result_in_memory() {
        let mut calc = Calculator::new();
        let result = calc.add(5.0, 3.0);
        assert_eq!(result, 8.0);
        assert_eq!(calc.memory_recall(), 8.0);
    }
    
    #[test]
    fn test_subtract_stores_result_in_memory() {
        let mut calc = Calculator::new();
        let result = calc.subtract(10.0, 4.0);
        assert_eq!(result, 6.0);
        assert_eq!(calc.memory_recall(), 6.0);
    }
    
    #[test]
    fn test_memory_clear() {
        let mut calc = Calculator::new();
        calc.add(5.0, 5.0);
        calc.memory_clear();
        assert_eq!(calc.memory_recall(), 0.0);
    }
}
```

## Testing Best Practices

```rust
// src/lib.rs

// 1. Use descriptive test names
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn user_with_valid_email_should_be_created_successfully() {
        let user = User::new("test@example.com", "password123");
        assert!(user.is_ok());
    }
    
    #[test]
    fn user_with_invalid_email_should_return_error() {
        let user = User::new("invalid-email", "password123");
        assert!(user.is_err());
    }
}

// 2. Test edge cases
#[test]
fn test_empty_string_handling() {
    assert_eq!(process_string(""), "");
}

#[test]
fn test_unicode_string_handling() {
    assert_eq!(process_string("🦀"), "🦀");
}

// 3. Use test fixtures
struct TestFixture {
    db: MockDatabase,
    user: User,
}

impl TestFixture {
    fn new() -> Self {
        let db = MockDatabase::new();
        let user = User::new("test@example.com", "password").unwrap();
        TestFixture { db, user }
    }
}

#[test]
fn test_with_fixture() {
    let fixture = TestFixture::new();
    // Use fixture.db and fixture.user
}

// 4. Test error messages
#[test]
fn test_error_message_is_helpful() {
    let result = divide(10.0, 0.0);
    match result {
        Err(msg) => assert!(msg.contains("zero")),
        Ok(_) => panic!("Expected error"),
    }
}

// Placeholder implementations
struct User;
impl User {
    fn new(_email: &str, _password: &str) -> Result<Self, String> {
        Ok(User)
    }
}

struct MockDatabase;
impl MockDatabase {
    fn new() -> Self { MockDatabase }
}

fn process_string(s: &str) -> &str { s }
fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        Err("Cannot divide by zero".to_string())
    } else {
        Ok(a / b)
    }
}
```

## Running Tests

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_name

# Run tests in specific module
cargo test module_name::

# Run only unit tests
cargo test --lib

# Run only integration tests
cargo test --test integration_test

# Run ignored tests
cargo test -- --ignored

# Run tests in release mode
cargo test --release

# Run with specific number of threads
cargo test -- --test-threads=1

# Generate documentation
cargo doc --no-deps --open

# Run documentation tests
cargo test --doc
```

## Exercises

1. **Test Coverage**: Write a module with functions for string manipulation (trim, capitalize, reverse) with 100% test coverage including edge cases.

2. **Mock Implementation**: Create a trait for a database connection and implement both a real and mock version for testing.

3. **Integration Test Suite**: Build an integration test suite for a CLI application that tests argument parsing and file I/O.

4. **Property Testing**: Use property-based testing to verify sorting algorithms maintain correct properties.

5. **Benchmark Suite**: Create benchmarks comparing different implementations of the same algorithm.

## Key Takeaways

- Unit tests go in the same file with `#[cfg(test)]`
- Integration tests go in the `tests/` directory
- Use `assert!`, `assert_eq!`, and `assert_ne!` macros
- Mock dependencies for isolated testing
- Documentation comments with `///` generate docs
- Examples in docs are tested automatically
- Benchmarks require nightly Rust
- Property-based testing finds edge cases
- Good test names describe what they test
- Test both success and failure cases

## Next Steps

In Tutorial 9, we'll explore **Concurrency & Threading**, learning how Rust's ownership system enables fearless concurrency and safe parallel programming.