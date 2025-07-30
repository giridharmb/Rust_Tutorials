# Tutorial 1: Getting Started & Basic Syntax

## Installation

### macOS/Linux
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Windows
Download and run: [rustup-init.exe](https://rustup.rs/)

### Verify Installation
```bash
rustc --version
cargo --version
```

## Your First Rust Program

### Create a New Project
```bash
cargo new hello_rust
cd hello_rust
```

### Project Structure
```
hello_rust/
├── Cargo.toml      # Project manifest
└── src/
    └── main.rs     # Entry point
```

### Hello World
```rust
// src/main.rs
fn main() {
    println!("Hello, Rust!");
}
```

### Run the Program
```bash
cargo run
```

## Variables and Mutability

```rust
// src/main.rs
fn main() {
    // Immutable by default
    let x = 5;
    println!("The value of x is: {}", x);
    
    // Mutable variable
    let mut y = 10;
    println!("Original y: {}", y);
    y = 20;
    println!("Modified y: {}", y);
    
    // Constants (must have type annotation)
    const MAX_POINTS: u32 = 100_000;
    println!("Max points: {}", MAX_POINTS);
    
    // Shadowing
    let spaces = "   ";
    let spaces = spaces.len();
    println!("Number of spaces: {}", spaces);
}
```

## Data Types

### Scalar Types
```rust
// src/main.rs
fn main() {
    // Integer types
    let a: i8 = -128;        // 8-bit signed
    let b: u8 = 255;         // 8-bit unsigned
    let c: i32 = -2_147_483_648; // 32-bit signed (default)
    let d: u64 = 18_446_744_073_709_551_615; // 64-bit unsigned
    
    // Floating-point
    let e: f32 = 3.14;
    let f: f64 = 2.718281828; // default
    
    // Boolean
    let g: bool = true;
    let h = false; // type inference
    
    // Character (Unicode)
    let i: char = 'Z';
    let j = '😀';
    
    println!("Integer i8: {}", a);
    println!("Float f64: {}", f);
    println!("Boolean: {}", g);
    println!("Emoji char: {}", j);
}
```

### Compound Types
```rust
// src/main.rs
fn main() {
    // Tuples
    let tup: (i32, f64, u8) = (500, 6.4, 1);
    let (x, y, z) = tup; // destructuring
    println!("The value of y is: {}", y);
    
    // Direct access
    let five_hundred = tup.0;
    let six_point_four = tup.1;
    println!("First element: {}", five_hundred);
    
    // Arrays (fixed size)
    let arr: [i32; 5] = [1, 2, 3, 4, 5];
    let first = arr[0];
    let last = arr[4];
    println!("Array first: {}, last: {}", first, last);
    
    // Array initialization
    let zeros = [0; 10]; // [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    println!("Zeros array length: {}", zeros.len());
}
```

## Functions

```rust
// src/main.rs
fn main() {
    // Calling functions
    greet("Rustacean");
    
    let sum = add(5, 3);
    println!("5 + 3 = {}", sum);
    
    let result = calculate(10, 20);
    println!("Result: {:?}", result);
    
    // Expression vs Statement
    let y = {
        let x = 3;
        x + 1 // no semicolon - this is an expression
    };
    println!("The value of y is: {}", y);
}

// Function with parameters
fn greet(name: &str) {
    println!("Hello, {}!", name);
}

// Function with return value
fn add(a: i32, b: i32) -> i32 {
    a + b // implicit return (no semicolon)
}

// Multiple return values using tuple
fn calculate(x: i32, y: i32) -> (i32, i32, i32) {
    let sum = x + y;
    let diff = x - y;
    let prod = x * y;
    (sum, diff, prod)
}
```

## Control Flow

### If Expressions
```rust
// src/main.rs
fn main() {
    let number = 7;
    
    // Basic if
    if number < 5 {
        println!("Less than 5");
    } else if number == 5 {
        println!("Equal to 5");
    } else {
        println!("Greater than 5");
    }
    
    // If as expression
    let condition = true;
    let value = if condition { 5 } else { 6 };
    println!("Value: {}", value);
}
```

### Loops
```rust
// src/main.rs
fn main() {
    // Infinite loop
    let mut counter = 0;
    let result = loop {
        counter += 1;
        if counter == 10 {
            break counter * 2; // break with value
        }
    };
    println!("Loop result: {}", result);
    
    // While loop
    let mut number = 3;
    while number != 0 {
        println!("{}!", number);
        number -= 1;
    }
    println!("LIFTOFF!!!");
    
    // For loop with range
    for i in 1..4 {
        println!("For loop: {}", i);
    }
    
    // Iterating over array
    let arr = [10, 20, 30, 40, 50];
    for element in arr.iter() {
        println!("Array element: {}", element);
    }
    
    // Reverse range
    for i in (1..4).rev() {
        println!("Countdown: {}", i);
    }
}
```

## Complete Example: Temperature Converter

```rust
// src/main.rs
use std::io;

fn main() {
    println!("Temperature Converter");
    println!("====================");
    
    loop {
        println!("\nChoose conversion:");
        println!("1. Celsius to Fahrenheit");
        println!("2. Fahrenheit to Celsius");
        println!("3. Exit");
        
        let mut choice = String::new();
        io::stdin()
            .read_line(&mut choice)
            .expect("Failed to read line");
        
        let choice: u32 = match choice.trim().parse() {
            Ok(num) => num,
            Err(_) => {
                println!("Invalid input. Please enter a number.");
                continue;
            }
        };
        
        match choice {
            1 => convert_c_to_f(),
            2 => convert_f_to_c(),
            3 => {
                println!("Goodbye!");
                break;
            }
            _ => println!("Invalid choice. Please select 1, 2, or 3."),
        }
    }
}

fn convert_c_to_f() {
    let celsius = get_temperature("Enter temperature in Celsius:");
    let fahrenheit = celsius * 9.0 / 5.0 + 32.0;
    println!("{}°C = {:.2}°F", celsius, fahrenheit);
}

fn convert_f_to_c() {
    let fahrenheit = get_temperature("Enter temperature in Fahrenheit:");
    let celsius = (fahrenheit - 32.0) * 5.0 / 9.0;
    println!("{}°F = {:.2}°C", fahrenheit, celsius);
}

fn get_temperature(prompt: &str) -> f64 {
    loop {
        println!("{}", prompt);
        
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line");
        
        match input.trim().parse::<f64>() {
            Ok(temp) => return temp,
            Err(_) => println!("Invalid temperature. Please enter a number."),
        }
    }
}
```

## String Types

```rust
// src/main.rs
fn main() {
    // String literal (&str)
    let s1: &str = "Hello, world!";
    println!("String literal: {}", s1);
    
    // String (owned, growable)
    let mut s2 = String::new();
    s2.push_str("Hello");
    s2.push(' ');
    s2.push_str("Rust!");
    println!("Mutable String: {}", s2);
    
    // From literal to String
    let s3 = String::from("Initial value");
    let s4 = "Another way".to_string();
    
    // String concatenation
    let s5 = s3 + " " + &s4; // s3 is moved here
    println!("Concatenated: {}", s5);
    
    // Format macro (doesn't take ownership)
    let s6 = String::from("First");
    let s7 = String::from("Second");
    let s8 = format!("{} and {}", s6, s7);
    println!("Formatted: {}", s8);
    println!("s6 still valid: {}", s6);
    
    // String slicing
    let hello = "Здравствуйте";
    let s = &hello[0..4]; // Be careful with UTF-8!
    println!("Slice: {}", s);
}
```

## Exercises

1. **Variable Challenge**: Create variables of each scalar type and print them with descriptive messages.

2. **Function Practice**: Write a function that takes two numbers and returns their average as a floating-point number.

3. **Loop Exercise**: Use a for loop to calculate the factorial of a number.

4. **String Manipulation**: Create a program that takes a user's name and prints a personalized greeting.

## Running the Examples

To run any of these examples:

1. Replace the contents of `src/main.rs` with the code
2. Run `cargo run` in your terminal
3. For examples with user input, follow the prompts

## Key Takeaways

- Variables are immutable by default (use `mut` for mutability)
- Rust is statically typed but has type inference
- Functions use snake_case naming convention
- Expressions return values, statements do not
- Control flow includes `if`, `loop`, `while`, and `for`
- Two main string types: `&str` (borrowed) and `String` (owned)

## Next Steps

In the next tutorial, we'll dive into Rust's most unique feature: **Ownership and Borrowing**. This is crucial for understanding how Rust achieves memory safety without a garbage collector.