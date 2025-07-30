# Rust Tips and Tricks: From Basic to Advanced

A collection of useful Rust tips, tricks, and patterns that can make your code more elegant, efficient, and idiomatic.

## Table of Contents

- [Basic Tips](#basic-tips)
- [Intermediate Tricks](#intermediate-tricks)
- [Advanced Techniques](#advanced-techniques)
- [Performance Tricks](#performance-tricks)
- [Macro Magic](#macro-magic)
- [Type System Wizardry](#type-system-wizardry)
- [Unsafe Superpowers](#unsafe-superpowers)
- [Dev Productivity](#dev-productivity)

## Basic Tips

### 1. Use `dbg!` macro for quick debugging

```rust
fn calculate_something(x: i32, y: i32) -> i32 {
    let temp = x * 2;
    dbg!(temp); // Prints: [src/main.rs:3] temp = 10
    
    let result = dbg!(temp + y); // Returns the value after printing
    result
}

// Even better - debug multiple values
let (a, b, c) = (1, 2, 3);
dbg!(a, b, c); // Prints all three with their names
```

### 2. Destructuring in function parameters

```rust
// Instead of this:
fn print_point(point: &(f64, f64)) {
    println!("x: {}, y: {}", point.0, point.1);
}

// Do this:
fn print_point(&(x, y): &(f64, f64)) {
    println!("x: {}, y: {}", x, y);
}

// Works with structs too
struct Point { x: f64, y: f64 }

fn distance_from_origin(Point { x, y }: &Point) -> f64 {
    (x * x + y * y).sqrt()
}
```

### 3. Use `if let` and `while let` for cleaner code

```rust
// Instead of:
match some_option {
    Some(value) => println!("{}", value),
    None => {},
}

// Use:
if let Some(value) = some_option {
    println!("{}", value);
}

// Great for loops too
while let Some(item) = iterator.next() {
    process(item);
}

// Combine with else
if let Ok(num) = "42".parse::<i32>() {
    println!("Parsed: {}", num);
} else {
    println!("Failed to parse");
}
```

### 4. The `?` operator works in more places than you think

```rust
// In Option-returning functions
fn get_user_age(user_id: u64) -> Option<u8> {
    let user = get_user(user_id)?;
    let profile = user.profile()?;
    Some(profile.age)
}

// In main function (since Rust 1.26)
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = load_config()?;
    run_app(config)?;
    Ok(())
}

// Custom types that implement Try
use std::ops::Try;
```

### 5. Use `..` to ignore fields

```rust
struct Config {
    host: String,
    port: u16,
    timeout: u64,
    retries: u32,
}

// Only care about some fields
let Config { host, port, .. } = config;

// In match expressions
match message {
    Message::Move { x, y, .. } => println!("Moving to {}, {}", x, y),
    Message::Write(text) => println!("Writing: {}", text),
    _ => {},
}
```

### 6. `todo!()` and `unimplemented!()` for development

```rust
fn complex_algorithm(data: &[u8]) -> Result<String, Error> {
    // Mark work in progress
    todo!("Implement the parsing logic")
}

fn deprecated_function() {
    unimplemented!("This function is no longer supported")
}

// With messages
todo!("Handle error case for user {}", user_id);
```

### 7. Use `_` creatively

```rust
// Ignore unused variables
let _ = expensive_operation(); // Explicitly ignore result

// Type inference helper
let numbers = vec![1, 2, 3];
let sum: i32 = numbers.iter().sum::<_>(); // Let compiler infer the intermediate type

// Partial type annotation
let map = HashMap::<String, _>::new(); // Only specify key type

// In closures
let squares: Vec<_> = (0..10).map(|x| x * x).collect();
```

### 8. Range patterns are powerful

```rust
match age {
    0..=12 => println!("Child"),
    13..=19 => println!("Teenager"),
    20..=65 => println!("Adult"),
    _ => println!("Senior"),
}

// With guards
match (x, y) {
    (0..=100, 0..=100) => println!("In bounds"),
    _ => println!("Out of bounds"),
}

// In array patterns
let arr = [1, 2, 3, 4, 5];
match arr {
    [first, .., last] => println!("First: {}, Last: {}", first, last),
    _ => unreachable!(),
}
```

## Intermediate Tricks

### 9. Use `mem::replace` and `mem::take` for ownership tricks

```rust
use std::mem;

// Swap values without clone
let mut name = String::from("Alice");
let old_name = mem::replace(&mut name, String::from("Bob"));
// name is now "Bob", old_name is "Alice"

// Take ownership from Option/Result without clone
let mut maybe_string = Some(String::from("Hello"));
let string = mem::take(&mut maybe_string).unwrap_or_default();
// maybe_string is now None

// Useful in destructors
struct Container {
    data: Option<ExpensiveResource>,
}

impl Drop for Container {
    fn drop(&mut self) {
        if let Some(resource) = mem::take(&mut self.data) {
            // Do cleanup with owned resource
            resource.cleanup();
        }
    }
}
```

### 10. Builder pattern with phantom types

```rust
use std::marker::PhantomData;

struct EmailBuilder<State> {
    to: Option<String>,
    subject: Option<String>,
    body: Option<String>,
    _state: PhantomData<State>,
}

struct Draft;
struct Ready;

impl EmailBuilder<Draft> {
    fn new() -> Self {
        EmailBuilder {
            to: None,
            subject: None,
            body: None,
            _state: PhantomData,
        }
    }
    
    fn to(mut self, to: String) -> Self {
        self.to = Some(to);
        self
    }
    
    fn subject(mut self, subject: String) -> Self {
        self.subject = Some(subject);
        self
    }
    
    fn body(mut self, body: String) -> EmailBuilder<Ready> {
        EmailBuilder {
            to: self.to,
            subject: self.subject,
            body: Some(body),
            _state: PhantomData,
        }
    }
}

impl EmailBuilder<Ready> {
    fn send(self) -> Result<(), Error> {
        // Can only send when Ready
        Ok(())
    }
}

// Usage
let email = EmailBuilder::new()
    .to("user@example.com".into())
    .subject("Hello".into())
    .body("Content".into()) // Returns EmailBuilder<Ready>
    .send()?; // Only available after body()
```

### 11. Use `Cow` for flexible borrowing

```rust
use std::borrow::Cow;

// Function that might or might not allocate
fn normalize_path(path: &str) -> Cow<str> {
    if path.starts_with("~/") {
        // Need to allocate
        Cow::Owned(path.replacen("~/", "/home/user/", 1))
    } else {
        // Can use the original
        Cow::Borrowed(path)
    }
}

// Avoid allocations in common cases
fn add_prefix<'a>(s: &'a str, needs_prefix: bool) -> Cow<'a, str> {
    if needs_prefix {
        Cow::Owned(format!("PREFIX: {}", s))
    } else {
        Cow::Borrowed(s)
    }
}
```

### 12. Implement `From` instead of `Into`

```rust
// Always implement From
impl From<String> for MyError {
    fn from(s: String) -> Self {
        MyError::Message(s)
    }
}

// You get Into for free!
let error: MyError = "Something went wrong".to_string().into();

// And it works with ?
fn might_fail() -> Result<(), MyError> {
    std::fs::read_to_string("file.txt")?; // std::io::Error converts to MyError
    Ok(())
}
```

### 13. Use extension traits for cleaner APIs

```rust
// Define extension trait
trait VecExt<T> {
    fn swap_remove_if<F>(&mut self, f: F) -> Option<T>
    where
        F: Fn(&T) -> bool;
}

impl<T> VecExt<T> for Vec<T> {
    fn swap_remove_if<F>(&mut self, f: F) -> Option<T>
    where
        F: Fn(&T) -> bool,
    {
        self.iter().position(f).map(|i| self.swap_remove(i))
    }
}

// Now you can use it
let mut numbers = vec![1, 2, 3, 4, 5];
let removed = numbers.swap_remove_if(|&x| x == 3);
```

### 14. Use newtype pattern for type safety

```rust
// Instead of using raw types everywhere
fn calculate_tax(amount: f64, rate: f64) -> f64 { /* ... */ }

// Create semantic types
#[derive(Debug, Clone, Copy)]
struct Dollars(f64);

#[derive(Debug, Clone, Copy)]
struct TaxRate(f64);

impl TaxRate {
    fn new(rate: f64) -> Result<Self, &'static str> {
        if rate >= 0.0 && rate <= 1.0 {
            Ok(TaxRate(rate))
        } else {
            Err("Tax rate must be between 0 and 1")
        }
    }
}

fn calculate_tax(amount: Dollars, rate: TaxRate) -> Dollars {
    Dollars(amount.0 * rate.0)
}
```

### 15. Leverage Deref for smart pointers

```rust
use std::ops::Deref;

struct SmartBuffer {
    data: Vec<u8>,
}

impl Deref for SmartBuffer {
    type Target = [u8];
    
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

// Now you can use slice methods directly
let buffer = SmartBuffer { data: vec![1, 2, 3, 4] };
let first_two = &buffer[..2]; // Works!
let length = buffer.len(); // Also works!
```

## Advanced Techniques

### 16. Const generics for compile-time guarantees

```rust
// Fixed-size matrix with compile-time dimensions
#[derive(Debug)]
struct Matrix<T, const ROWS: usize, const COLS: usize> {
    data: [[T; COLS]; ROWS],
}

impl<T: Default + Copy, const ROWS: usize, const COLS: usize> Matrix<T, ROWS, COLS> {
    fn new() -> Self {
        Matrix {
            data: [[T::default(); COLS]; ROWS],
        }
    }
    
    fn transpose(&self) -> Matrix<T, COLS, ROWS> {
        let mut result = Matrix::<T, COLS, ROWS>::new();
        for i in 0..ROWS {
            for j in 0..COLS {
                result.data[j][i] = self.data[i][j];
            }
        }
        result
    }
}

// Compile-time size checking!
let m1: Matrix<i32, 2, 3> = Matrix::new();
let m2: Matrix<i32, 3, 2> = m1.transpose();
```

### 17. Higher-Ranked Trait Bounds (HRTB)

```rust
// Function that works with any lifetime
fn apply_to_ref<F>(f: F) -> F
where
    F: for<'a> Fn(&'a str) -> &'a str,
{
    f
}

// More practical example
trait Logger {
    fn log(&self, message: &str);
}

fn with_logger<F, R>(logger: &dyn Logger, f: F) -> R
where
    F: for<'a> FnOnce(&'a dyn Logger) -> R,
{
    logger.log("Starting operation");
    let result = f(logger);
    logger.log("Operation complete");
    result
}
```

### 18. Zero-cost state machines with enums

```rust
#[derive(Debug)]
enum ConnectionState {
    Disconnected,
    Connecting { attempt: u32 },
    Connected { session_id: String },
    Disconnecting,
}

impl ConnectionState {
    fn connect(self) -> Result<Self, &'static str> {
        match self {
            ConnectionState::Disconnected => {
                Ok(ConnectionState::Connecting { attempt: 1 })
            }
            ConnectionState::Connecting { attempt } if attempt < 3 => {
                Ok(ConnectionState::Connecting { attempt: attempt + 1 })
            }
            ConnectionState::Connecting { .. } => {
                Ok(ConnectionState::Connected {
                    session_id: uuid::Uuid::new_v4().to_string(),
                })
            }
            _ => Err("Invalid state for connect"),
        }
    }
    
    fn disconnect(self) -> Result<Self, &'static str> {
        match self {
            ConnectionState::Connected { .. } => Ok(ConnectionState::Disconnecting),
            ConnectionState::Disconnecting => Ok(ConnectionState::Disconnected),
            _ => Err("Invalid state for disconnect"),
        }
    }
}
```

### 19. Phantom types for units and constraints

```rust
use std::marker::PhantomData;

// Type-safe units
struct Quantity<T, Unit> {
    value: T,
    _unit: PhantomData<Unit>,
}

struct Meters;
struct Feet;
struct Seconds;

type Distance<T> = Quantity<T, Meters>;
type Time<T> = Quantity<T, Seconds>;

impl<T: Copy> Quantity<T, Meters> {
    fn new(value: T) -> Self {
        Quantity { value, _unit: PhantomData }
    }
    
    fn to_feet(self) -> Quantity<f64, Feet>
    where
        T: Into<f64>,
    {
        Quantity {
            value: self.value.into() * 3.28084,
            _unit: PhantomData,
        }
    }
}

// Compile-time unit checking
fn calculate_speed(distance: Distance<f64>, time: Time<f64>) -> f64 {
    distance.value / time.value
}
```

### 20. Associated types vs generic parameters

```rust
// Use associated types when there's only one logical implementation
trait Container {
    type Item;
    fn get(&self, index: usize) -> Option<&Self::Item>;
}

// Use generic parameters when you need flexibility
trait Converter<T> {
    fn convert(&self, input: &str) -> Result<T, Error>;
}

// Can implement multiple times with different T
impl Converter<i32> for MyParser {
    fn convert(&self, input: &str) -> Result<i32, Error> {
        input.parse().map_err(Into::into)
    }
}

impl Converter<f64> for MyParser {
    fn convert(&self, input: &str) -> Result<f64, Error> {
        input.parse().map_err(Into::into)
    }
}
```

### 21. Custom derive macros with syn and quote

```rust
// In your proc-macro crate
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(Builder)]
pub fn derive_builder(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let builder_name = format_ident!("{}Builder", name);
    
    let expanded = quote! {
        impl #name {
            pub fn builder() -> #builder_name {
                #builder_name::default()
            }
        }
    };
    
    TokenStream::from(expanded)
}
```

## Performance Tricks

### 22. Use `SmallVec` for small collections

```rust
use smallvec::{SmallVec, smallvec};

// Stack-allocated for up to 4 elements
let mut vec: SmallVec<[i32; 4]> = smallvec![1, 2, 3];
vec.push(4); // Still on stack
vec.push(5); // Now heap allocated

// Great for temporary collections
fn process_items(items: &[Item]) -> SmallVec<[ProcessedItem; 8]> {
    items.iter()
        .filter_map(|item| process(item))
        .collect()
}
```

### 23. Avoid allocations with `ArrayVec`

```rust
use arrayvec::ArrayVec;

// Fixed capacity, no heap allocation
let mut array: ArrayVec<i32, 100> = ArrayVec::new();

// Perfect for known maximum sizes
fn parse_numbers(input: &str) -> ArrayVec<i32, 10> {
    input.split(',')
        .filter_map(|s| s.parse().ok())
        .take(10)
        .collect()
}
```

### 24. Use `once_cell` for lazy statics

```rust
use once_cell::sync::Lazy;
use regex::Regex;

// Lazy static with no macros
static EMAIL_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$").unwrap()
});

fn is_valid_email(email: &str) -> bool {
    EMAIL_REGEX.is_match(email)
}

// Thread-local lazy initialization
thread_local! {
    static BUFFER: RefCell<String> = RefCell::new(String::with_capacity(1024));
}
```

### 25. Optimize with `#[inline]` strategically

```rust
// Always inline small functions
#[inline(always)]
fn is_even(n: u32) -> bool {
    n & 1 == 0
}

// Let compiler decide for medium functions
#[inline]
fn calculate_something(x: i32, y: i32) -> i32 {
    let temp = x * 2;
    temp + y * 3
}

// Prevent inlining for large functions
#[inline(never)]
fn complex_algorithm(data: &[u8]) -> Result<String, Error> {
    // Lots of code here
}

// Cross-crate inlining
#[inline]
pub fn public_api_function() {
    // Will be inlined even when used from other crates
}
```

### 26. Use `MaybeUninit` for performance

```rust
use std::mem::MaybeUninit;

// Initialize array without default values
fn create_array() -> [String; 1000] {
    let mut array: [MaybeUninit<String>; 1000] = unsafe {
        MaybeUninit::uninit().assume_init()
    };
    
    for i in 0..1000 {
        array[i] = MaybeUninit::new(format!("Item {}", i));
    }
    
    unsafe { std::mem::transmute::<_, [String; 1000]>(array) }
}

// More idiomatic approach
fn create_array_safe() -> Box<[String; 1000]> {
    let mut array = Box::new([(); 1000].map(|_| MaybeUninit::<String>::uninit()));
    
    for (i, elem) in array.iter_mut().enumerate() {
        elem.write(format!("Item {}", i));
    }
    
    unsafe { Box::from_raw(Box::into_raw(array) as *mut [String; 1000]) }
}
```

## Macro Magic

### 27. Variadic macros with repetition

```rust
macro_rules! create_function {
    ($func_name:ident) => {
        fn $func_name() {
            println!("Called {}", stringify!($func_name));
        }
    };
}

macro_rules! create_functions {
    ($($func_name:ident),*) => {
        $(create_function!($func_name);)*
    };
}

create_functions!(foo, bar, baz);

// Advanced: counting with macros
macro_rules! count_args {
    () => (0usize);
    ($head:expr) => (1usize);
    ($head:expr, $($tail:expr),*) => (1usize + count_args!($($tail),*));
}

let count = count_args!(1, 2, 3, 4); // Returns 4
```

### 28. Macro callbacks and higher-order macros

```rust
macro_rules! with_each {
    ($macro:ident, $($item:expr),*) => {
        $($macro!($item);)*
    };
}

macro_rules! print_item {
    ($item:expr) => {
        println!("Item: {}", $item);
    };
}

// Usage
with_each!(print_item, "first", "second", "third");

// More complex: DSL creation
macro_rules! html {
    (div { $($content:tt)* }) => {
        concat!("<div>", html!($($content)*), "</div>")
    };
    (p { $($content:tt)* }) => {
        concat!("<p>", html!($($content)*), "</p>")
    };
    ($text:expr) => { $text };
}

let html = html!(div {
    p { "Hello, world!" }
});
```

### 29. Declarative macro patterns

```rust
// Pattern matching in macros
macro_rules! match_json {
    ({ $($key:ident : $value:expr),* }) => {
        json::object! {
            $(stringify!($key) => $value),*
        }
    };
    ([ $($element:expr),* ]) => {
        json::array![$($element),*]
    };
}

// TT munching for parsing
macro_rules! parse_kv {
    (@object $obj:ident () () ()) => {};
    
    (@object $obj:ident [$($key:tt)+] ($value:expr) , $($rest:tt)*) => {
        $obj.insert($($key)+, $value);
        parse_kv!(@object $obj () () $($rest)*);
    };
    
    (@object $obj:ident ($($key:tt)*) (: $($rest:tt)*)) => {
        parse_kv!(@object $obj [$($key)*] () $($rest)*);
    };
    
    (@object $obj:ident ($($key:tt)*) ($($rest:tt)*)) => {
        parse_kv!(@object $obj ($($key)*) ($($rest)*));
    };
}
```

## Type System Wizardry

### 30. GATs (Generic Associated Types)

```rust
trait StreamingIterator {
    type Item<'a> where Self: 'a;
    
    fn next(&mut self) -> Option<Self::Item<'_>>;
}

struct WindowIter<'a, T> {
    slice: &'a [T],
    size: usize,
    pos: usize,
}

impl<'a, T> StreamingIterator for WindowIter<'a, T> {
    type Item<'b> = &'b [T] where Self: 'b;
    
    fn next(&mut self) -> Option<Self::Item<'_>> {
        if self.pos + self.size <= self.slice.len() {
            let window = &self.slice[self.pos..self.pos + self.size];
            self.pos += 1;
            Some(window)
        } else {
            None
        }
    }
}
```

### 31. Existential types with `impl Trait`

```rust
// Return type position
fn make_adder(x: i32) -> impl Fn(i32) -> i32 {
    move |y| x + y
}

// Argument position (universal)
fn apply_twice(f: impl Fn(i32) -> i32, x: i32) -> i32 {
    f(f(x))
}

// Storing in structs
struct Container {
    // Can't do this directly:
    // processor: impl Fn(&str) -> String,
    
    // But can do this:
    processor: Box<dyn Fn(&str) -> String>,
}

// Or with generics:
struct Container<F>
where
    F: Fn(&str) -> String,
{
    processor: F,
}
```

### 32. Variance tricks

```rust
use std::marker::PhantomData;

// Covariant wrapper
struct Covariant<'a, T> {
    data: &'a T,
}

// Contravariant wrapper
struct Contravariant<'a, T> {
    func: fn(&'a T),
    _phantom: PhantomData<fn(&'a T)>,
}

// Invariant wrapper
struct Invariant<'a, T> {
    data: &'a mut T,
    _phantom: PhantomData<&'a mut T>,
}

// Use PhantomData to control variance
struct MyType<'a, T> {
    // Make it covariant over 'a
    _phantom: PhantomData<&'a T>,
}
```

## Unsafe Superpowers

### 33. Efficient transmutes

```rust
use std::mem;

// Convert between types of same size
fn bytes_to_u32(bytes: [u8; 4]) -> u32 {
    unsafe { mem::transmute(bytes) }
}

// More complex: reinterpret collections
fn vec_u8_to_u32(mut v: Vec<u8>) -> Vec<u32> {
    assert!(v.len() % 4 == 0);
    let capacity = v.capacity() / 4;
    let len = v.len() / 4;
    let ptr = v.as_mut_ptr() as *mut u32;
    
    mem::forget(v);
    
    unsafe { Vec::from_raw_parts(ptr, len, capacity) }
}

// Safe wrapper using bytemuck
use bytemuck::{cast_slice, cast_vec};

fn safe_conversion(v: Vec<u8>) -> Vec<u32> {
    cast_vec(v) // Compile-time safety!
}
```

### 34. Custom allocators

```rust
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingAllocator {
    inner: System,
    allocated: AtomicUsize,
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.allocated.fetch_add(layout.size(), Ordering::Relaxed);
        self.inner.alloc(layout)
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.allocated.fetch_sub(layout.size(), Ordering::Relaxed);
        self.inner.dealloc(ptr, layout)
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator {
    inner: System,
    allocated: AtomicUsize::new(0),
};
```

### 35. Pin projections

```rust
use std::pin::Pin;
use pin_project::pin_project;

#[pin_project]
struct MyFuture<F> {
    #[pin]
    future: F,
    value: String,
}

impl<F: Future> Future for MyFuture<F> {
    type Output = (F::Output, String);
    
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        match this.future.poll(cx) {
            Poll::Ready(output) => {
                let value = mem::take(this.value);
                Poll::Ready((output, value))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}
```

## Dev Productivity

### 36. Custom test frameworks

```rust
// Custom test attribute
#[test]
#[should_panic(expected = "division by zero")]
fn test_panic() {
    let _ = 1 / 0;
}

// Conditional tests
#[test]
#[cfg(feature = "expensive-tests")]
fn expensive_test() {
    // Only runs with --features expensive-tests
}

// Custom test runner
#![feature(custom_test_frameworks)]
#![test_runner(my_test_runner)]

fn my_test_runner(tests: &[&dyn Fn()]) {
    println!("Running {} tests", tests.len());
    for test in tests {
        test();
    }
}
```

### 37. Cargo tricks

```toml
# Cargo.toml tricks

# Platform-specific dependencies
[target.'cfg(unix)'.dependencies]
nix = "0.23"

[target.'cfg(windows)'.dependencies]
winapi = { version = "0.3", features = ["winuser"] }

# Optional dependencies with features
[dependencies]
serde = { version = "1.0", optional = true }
tokio = { version = "1", features = ["full"], optional = true }

[features]
default = ["async"]
async = ["tokio"]
serialization = ["serde"]

# Dev-only optimizations
[profile.dev]
opt-level = 1  # Slightly optimize dev builds

[profile.dev.package."*"]
opt-level = 3  # Optimize dependencies in dev

# Custom profiles
[profile.profiling]
inherits = "release"
debug = true
```

### 38. Conditional compilation patterns

```rust
// Feature-gated implementations
#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub fn process_data(data: &[Item]) -> Vec<Result> {
    #[cfg(feature = "parallel")]
    {
        data.par_iter().map(process_item).collect()
    }
    
    #[cfg(not(feature = "parallel"))]
    {
        data.iter().map(process_item).collect()
    }
}

// Platform-specific code
pub fn get_system_info() -> SystemInfo {
    #[cfg(target_os = "linux")]
    {
        linux::get_info()
    }
    
    #[cfg(target_os = "windows")]
    {
        windows::get_info()
    }
    
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        generic::get_info()
    }
}
```

### 39. Documentation tricks

```rust
/// # Examples
/// 
/// ```
/// use my_crate::MyStruct;
/// 
/// let s = MyStruct::new();
/// assert_eq!(s.value(), 42);
/// ```
/// 
/// ```should_panic
/// # use my_crate::MyStruct;
/// let s = MyStruct::new();
/// s.panic_method(); // This should panic
/// ```
/// 
/// ```no_run
/// # use my_crate::MyStruct;
/// let s = MyStruct::new();
/// s.expensive_operation(); // Won't run during doc tests
/// ```
pub struct MyStruct;

// Hide implementation details
/// Main documentation here
pub struct PublicApi {
    /// This field is documented
    pub visible: String,
    
    #[doc(hidden)]
    pub _internal: Internal,
}

// Intra-doc links
/// See also [`other_function`] and [`MyStruct`].
/// 
/// [`other_function`]: crate::module::other_function
pub fn my_function() {}
```

### 40. Advanced debugging

```rust
// Custom Debug implementation
use std::fmt;

struct SecretValue(String);

impl fmt::Debug for SecretValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SecretValue(***)")
    }
}

// Conditional debugging
#[derive(Debug)]
struct ComplexStruct {
    #[cfg(debug_assertions)]
    debug_info: String,
    
    data: Vec<u8>,
}

// Pretty printing
#[derive(Debug)]
struct PrettyStruct {
    name: String,
    values: Vec<i32>,
}

impl fmt::Display for PrettyStruct {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "PrettyStruct {{")?;
        writeln!(f, "    name: {:?}", self.name)?;
        write!(f, "    values: [")?;
        for (i, v) in self.values.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{}", v)?;
        }
        writeln!(f, "]")?;
        write!(f, "}}")
    }
}

// Debug helpers
#[allow(dead_code)]
fn debug_size_of<T>() {
    println!("{}: {} bytes", std::any::type_name::<T>(), std::mem::size_of::<T>());
}

// Use like: debug_size_of::<YourType>();
```

## Bonus: Lesser-known std library gems

```rust
// std::mem::discriminant - compare enum variants
use std::mem::discriminant;

enum Message {
    Text(String),
    Number(i32),
}

let m1 = Message::Text("hello".into());
let m2 = Message::Text("world".into());
let m3 = Message::Number(42);

assert_eq!(discriminant(&m1), discriminant(&m2)); // Same variant
assert_ne!(discriminant(&m1), discriminant(&m3)); // Different variants

// std::any::type_name - get type names for debugging
fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>());
}

// std::hint::black_box - prevent optimizations in benchmarks
use std::hint::black_box;

fn benchmark() {
    let result = black_box(expensive_computation());
    // Compiler won't optimize away the computation
}

// std::sync::OnceLock - thread-safe one-time initialization
use std::sync::OnceLock;

static CONFIG: OnceLock<Config> = OnceLock::new();

fn get_config() -> &'static Config {
    CONFIG.get_or_init(|| {
        Config::load().expect("Failed to load config")
    })
}

// std::ops::ControlFlow - early returns in iterations
use std::ops::ControlFlow;

fn find_and_process(items: &[Item]) -> ControlFlow<Error, Success> {
    for item in items {
        match process(item) {
            Ok(result) if result.is_complete() => return ControlFlow::Break(result),
            Err(e) => return ControlFlow::Continue(e),
            Ok(_) => continue,
        }
    }
    ControlFlow::Continue(Error::NotFound)
}
```

## Summary

These tips and tricks showcase Rust's power and flexibility. Remember:

1. **Start simple**: Master the basics before diving into advanced techniques
2. **Measure performance**: Don't optimize prematurely
3. **Prioritize readability**: Clever code isn't always better code
4. **Use the type system**: Let the compiler catch bugs for you
5. **Learn from others**: Read high-quality Rust codebases

The Rust ecosystem is constantly evolving, so keep exploring and learning!