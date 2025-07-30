# Rust Do's and Don'ts: A Comprehensive Guide

Essential practices and anti-patterns in Rust, from beginner to advanced level.

## Table of Contents

- [Basic Level](#basic-level)
  - [Ownership and Borrowing](#ownership-and-borrowing)
  - [Error Handling](#error-handling)
  - [Type System](#type-system)
  - [Memory Management](#memory-management)
- [Intermediate Level](#intermediate-level)
  - [API Design](#api-design)
  - [Performance](#performance)
  - [Concurrency](#concurrency)
  - [Testing](#testing)
- [Advanced Level](#advanced-level)
  - [Unsafe Code](#unsafe-code)
  - [Macro Design](#macro-design)
  - [Async Programming](#async-programming)
  - [Library Design](#library-design)
- [Code Organization](#code-organization)
- [Common Anti-patterns](#common-anti-patterns)

## Basic Level

### Ownership and Borrowing

#### ✅ DO: Use references when you don't need ownership

```rust
// ✅ GOOD: Borrows the string
fn print_length(s: &str) {
    println!("Length: {}", s.len());
}

// ❌ BAD: Takes unnecessary ownership
fn print_length_bad(s: String) {
    println!("Length: {}", s.len());
    // s is dropped here, caller loses the string!
}

// ✅ GOOD: Use &str for string parameters
fn greet(name: &str) {
    println!("Hello, {}!", name);
}

// Works with both String and &str
let owned = String::from("Alice");
greet(&owned);  // Works
greet("Bob");   // Also works
```

#### ❌ DON'T: Clone unnecessarily

```rust
// ❌ BAD: Unnecessary clone
fn process_data(data: &Vec<i32>) -> Vec<i32> {
    let mut copy = data.clone();  // Expensive!
    copy.sort();
    copy
}

// ✅ GOOD: Work with what you need
fn process_data_good(data: &[i32]) -> Vec<i32> {
    let mut copy = data.to_vec();  // Only clone when necessary
    copy.sort();
    copy
}

// ✅ BETTER: Sometimes you don't need to clone at all
fn find_max(data: &[i32]) -> Option<&i32> {
    data.iter().max()
}
```

#### ✅ DO: Use `&mut` only when necessary

```rust
// ❌ BAD: Mutable reference when not needed
fn calculate_sum(numbers: &mut Vec<i32>) -> i32 {
    numbers.iter().sum()  // Not modifying!
}

// ✅ GOOD: Immutable reference
fn calculate_sum_good(numbers: &[i32]) -> i32 {
    numbers.iter().sum()
}

// ✅ GOOD: Use &mut when you actually mutate
fn normalize(numbers: &mut Vec<f64>) {
    let sum: f64 = numbers.iter().sum();
    for n in numbers {
        *n /= sum;
    }
}
```

### Error Handling

#### ✅ DO: Use Result for fallible operations

```rust
use std::fs::File;
use std::io::{self, Read};

// ✅ GOOD: Return Result for operations that can fail
fn read_username_from_file() -> Result<String, io::Error> {
    let mut file = File::open("username.txt")?;
    let mut username = String::new();
    file.read_to_string(&mut username)?;
    Ok(username)
}

// ❌ BAD: Using panic for expected errors
fn read_username_bad() -> String {
    let mut file = File::open("username.txt").unwrap();  // Will panic!
    let mut username = String::new();
    file.read_to_string(&mut username).unwrap();
    username
}
```

#### ❌ DON'T: Use unwrap() in production code

```rust
// ❌ BAD: Unwrap can panic
fn parse_config(input: &str) -> Config {
    let value: i32 = input.parse().unwrap();  // Panic on invalid input!
    Config { value }
}

// ✅ GOOD: Handle errors properly
fn parse_config_good(input: &str) -> Result<Config, ConfigError> {
    let value = input.parse()
        .map_err(|_| ConfigError::InvalidFormat)?;
    Ok(Config { value })
}

// ✅ GOOD: Use expect() with meaningful messages during development
fn development_only() {
    let file = File::open("config.toml")
        .expect("config.toml must exist in development");
}

// ✅ GOOD: Use unwrap_or_default() for safe defaults
fn get_optional_value(map: &HashMap<String, i32>, key: &str) -> i32 {
    map.get(key).copied().unwrap_or_default()
}
```

#### ✅ DO: Create custom error types

```rust
// ✅ GOOD: Custom error types with context
#[derive(Debug, thiserror::Error)]
pub enum DataError {
    #[error("Failed to parse data: {0}")]
    ParseError(String),
    
    #[error("IO error occurred")]
    IoError(#[from] std::io::Error),
    
    #[error("Data validation failed: {reason}")]
    ValidationError { reason: String },
}

// ❌ BAD: Stringly-typed errors
fn process_bad() -> Result<Data, String> {
    Err("Something went wrong".to_string())  // No structure!
}
```

### Type System

#### ✅ DO: Use strong typing

```rust
// ❌ BAD: Using primitives for everything
fn calculate_price(amount: f64, tax_rate: f64) -> f64 {
    amount * (1.0 + tax_rate)
}

// ✅ GOOD: Create meaningful types
#[derive(Debug, Clone, Copy)]
struct Dollars(f64);

#[derive(Debug, Clone, Copy)]
struct TaxRate(f64);

impl TaxRate {
    fn new(rate: f64) -> Result<Self, &'static str> {
        if (0.0..=1.0).contains(&rate) {
            Ok(TaxRate(rate))
        } else {
            Err("Tax rate must be between 0 and 1")
        }
    }
}

fn calculate_price_good(amount: Dollars, tax_rate: TaxRate) -> Dollars {
    Dollars(amount.0 * (1.0 + tax_rate.0))
}
```

#### ❌ DON'T: Overuse `Option<T>` for invalid states

```rust
// ❌ BAD: Using Option for invalid states
struct User {
    id: Option<u64>,  // A user without ID is invalid!
    name: String,
}

// ✅ GOOD: Use builder pattern or separate types
struct NewUser {
    name: String,
}

struct User {
    id: u64,
    name: String,
}

impl NewUser {
    fn save(self) -> Result<User, DbError> {
        let id = db::insert_user(&self.name)?;
        Ok(User { id, name: self.name })
    }
}
```

### Memory Management

#### ✅ DO: Prefer stack allocation

```rust
// ✅ GOOD: Use arrays for fixed-size collections
fn process_small_data() {
    let data = [1, 2, 3, 4, 5];  // Stack allocated
    // Process data...
}

// ❌ BAD: Unnecessary heap allocation
fn process_small_data_bad() {
    let data = vec![1, 2, 3, 4, 5];  // Heap allocated!
}

// ✅ GOOD: Use SmallVec for usually-small collections
use smallvec::{SmallVec, smallvec};

fn collect_items() -> SmallVec<[Item; 8]> {
    // Stack allocated until more than 8 items
    smallvec![item1, item2, item3]
}
```

#### ❌ DON'T: Leak memory

```rust
// ❌ BAD: Creating cycles with Rc
use std::rc::Rc;
use std::cell::RefCell;

struct Node {
    next: Option<Rc<RefCell<Node>>>,
}

// This can create cycles!

// ✅ GOOD: Use Weak references to break cycles
struct NodeGood {
    next: Option<Rc<RefCell<NodeGood>>>,
    parent: Option<Weak<RefCell<NodeGood>>>,  // Weak reference
}

// ✅ GOOD: Or use indices instead of references
struct Graph {
    nodes: Vec<NodeData>,
    edges: Vec<(usize, usize)>,  // Indices into nodes
}
```

## Intermediate Level

### API Design

#### ✅ DO: Accept general types, return specific types

```rust
use std::path::{Path, PathBuf};

// ✅ GOOD: Accept &Path, return PathBuf
fn canonicalize(path: &Path) -> Result<PathBuf, std::io::Error> {
    path.canonicalize()
}

// Can be called with many types:
canonicalize(Path::new("/tmp"));
canonicalize(&PathBuf::from("/tmp"));
canonicalize("/tmp".as_ref());

// ❌ BAD: Too specific parameter type
fn canonicalize_bad(path: PathBuf) -> Result<PathBuf, std::io::Error> {
    path.canonicalize()  // Forces caller to have ownership
}
```

#### ❌ DON'T: Over-genericize

```rust
// ❌ BAD: Overly generic when not needed
fn add<T: std::ops::Add<Output = T>>(a: T, b: T) -> T {
    a + b
}

// ✅ GOOD: Be specific when that's all you need
fn add_numbers(a: i32, b: i32) -> i32 {
    a + b
}

// ✅ GOOD: Generic when it provides value
fn find_min<T: Ord>(slice: &[T]) -> Option<&T> {
    slice.iter().min()
}
```

#### ✅ DO: Use builder pattern for complex types

```rust
// ❌ BAD: Constructor with many parameters
impl Server {
    fn new(
        host: String,
        port: u16,
        timeout: Duration,
        max_connections: usize,
        tls: bool,
        compression: bool,
    ) -> Self {
        // ...
    }
}

// ✅ GOOD: Builder pattern
#[derive(Default)]
struct ServerBuilder {
    host: Option<String>,
    port: Option<u16>,
    timeout: Duration,
    max_connections: usize,
    tls: bool,
    compression: bool,
}

impl ServerBuilder {
    fn new() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            max_connections: 100,
            ..Default::default()
        }
    }
    
    fn host(mut self, host: impl Into<String>) -> Self {
        self.host = Some(host.into());
        self
    }
    
    fn port(mut self, port: u16) -> Self {
        self.port = Some(port);
        self
    }
    
    fn build(self) -> Result<Server, BuildError> {
        Ok(Server {
            host: self.host.ok_or(BuildError::MissingHost)?,
            port: self.port.ok_or(BuildError::MissingPort)?,
            // ...
        })
    }
}

// Usage
let server = ServerBuilder::new()
    .host("localhost")
    .port(8080)
    .build()?;
```

### Performance

#### ✅ DO: Use iterators

```rust
// ❌ BAD: Manual loops with index
fn sum_evens_bad(numbers: &[i32]) -> i32 {
    let mut sum = 0;
    for i in 0..numbers.len() {
        if numbers[i] % 2 == 0 {
            sum += numbers[i];
        }
    }
    sum
}

// ✅ GOOD: Iterator chains
fn sum_evens(numbers: &[i32]) -> i32 {
    numbers.iter()
        .filter(|&&n| n % 2 == 0)
        .sum()
}

// ✅ GOOD: Collect into specific types
fn parse_numbers(input: &str) -> Result<Vec<i32>, ParseIntError> {
    input.split_whitespace()
        .map(|s| s.parse())
        .collect()  // Automatically handles Result!
}
```

#### ❌ DON'T: Allocate in hot loops

```rust
// ❌ BAD: Allocation in loop
fn process_messages_bad(messages: &[Message]) {
    for msg in messages {
        let formatted = format!("Processing: {}", msg.id);  // Allocates!
        log(&formatted);
    }
}

// ✅ GOOD: Reuse allocations
fn process_messages(messages: &[Message]) {
    let mut buffer = String::with_capacity(64);
    for msg in messages {
        buffer.clear();
        write!(&mut buffer, "Processing: {}", msg.id).unwrap();
        log(&buffer);
    }
}

// ✅ GOOD: Pre-allocate collections
fn collect_results(items: &[Item]) -> Vec<Result<Processed, Error>> {
    let mut results = Vec::with_capacity(items.len());
    for item in items {
        results.push(process(item));
    }
    results
}
```

### Concurrency

#### ✅ DO: Prefer channels over shared state

```rust
use std::sync::mpsc;
use std::thread;

// ✅ GOOD: Message passing
fn concurrent_processing() -> Vec<Result<Data, Error>> {
    let (tx, rx) = mpsc::channel();
    
    let handles: Vec<_> = (0..4)
        .map(|id| {
            let tx = tx.clone();
            thread::spawn(move || {
                let result = process_work(id);
                tx.send(result).unwrap();
            })
        })
        .collect();
    
    drop(tx);  // Close channel
    
    rx.into_iter().collect()
}

// ❌ BAD: Excessive shared state
fn concurrent_processing_bad() -> Vec<Result<Data, Error>> {
    let results = Arc::new(Mutex::new(Vec::new()));
    let handles: Vec<_> = (0..4)
        .map(|id| {
            let results = results.clone();
            thread::spawn(move || {
                let result = process_work(id);
                results.lock().unwrap().push(result);  // Lock contention!
            })
        })
        .collect();
    
    // ...
}
```

#### ❌ DON'T: Hold locks across await points

```rust
// ❌ BAD: Holding lock across await
async fn update_cache_bad(cache: Arc<Mutex<Cache>>) {
    let mut cache = cache.lock().unwrap();
    let data = fetch_data().await;  // Lock held during async operation!
    cache.insert(data);
}

// ✅ GOOD: Minimize lock scope
async fn update_cache(cache: Arc<Mutex<Cache>>) {
    let data = fetch_data().await;
    cache.lock().unwrap().insert(data);  // Lock only for insertion
}

// ✅ BETTER: Use async-aware locks
async fn update_cache_async(cache: Arc<tokio::sync::Mutex<Cache>>) {
    let data = fetch_data().await;
    cache.lock().await.insert(data);
}
```

### Testing

#### ✅ DO: Write focused unit tests

```rust
// ✅ GOOD: Test one thing at a time
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn parse_valid_input() {
        let input = "42";
        let result = parse_number(input);
        assert_eq!(result, Ok(42));
    }
    
    #[test]
    fn parse_invalid_input() {
        let input = "not a number";
        let result = parse_number(input);
        assert!(result.is_err());
    }
    
    #[test]
    fn parse_empty_input() {
        let input = "";
        let result = parse_number(input);
        assert!(matches!(result, Err(ParseError::EmptyInput)));
    }
}

// ❌ BAD: Testing too much in one test
#[test]
fn test_everything() {
    let result1 = parse_number("42");
    assert_eq!(result1, Ok(42));
    
    let result2 = parse_number("invalid");
    assert!(result2.is_err());
    
    let result3 = parse_number("");
    assert!(result3.is_err());
    // If this fails, which part broke?
}
```

#### ❌ DON'T: Test implementation details

```rust
// ❌ BAD: Testing private implementation
struct Calculator {
    cache: HashMap<String, f64>,  // Private!
}

#[test]
fn test_cache() {
    let calc = Calculator::new();
    calc.calculate("2 + 2");
    assert!(calc.cache.contains_key("2 + 2"));  // Can't access private field!
}

// ✅ GOOD: Test public behavior
#[test]
fn test_calculation() {
    let mut calc = Calculator::new();
    assert_eq!(calc.calculate("2 + 2"), Ok(4.0));
    // Don't care HOW it works, just that it works
}
```

## Advanced Level

### Unsafe Code

#### ✅ DO: Minimize unsafe blocks

```rust
// ❌ BAD: Large unsafe block
unsafe fn process_raw_data(ptr: *const u8, len: usize) -> Vec<u8> {
    unsafe {
        let slice = std::slice::from_raw_parts(ptr, len);
        let mut result = Vec::new();
        for &byte in slice {
            result.push(byte.wrapping_add(1));
        }
        result
    }
}

// ✅ GOOD: Minimize unsafe scope
fn process_raw_data_good(ptr: *const u8, len: usize) -> Vec<u8> {
    // Validate inputs first
    if ptr.is_null() || len == 0 {
        return Vec::new();
    }
    
    // Only unsafe for what needs it
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
    
    // Safe code for processing
    slice.iter().map(|&b| b.wrapping_add(1)).collect()
}
```

#### ❌ DON'T: Assume invariants without checking

```rust
// ❌ BAD: No validation
pub unsafe fn from_raw_parts_bad<T>(ptr: *mut T, len: usize) -> Vec<T> {
    Vec::from_raw_parts(ptr, len, len)  // What if ptr is null?
}

// ✅ GOOD: Document and validate invariants
/// # Safety
/// 
/// - `ptr` must be valid for reads and writes for `len * mem::size_of::<T>()` bytes
/// - `ptr` must be properly aligned for type `T`
/// - `ptr` must point to `len` initialized elements of type `T`
/// - The memory must have been allocated by the same allocator
pub unsafe fn from_raw_parts_good<T>(ptr: *mut T, len: usize) -> Vec<T> {
    assert!(!ptr.is_null(), "Null pointer passed to from_raw_parts");
    assert!(len <= isize::MAX as usize, "Length overflow");
    
    unsafe { Vec::from_raw_parts(ptr, len, len) }
}
```

### Macro Design

#### ✅ DO: Make macros hygienic

```rust
// ❌ BAD: Unhygienic macro
macro_rules! bad_macro {
    ($x:expr) => {
        let temp = $x;  // 'temp' might conflict!
        temp * 2
    };
}

// ✅ GOOD: Use unique names or blocks
macro_rules! good_macro {
    ($x:expr) => {{
        let __good_macro_temp = $x;
        __good_macro_temp * 2
    }};
}

// ✅ BETTER: Use hygiene from macro_rules 2.0 or proc macros
```

#### ❌ DON'T: Overuse macros

```rust
// ❌ BAD: Macro for simple function
macro_rules! add {
    ($a:expr, $b:expr) => { $a + $b };
}

// ✅ GOOD: Just use a function
fn add(a: i32, b: i32) -> i32 {
    a + b
}

// ✅ GOOD: Macros for compile-time work
macro_rules! create_constants {
    ($($name:ident = $value:expr),*) => {
        $(
            const $name: i32 = $value;
        )*
    };
}

create_constants!(
    MAX_RETRIES = 3,
    TIMEOUT_SECONDS = 30,
    BUFFER_SIZE = 1024
);
```

### Async Programming

#### ✅ DO: Avoid blocking in async code

```rust
// ❌ BAD: Blocking in async
async fn read_file_bad(path: &str) -> Result<String, io::Error> {
    std::fs::read_to_string(path)  // Blocks the thread!
}

// ✅ GOOD: Use async versions
async fn read_file(path: &str) -> Result<String, io::Error> {
    tokio::fs::read_to_string(path).await
}

// ✅ GOOD: Use spawn_blocking for CPU-bound work
async fn expensive_computation(data: Vec<u8>) -> Result<ProcessedData, Error> {
    tokio::task::spawn_blocking(move || {
        // CPU-intensive work here
        process_data_sync(data)
    })
    .await?
}
```

#### ❌ DON'T: Create too many small futures

```rust
// ❌ BAD: Many small futures
async fn process_items_bad(items: Vec<Item>) -> Vec<Result<Processed, Error>> {
    let futures: Vec<_> = items.into_iter()
        .map(|item| async move {
            // Tiny async operation
            validate(&item).await
        })
        .collect();
    
    futures::future::join_all(futures).await
}

// ✅ GOOD: Batch operations
async fn process_items(items: Vec<Item>) -> Vec<Result<Processed, Error>> {
    // Process in reasonable chunks
    let chunks: Vec<_> = items.chunks(100).collect();
    let mut results = Vec::with_capacity(items.len());
    
    for chunk in chunks {
        let batch_results = process_batch(chunk).await;
        results.extend(batch_results);
    }
    
    results
}
```

### Library Design

#### ✅ DO: Make zero-cost abstractions

```rust
// ✅ GOOD: Zero-cost newtype
#[repr(transparent)]
pub struct UserId(u64);

impl UserId {
    #[inline]
    pub fn new(id: u64) -> Self {
        UserId(id)
    }
    
    #[inline]
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

// ❌ BAD: Unnecessary allocations
pub struct UserIdBad {
    id: Box<u64>,  // Why box a u64?
}
```

#### ❌ DON'T: Break backward compatibility

```rust
// ❌ BAD: Changing public API
// Version 1.0
pub fn process(data: &str) -> Result<String, Error> { /* ... */ }

// Version 1.1 - Breaking change!
pub fn process(data: &str, options: Options) -> Result<String, Error> { /* ... */ }

// ✅ GOOD: Maintain compatibility
// Version 1.0
pub fn process(data: &str) -> Result<String, Error> {
    process_with_options(data, Options::default())
}

// Version 1.1 - Add new function
pub fn process_with_options(data: &str, options: Options) -> Result<String, Error> {
    // Implementation
}

// ✅ GOOD: Use non_exhaustive for enums
#[non_exhaustive]
pub enum Error {
    Io(io::Error),
    Parse(ParseError),
    // Can add variants without breaking
}
```

## Code Organization

### ✅ DO: Organize modules logically

```rust
// ✅ GOOD: Clear module structure
// src/lib.rs
pub mod config;
pub mod error;
pub mod models;

mod implementation;  // Private implementation details

pub use error::Error;
pub use models::User;

// src/models/mod.rs
mod user;
mod post;

pub use user::User;
pub use post::Post;

// ❌ BAD: Everything in one file
// src/lib.rs - 5000 lines of code!
```

### ❌ DON'T: Expose implementation details

```rust
// ❌ BAD: Leaking internals
pub struct Database {
    pub connection: SqliteConnection,  // Don't expose!
    pub cache: HashMap<String, String>,
}

// ✅ GOOD: Hide implementation
pub struct Database {
    connection: SqliteConnection,
    cache: HashMap<String, String>,
}

impl Database {
    pub fn new(path: &Path) -> Result<Self, Error> {
        // ...
    }
    
    pub fn query(&self, sql: &str) -> Result<QueryResult, Error> {
        // ...
    }
}
```

## Common Anti-patterns

### String Handling

```rust
// ❌ BAD: Unnecessary String allocations
fn check_prefix_bad(s: String) -> bool {
    s.starts_with("prefix")  // Took ownership for no reason
}

// ✅ GOOD: Use string slices
fn check_prefix(s: &str) -> bool {
    s.starts_with("prefix")
}

// ❌ BAD: format! for simple concatenation
let greeting = format!("{} {}", "Hello", name);  // Overkill

// ✅ GOOD: Use simpler methods when appropriate
let greeting = ["Hello", name].join(" ");
// Or for just two strings:
let greeting = format!("Hello {}", name);
```

### Collection Anti-patterns

```rust
// ❌ BAD: Collecting just to iterate again
let vec: Vec<_> = (0..100).collect();
for i in vec {
    println!("{}", i);
}

// ✅ GOOD: Iterate directly
for i in 0..100 {
    println!("{}", i);
}

// ❌ BAD: Using wrong collection type
fn unique_items_bad(items: Vec<String>) -> Vec<String> {
    let mut seen = Vec::new();
    let mut result = Vec::new();
    
    for item in items {
        if !seen.contains(&item) {  // O(n) lookup!
            seen.push(item.clone());
            result.push(item);
        }
    }
    result
}

// ✅ GOOD: Use appropriate collection
fn unique_items(items: Vec<String>) -> Vec<String> {
    let set: HashSet<_> = items.into_iter().collect();
    set.into_iter().collect()
}
```

### Error Handling Anti-patterns

```rust
// ❌ BAD: Ignoring errors silently
let _ = write_to_file(data);  // Error ignored!

// ✅ GOOD: Handle or propagate
write_to_file(data)?;
// Or at least log:
if let Err(e) = write_to_file(data) {
    log::error!("Failed to write file: {}", e);
}

// ❌ BAD: Stringly-typed errors
fn parse_bad() -> Result<Data, String> {
    Err("Failed to parse".to_string())
}

// ✅ GOOD: Structured errors
#[derive(Debug, thiserror::Error)]
enum ParseError {
    #[error("Invalid format at line {line}")]
    InvalidFormat { line: usize },
    
    #[error("Unexpected token: {0}")]
    UnexpectedToken(String),
}
```

## Summary Checklist

### Basic Level
- [ ] Use borrowing instead of taking ownership when possible
- [ ] Handle errors with Result instead of panicking
- [ ] Create meaningful types instead of primitive obsession
- [ ] Prefer stack allocation for small data

### Intermediate Level
- [ ] Design APIs that are hard to misuse
- [ ] Use iterators instead of manual loops
- [ ] Prefer message passing over shared state
- [ ] Write focused, single-purpose tests

### Advanced Level
- [ ] Document safety invariants for unsafe code
- [ ] Keep macros hygienic and simple
- [ ] Avoid blocking operations in async code
- [ ] Design for backward compatibility

Remember: These guidelines have exceptions. Use your judgment and consider the specific context of your code!