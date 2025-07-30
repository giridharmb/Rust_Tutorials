# Rust Attributes: The Complete Guide to `#[something]`

A comprehensive guide to understanding and using attributes in Rust - those `#[something]` annotations you see everywhere.

## Table of Contents

- [What Are Attributes?](#what-are-attributes)
- [Basic Syntax](#basic-syntax)
- [Common Built-in Attributes](#common-built-in-attributes)
- [Derive Attributes](#derive-attributes)
- [Conditional Compilation](#conditional-compilation)
- [Function and Method Attributes](#function-and-method-attributes)
- [Struct and Enum Attributes](#struct-and-enum-attributes)
- [Module and Crate Attributes](#module-and-crate-attributes)
- [Testing Attributes](#testing-attributes)
- [Documentation Attributes](#documentation-attributes)
- [Linting Attributes](#linting-attributes)
- [Procedural Macro Attributes](#procedural-macro-attributes)
- [Custom Attributes](#custom-attributes)
- [Advanced Patterns](#advanced-patterns)

## What Are Attributes?

Attributes are metadata applied to modules, crates, items (functions, structs, enums, etc.), or expressions. They tell the compiler or other tools how to process the annotated item.

```rust
#[derive(Debug)]  // This is an attribute
struct Person {
    name: String,
    age: u32,
}

#[test]  // This is also an attribute
fn test_something() {
    assert_eq!(2 + 2, 4);
}
```

## Basic Syntax

### Outer Attributes
Applied to the item that follows:
```rust
#[attribute]
struct MyStruct;

#[attribute(with_args)]
fn my_function() {}

#[attribute = "value"]
mod my_module {}
```

### Inner Attributes
Applied to the enclosing item:
```rust
mod my_module {
    #![attribute]  // Note the ! - applies to my_module
}

fn my_function() {
    #![attribute]  // Applies to my_function
}

// Commonly seen at crate root:
#![allow(dead_code)]
#![feature(async_trait)]
```

### Multiple Attributes
```rust
#[derive(Debug, Clone)]
#[serde(rename_all = "camelCase")]
#[doc = "A person struct"]
struct Person {
    first_name: String,
    last_name: String,
}
```

## Common Built-in Attributes

### 1. `#[derive(...)]` - Auto-implement traits

```rust
// Basic derives
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Point {
    x: i32,
    y: i32,
}

// Common combinations
#[derive(Debug, Clone, Default)]
struct Config {
    host: String,
    port: u16,
}

// Order matters for some derives
#[derive(Eq, PartialEq, Ord, PartialOrd)]
struct Version {
    major: u32,
    minor: u32,
    patch: u32,
}

// What derive actually does:
// The compiler generates implementations like:
impl Debug for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Point")
            .field("x", &self.x)
            .field("y", &self.y)
            .finish()
    }
}
```

### 2. `#[cfg(...)]` - Conditional compilation

```rust
// Platform-specific code
#[cfg(target_os = "windows")]
fn get_home_dir() -> PathBuf {
    PathBuf::from(env::var("USERPROFILE").unwrap())
}

#[cfg(not(target_os = "windows"))]
fn get_home_dir() -> PathBuf {
    PathBuf::from(env::var("HOME").unwrap())
}

// Feature-based compilation
#[cfg(feature = "advanced")]
mod advanced_features {
    // Only compiled when "advanced" feature is enabled
}

// Complex conditions
#[cfg(all(unix, target_pointer_width = "64"))]
fn unix_64_bit_only() {}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn unix_like_systems() {}

// Architecture specific
#[cfg(target_arch = "wasm32")]
fn wasm_specific() {}
```

### 3. `#[allow]`, `#[warn]`, `#[deny]`, `#[forbid]` - Control warnings

```rust
// Allow specific warnings
#[allow(dead_code)]
fn unused_function() {}

#[allow(unused_variables)]
fn example() {
    let x = 5; // Won't warn about unused variable
}

// Deny converts warnings to errors
#[deny(missing_docs)]
pub struct DocumentedStruct {
    /// This field must be documented
    pub field: i32,
}

// Module-wide lint configuration
#[allow(dead_code, unused_imports)]
mod development {
    use std::collections::HashMap;
    
    fn work_in_progress() {}
}

// Common patterns
#[allow(clippy::all)]  // Disable all clippy lints
#[warn(clippy::pedantic)]  // Enable pedantic clippy warnings
```

### 4. `#[must_use]` - Warn if value is ignored

```rust
#[must_use]
fn important_result() -> i32 {
    42
}

#[must_use = "this Result contains an error that should be handled"]
fn fallible_operation() -> Result<(), Error> {
    Ok(())
}

// Common on methods that return Self
struct Builder {
    value: i32,
}

impl Builder {
    #[must_use]
    fn with_value(mut self, value: i32) -> Self {
        self.value = value;
        self
    }
}

// Usage:
important_result(); // Warning: unused return value
let _ = important_result(); // OK: explicitly ignored
```

## Derive Attributes

### Standard Derives

```rust
// Debug - for printing with {:?}
#[derive(Debug)]
struct Point { x: i32, y: i32 }

// Clone - for .clone()
#[derive(Clone)]
struct Data { values: Vec<i32> }

// Copy - for automatic copying (requires Clone)
#[derive(Copy, Clone)]
struct Pixel { r: u8, g: u8, b: u8 }

// PartialEq and Eq - for equality comparison
#[derive(PartialEq, Eq)]
struct Id(u64);

// PartialOrd and Ord - for ordering
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Priority(u8);

// Hash - for use in HashMap/HashSet
#[derive(Hash, PartialEq, Eq)]
struct Key { id: u64, name: String }

// Default - for default values
#[derive(Default)]
struct Config {
    timeout: u64,  // Will be 0
    retries: u32,  // Will be 0
    #[default = "localhost"]
    host: String,  // Requires derive_more crate
}
```

### Third-party Derives (with serde)

```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct Person {
    first_name: String,
    last_name: String,
    #[serde(default)]
    age: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    email: Option<String>,
}

// Complex serde attributes
#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
enum Message {
    #[serde(rename = "text_message")]
    Text { content: String },
    #[serde(rename = "image_message")]
    Image { url: String, #[serde(default)] alt: String },
}
```

## Function and Method Attributes

### Common Function Attributes

```rust
// Inline hints
#[inline]
fn frequently_called() {}

#[inline(always)]
fn definitely_inline() {}

#[inline(never)]
fn never_inline() {}

// Calling conventions
#[no_mangle]
extern "C" fn callable_from_c() {}

// Async-related
#[tokio::main]
async fn main() {
    println!("Async main function");
}

#[async_trait]
trait AsyncProcessor {
    async fn process(&self) -> Result<()>;
}

// Optimization hints
#[cold]
fn rarely_called_error_path() {
    // Compiler optimizes for this being called rarely
}

// Tracking issues
#[deprecated(since = "1.5.0", note = "Use `new_function` instead")]
fn old_function() {}

#[track_caller]
fn function_that_panics() {
    panic!("This panic will show the caller's location");
}
```

### Method-specific Attributes

```rust
struct MyStruct;

impl MyStruct {
    // Constructor patterns
    #[must_use]
    pub fn new() -> Self {
        Self
    }
    
    // Getter patterns
    #[inline]
    #[must_use]
    pub fn value(&self) -> i32 {
        42
    }
    
    // Platform-specific methods
    #[cfg(feature = "experimental")]
    pub fn experimental_feature(&self) {}
}
```

## Struct and Enum Attributes

### Struct Attributes

```rust
// Representation attributes
#[repr(C)]
struct CCompatible {
    x: i32,
    y: i32,
}

#[repr(packed)]
struct Packed {
    a: u8,
    b: u32,
}

#[repr(align(16))]
struct Aligned {
    data: [u8; 16],
}

// Non-exhaustive structs
#[non_exhaustive]
pub struct Config {
    pub setting1: bool,
    pub setting2: i32,
    // Can add fields without breaking compatibility
}

// Derive with custom behavior
#[derive(Debug)]
#[debug(skip)]  // With derivative crate
struct SecureData {
    #[debug = "***"]
    password: String,
    username: String,
}
```

### Enum Attributes

```rust
// Discriminant control
#[repr(u8)]
enum Color {
    Red = 0,
    Green = 1,
    Blue = 2,
}

// FFI-safe enums
#[repr(C)]
enum Status {
    Ok,
    Error,
}

// Non-exhaustive enums
#[non_exhaustive]
pub enum Error {
    Network,
    Parse,
    // Can add variants without breaking compatibility
}

// Complex derive scenarios
#[derive(Debug, Clone)]
enum Message {
    #[deprecated]
    OldFormat(String),
    
    Text {
        content: String,
        #[debug(skip)]
        metadata: Vec<u8>,
    },
}
```

## Module and Crate Attributes

### Crate-level Attributes

```rust
// At the top of main.rs or lib.rs

// Crate metadata
#![crate_name = "my_crate"]
#![crate_type = "lib"]

// Feature gates
#![feature(async_trait)]
#![feature(const_generics)]

// Global lints
#![warn(missing_docs)]
#![deny(unsafe_code)]
#![allow(dead_code)]

// Documentation
#![doc = include_str!("../README.md")]

// Recursion limit for macros
#![recursion_limit = "256"]

// No std
#![no_std]
#![no_main]
```

### Module Attributes

```rust
// Module configuration
#[path = "custom_path.rs"]
mod my_module;

#[cfg(test)]
mod tests {
    #![allow(unused_imports)]
    use super::*;
}

// Controlling visibility
#[doc(hidden)]
pub mod internal {
    // Won't show in docs
}

// Platform-specific modules
#[cfg_attr(windows, path = "windows.rs")]
#[cfg_attr(unix, path = "unix.rs")]
mod platform;
```

## Testing Attributes

### Test Functions

```rust
#[test]
fn basic_test() {
    assert_eq!(2 + 2, 4);
}

#[test]
#[should_panic]
fn test_panic() {
    panic!("This is expected");
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn test_specific_panic() {
    let v = vec![1, 2, 3];
    let _ = v[10]; // Specific panic message
}

#[test]
#[ignore]
fn expensive_test() {
    // Run with cargo test -- --ignored
}

#[test]
#[timeout(Duration::from_secs(5))]  // With test-timeout crate
fn test_with_timeout() {
    // Fails if takes longer than 5 seconds
}
```

### Async Tests

```rust
#[tokio::test]
async fn async_test() {
    let result = async_operation().await;
    assert!(result.is_ok());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn parallel_async_test() {
    // Test with specific runtime configuration
}

#[async_std::test]
async fn async_std_test() {
    // Using async-std instead
}
```

### Benchmark Attributes

```rust
#![feature(test)]
extern crate test;

#[bench]
fn bench_function(b: &mut test::Bencher) {
    b.iter(|| {
        // Code to benchmark
        expensive_operation()
    });
}

// With criterion
#[criterion::benchmark]
fn fibonacci_benchmark(c: &mut Criterion) {
    c.bench_function("fib 20", |b| b.iter(|| fibonacci(20)));
}
```

## Documentation Attributes

### Basic Documentation

```rust
/// This is a doc comment (sugar for #[doc = "..."])
/// 
/// # Examples
/// 
/// ```
/// let x = 5;
/// ```
pub fn documented_function() {}

// Equivalent to:
#[doc = "This is a doc comment"]
#[doc = ""]
#[doc = "# Examples"]
#[doc = ""]
#[doc = "```"]
#[doc = "let x = 5;"]
#[doc = "```"]
pub fn documented_function2() {}
```

### Advanced Documentation

```rust
// Include external files
#[doc = include_str!("../docs/detailed.md")]
pub struct ComplexType;

// Hidden from docs
#[doc(hidden)]
pub fn internal_function() {}

// Feature-gated documentation
#[cfg_attr(feature = "advanced", doc = "This is only available with the `advanced` feature")]
pub fn feature_gated() {}

// Alias for search
#[doc(alias = "create")]
#[doc(alias = "make")]
pub fn new() -> Self {
    Self
}

// Notable traits
#[doc(notable_trait)]
pub trait Important {
    fn important_method(&self);
}
```

## Linting Attributes

### Clippy Lints

```rust
// Allow specific clippy lints
#[allow(clippy::needless_return)]
fn example() -> i32 {
    return 42;
}

// Configure clippy
#![warn(clippy::all)]
#![deny(clippy::correctness)]
#![allow(clippy::style)]

// Clippy configuration for specific items
#[allow(clippy::too_many_arguments)]
fn complex_function(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32, g: i32, h: i32) {}

// Restrict implementations
#[clippy::has_significant_drop]
struct ImportantResource {
    handle: Handle,
}
```

### Rustc Lints

```rust
// Common lints
#![warn(rust_2018_idioms)]
#![deny(missing_debug_implementations)]

// Unsafe code lints
#![forbid(unsafe_code)]
#![warn(unsafe_op_in_unsafe_fn)]

// Future compatibility
#![warn(future_incompatible)]

// Performance lints
#[warn(clippy::inefficient_to_string)]
#[warn(clippy::manual_memcpy)]
```

## Procedural Macro Attributes

### Custom Derive Macros

```rust
// Using derive macros from external crates
#[derive(Builder)]  // From derive_builder crate
struct ServerConfig {
    host: String,
    port: u16,
}

#[derive(From, Display)]  // From derive_more crate
enum Error {
    #[display(fmt = "IO error: {}", _0)]
    Io(std::io::Error),
    #[display(fmt = "Parse error: {}", _0)]
    Parse(String),
}

// Async trait implementations
#[async_trait]
impl Handler for MyHandler {
    async fn handle(&self, req: Request) -> Response {
        // Implementation
    }
}
```

### Attribute Macros

```rust
// Web framework attributes
#[get("/users/{id}")]
async fn get_user(id: web::Path<u64>) -> impl Responder {
    // Handler implementation
}

#[tokio::main]
async fn main() {
    // Transforms into a runtime setup
}

// Custom attribute macros
#[instrument(skip(conn))]
async fn database_query(conn: &Connection, query: &str) -> Result<Vec<Row>> {
    // Adds tracing
}

// Macro attributes with arguments
#[cached(size = 100, time = 60)]
fn expensive_computation(input: i32) -> i32 {
    // Results are cached
}
```

## Custom Attributes

### Creating Custom Attributes

```rust
// In a proc-macro crate
use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn my_attribute(args: TokenStream, input: TokenStream) -> TokenStream {
    // Process the attribute
    input
}

// Usage in another crate
#[my_attribute(option = "value")]
fn decorated_function() {}

// More complex example
#[route(GET, "/api/users")]
fn list_users() -> Json<Vec<User>> {
    // Implementation
}
```

### Framework-specific Attributes

```rust
// Rocket web framework
#[get("/hello/<name>/<age>")]
fn hello(name: String, age: u8) -> String {
    format!("Hello, {} year old named {}!", age, name)
}

// Diesel ORM
#[derive(Queryable, Insertable)]
#[table_name = "users"]
struct User {
    id: i32,
    name: String,
}

// Serde with custom attributes
#[derive(Serialize)]
#[serde(rename_all = "UPPERCASE")]
struct Config {
    #[serde(with = "humantime_serde")]
    timeout: Duration,
}
```

## Advanced Patterns

### Conditional Attributes

```rust
// Apply attributes conditionally
#[cfg_attr(test, derive(PartialEq))]
struct MyStruct {
    field: i32,
}

// Platform-specific attributes
#[cfg_attr(target_os = "windows", windows_subsystem = "windows")]
fn main() {}

// Feature-based derives
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
struct Data {
    value: String,
}
```

### Multiple Attribute Patterns

```rust
// Combining multiple attributes effectively
#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C)]
pub struct CrossPlatformData {
    #[cfg_attr(feature = "serde", serde(rename = "ID"))]
    pub id: u64,
    
    #[cfg(unix)]
    pub unix_specific: i32,
    
    #[cfg(windows)]
    pub windows_specific: i32,
}
```

### Attribute Macros in Practice

```rust
// Database models with multiple attributes
#[derive(Debug, Clone, Queryable, Insertable, AsChangeset)]
#[diesel(table_name = posts)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Post {
    #[diesel(deserialize_as = "i32")]
    pub id: Option<i32>,
    
    pub title: String,
    
    #[diesel(column_name = "body_text")]
    pub body: String,
    
    #[cfg_attr(feature = "serde", serde(with = "chrono::serde::ts_seconds"))]
    pub created_at: NaiveDateTime,
}

// Complex API endpoint
#[instrument(skip(db))]
#[cache(ttl = "60")]
#[validate(auth = "required", role = "admin")]
#[post("/api/v1/posts")]
async fn create_post(
    db: web::Data<DbPool>,
    #[body] post: Json<NewPost>,
    #[header("X-Auth-Token")] token: String,
) -> Result<HttpResponse, ApiError> {
    // Implementation
}
```

## Best Practices

1. **Use built-in derives when possible** - They're well-tested and optimized
2. **Order matters** - Some attributes must come before others
3. **Document your attributes** - Especially custom ones
4. **Use cfg_attr for conditional compilation** - Keeps code clean
5. **Prefer allow over forbid** - More flexible for users
6. **Group related attributes** - Makes code more readable

## Common Pitfalls

```rust
// Wrong: Order matters for some derives
#[derive(PartialOrd, Ord, Eq, PartialEq)]  // Error!
struct Bad;

// Correct: Dependencies first
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Good;

// Wrong: Inner attribute in wrong place
fn bad() {
    #![allow(unused)]  // Error: must be at start
    let x = 5;
}

// Correct: At the beginning
fn good() {
    #![allow(unused)]
    let x = 5;
}
```

## Summary

Attributes are a powerful metaprogramming feature in Rust that:
- Control compilation behavior
- Generate code (derive)
- Configure tools (tests, docs, lints)
- Enable conditional compilation
- Integrate with external tools and frameworks

Understanding attributes is crucial for writing idiomatic Rust and leveraging the ecosystem effectively!