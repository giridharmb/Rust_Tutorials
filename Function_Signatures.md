# Rust Function Signatures: A Comprehensive Guide

A detailed guide to understanding Rust function signatures, from basic to advanced patterns.

## Table of Contents

- [Basic Function Signatures](#basic-function-signatures)
- [Lifetime Parameters](#lifetime-parameters)
- [Generic Functions](#generic-functions)
- [Trait Bounds](#trait-bounds)
- [Higher-Order Functions](#higher-order-functions)
- [Async Functions](#async-functions)
- [Associated Functions and Methods](#associated-functions-and-methods)
- [Complex Real-World Examples](#complex-real-world-examples)
- [Common Confusing Patterns](#common-confusing-patterns)

## Basic Function Signatures

### Simple Function
```rust
fn add(x: i32, y: i32) -> i32 {
    x + y
}
```
**What it means:**
- Takes two `i32` parameters
- Returns an `i32`
- No borrowing, ownership is moved

### Function with No Return Value
```rust
fn print_message(msg: &str) {
    println!("{}", msg);
}
```
**What it means:**
- Takes a string slice reference
- Returns `()` (unit type) implicitly
- Borrows the string, doesn't take ownership

### Function with References
```rust
fn get_length(s: &String) -> usize {
    s.len()
}
```
**What it means:**
- Takes an immutable reference to a String
- Returns a `usize`
- Can read but not modify the input

### Function with Mutable References
```rust
fn append_world(s: &mut String) {
    s.push_str(" world");
}
```
**What it means:**
- Takes a mutable reference to a String
- Can modify the original String
- Returns nothing (unit type)

## Lifetime Parameters

### Single Lifetime
```rust
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
```
**What it means:**
- `'a` is a lifetime parameter
- Both inputs and output have the same lifetime
- The returned reference will be valid as long as both inputs are valid

### Multiple Lifetimes
```rust
fn first_word<'a, 'b>(s: &'a str, prefix: &'b str) -> &'a str {
    if s.starts_with(prefix) {
        &s[..prefix.len()]
    } else {
        s
    }
}
```
**What it means:**
- Two different lifetimes: `'a` and `'b`
- Output lifetime tied only to first parameter
- `prefix` can have a different (shorter) lifetime

### Lifetime Elision
```rust
// These two are equivalent:
fn get_first(s: &str) -> &str { s }
fn get_first<'a>(s: &'a str) -> &'a str { s }
```
**What it means:**
- Rust can infer simple lifetime relationships
- Single input reference → output gets same lifetime

## Generic Functions

### Simple Generic
```rust
fn identity<T>(value: T) -> T {
    value
}
```
**What it means:**
- Works with any type `T`
- Takes ownership of value and returns it
- Type is determined at compile time

### Multiple Generic Parameters
```rust
fn pair<T, U>(first: T, second: U) -> (T, U) {
    (first, second)
}
```
**What it means:**
- Two different generic types
- Can be different types or the same
- Returns a tuple of both values

### Generic with Constraints
```rust
fn largest<T: PartialOrd>(list: &[T]) -> &T {
    let mut largest = &list[0];
    for item in list {
        if item > largest {
            largest = item;
        }
    }
    largest
}
```
**What it means:**
- `T` must implement `PartialOrd` trait
- Can only be used with types that support comparison
- Returns a reference to an element in the slice

## Trait Bounds

### Single Trait Bound
```rust
fn print_debug<T: std::fmt::Debug>(value: &T) {
    println!("{:?}", value);
}
```
**What it means:**
- `T` must implement the `Debug` trait
- Can print any debuggable type

### Multiple Trait Bounds
```rust
fn clone_and_compare<T: Clone + PartialOrd>(x: &T, y: &T) -> T {
    if x > y {
        x.clone()
    } else {
        y.clone()
    }
}
```
**What it means:**
- `T` must implement both `Clone` AND `PartialOrd`
- Can clone and compare values

### Where Clauses
```rust
fn complex_function<T, U>(t: T, u: U) -> (T, U)
where
    T: Clone + Debug,
    U: Clone + Debug + PartialOrd,
{
    println!("{:?} {:?}", t, u);
    (t.clone(), u.clone())
}
```
**What it means:**
- Same as inline bounds but more readable
- Useful for complex constraints
- Each type can have multiple bounds

### Trait Bound with Lifetime
```rust
fn find_first<'a, T, P>(slice: &'a [T], predicate: P) -> Option<&'a T>
where
    P: Fn(&T) -> bool,
{
    slice.iter().find(|&&item| predicate(&item))
}
```
**What it means:**
- Combines lifetimes and trait bounds
- `P` is a function that takes `&T` and returns `bool`
- Returns an optional reference with lifetime `'a`

## Higher-Order Functions

### Function as Parameter
```rust
fn apply_twice<F>(f: F, x: i32) -> i32
where
    F: Fn(i32) -> i32,
{
    f(f(x))
}
```
**What it means:**
- Takes a function `F` as parameter
- `F` must take and return `i32`
- Applies the function twice

### Returning Closures
```rust
fn make_adder(x: i32) -> impl Fn(i32) -> i32 {
    move |y| x + y
}
```
**What it means:**
- Returns a closure that captures `x`
- `impl Trait` means "some type that implements this trait"
- The closure takes an `i32` and returns an `i32`

### Complex Closure Bounds
```rust
fn filter_map<T, U, F, P>(items: Vec<T>, predicate: P, mapper: F) -> Vec<U>
where
    P: Fn(&T) -> bool,
    F: Fn(T) -> U,
{
    items.into_iter()
        .filter(|item| predicate(item))
        .map(mapper)
        .collect()
}
```
**What it means:**
- Two function parameters with different signatures
- `predicate` borrows items, `mapper` consumes them
- Filters then transforms elements

## Async Functions

### Simple Async
```rust
async fn fetch_data(url: &str) -> Result<String, Error> {
    // async operations
    Ok(String::from("data"))
}
```
**What it means:**
- Returns a `Future` that yields `Result<String, Error>`
- Must be awaited to get the actual result
- Runs asynchronously

### Async with Lifetime
```rust
async fn process_borrowed<'a>(data: &'a str) -> &'a str {
    tokio::time::sleep(Duration::from_millis(100)).await;
    data
}
```
**What it means:**
- Async function that borrows data
- Lifetime must be explicitly specified
- Returns the same borrowed data

### Async Trait Methods
```rust
use async_trait::async_trait;

#[async_trait]
trait DataProcessor {
    async fn process(&self, data: &str) -> Result<String, Error>;
}
```
**What it means:**
- Trait with async methods (requires async-trait crate)
- Implementors must provide async implementations

## Associated Functions and Methods

### Associated Function (No self)
```rust
impl Rectangle {
    fn new(width: f64, height: f64) -> Self {
        Self { width, height }
    }
}
```
**What it means:**
- Called on the type: `Rectangle::new(10.0, 20.0)`
- Often used as constructors
- No `self` parameter

### Method with Immutable Self
```rust
impl Rectangle {
    fn area(&self) -> f64 {
        self.width * self.height
    }
}
```
**What it means:**
- Called on instance: `rect.area()`
- Borrows self immutably
- Can read but not modify fields

### Method with Mutable Self
```rust
impl Rectangle {
    fn double_size(&mut self) {
        self.width *= 2.0;
        self.height *= 2.0;
    }
}
```
**What it means:**
- Requires mutable reference to call
- Can modify the instance

### Method Consuming Self
```rust
impl Rectangle {
    fn into_square(self) -> Square {
        Square { side: self.width.max(self.height) }
    }
}
```
**What it means:**
- Takes ownership of self
- Original instance is consumed/moved
- Often used for transformations

## Complex Real-World Examples

### Database Connection Pool
```rust
pub fn with_connection<T, F, E>(pool: &Pool, f: F) -> Result<T, E>
where
    F: FnOnce(&mut Connection) -> Result<T, E>,
    E: From<PoolError>,
{
    let mut conn = pool.get()?;
    f(&mut conn)
}
```
**What it means:**
- Generic over return type `T` and error type `E`
- Takes a closure that receives a mutable connection
- `FnOnce` means the closure can only be called once
- Error type must be convertible from `PoolError`

### Iterator Adapter
```rust
fn chunk_by<T, K, F>(items: Vec<T>, key_fn: F) -> HashMap<K, Vec<T>>
where
    T: Clone,
    K: Eq + Hash,
    F: Fn(&T) -> K,
{
    let mut groups = HashMap::new();
    for item in items {
        let key = key_fn(&item);
        groups.entry(key).or_insert_with(Vec::new).push(item);
    }
    groups
}
```
**What it means:**
- Groups items by a key extracted via `key_fn`
- `K` must be hashable and comparable for HashMap
- Items must be cloneable
- Function borrows items to extract keys

### Async Stream Processor
```rust
use futures::Stream;

fn process_stream<S, T, E, F, Fut>(
    stream: S,
    processor: F,
) -> impl Stream<Item = Result<T, E>>
where
    S: Stream<Item = String> + Send + 'static,
    F: Fn(String) -> Fut + Send + Sync + Clone + 'static,
    Fut: Future<Output = Result<T, E>> + Send,
    T: Send + 'static,
    E: Send + 'static,
{
    stream.then(move |item| processor(item))
}
```
**What it means:**
- Transforms a stream of strings into processed results
- `processor` is an async function
- All types must be `Send` for thread safety
- `'static` means no borrowed data (required for async)

### Builder Pattern with Phantom Data
```rust
use std::marker::PhantomData;

struct Builder<State> {
    name: Option<String>,
    age: Option<u32>,
    _phantom: PhantomData<State>,
}

struct Incomplete;
struct Complete;

impl Builder<Incomplete> {
    fn new() -> Self {
        Builder {
            name: None,
            age: None,
            _phantom: PhantomData,
        }
    }
    
    fn with_name(mut self, name: String) -> Builder<Complete> {
        self.name = Some(name);
        Builder {
            name: self.name,
            age: self.age,
            _phantom: PhantomData,
        }
    }
}

impl Builder<Complete> {
    fn build(self) -> Person {
        Person {
            name: self.name.unwrap(),
            age: self.age.unwrap_or(0),
        }
    }
}
```
**What it means:**
- Type-state pattern using phantom types
- Compile-time guarantee of builder completeness
- Different methods available in different states

## Common Confusing Patterns

### impl Trait vs dyn Trait
```rust
// Static dispatch (compile-time)
fn static_dispatch(x: impl Display) -> impl Debug {
    format!("{}", x)
}

// Dynamic dispatch (runtime)
fn dynamic_dispatch(x: &dyn Display) -> Box<dyn Debug> {
    Box::new(format!("{}", x))
}
```
**What it means:**
- `impl Trait`: Concrete type known at compile time, zero cost
- `dyn Trait`: Type erased, requires pointer indirection

### Function Pointer Types
```rust
// Function pointer
fn apply_fn(f: fn(i32) -> i32, x: i32) -> i32 {
    f(x)
}

// Fn trait (more flexible)
fn apply_closure<F: Fn(i32) -> i32>(f: F, x: i32) -> i32 {
    f(x)
}
```
**What it means:**
- `fn()`: Only function pointers, no closures with captures
- `Fn()`: Any callable, including closures

### Self Types in Traits
```rust
trait Cloneable {
    fn clone(&self) -> Self;
}

trait Iterator {
    type Item;
    fn next(&mut self) -> Option<Self::Item>;
}
```
**What it means:**
- `Self`: The implementing type
- `Self::Item`: Associated type defined by implementor

### Higher-Ranked Trait Bounds (HRTB)
```rust
fn higher_ranked<F>(f: F)
where
    F: for<'a> Fn(&'a str) -> &'a str,
{
    let result = f("hello");
    println!("{}", result);
}
```
**What it means:**
- `for<'a>`: Works for ANY lifetime
- More flexible than specific lifetime parameters
- Common with closure parameters

### Const Generics
```rust
fn create_array<T: Default, const N: usize>() -> [T; N] {
    [T::default(); N]
}

// Usage
let arr: [i32; 5] = create_array();
```
**What it means:**
- `const N: usize`: Compile-time constant parameter
- Can use in array sizes and other const contexts

### GATs (Generic Associated Types)
```rust
trait Container {
    type Item<'a> where Self: 'a;
    
    fn get<'a>(&'a self, index: usize) -> Option<Self::Item<'a>>;
}
```
**What it means:**
- Associated types that are themselves generic
- Allows more flexible trait definitions
- Useful for lending iterators and similar patterns

## Quick Reference: Reading Complex Signatures

When you see a complex signature like:
```rust
pub fn map_async<'a, T, U, F, Fut>(
    items: &'a [T],
    f: F,
) -> impl Future<Output = Vec<U>> + 'a
where
    T: Send + Sync + 'a,
    U: Send + 'static,
    F: Fn(&'a T) -> Fut + Send + Sync + Clone + 'a,
    Fut: Future<Output = U> + Send + 'a,
```

**How to read it:**
1. **Function name and generics**: `map_async` with lifetime `'a` and types `T`, `U`, `F`, `Fut`
2. **Parameters**: Takes a slice of `T` and a function `F`
3. **Return type**: Returns a future that yields `Vec<U>`
4. **Constraints**:
   - `T`: Must be thread-safe (`Send + Sync`) and live at least `'a`
   - `U`: Must be sendable and have static lifetime
   - `F`: A function that takes `&T` and returns a `Fut`
   - `Fut`: A future that yields `U`

## Advanced Patterns

### Pin and Unpin Traits

```rust
use std::pin::Pin;
use std::future::Future;

// Pinned future that can't be moved
fn process_pinned<F>(future: Pin<&mut F>) -> Pin<&mut F>
where
    F: Future<Output = ()>,
{
    future
}

// Self-referential struct requiring Pin
struct SelfReferential {
    data: String,
    ptr: *const String,
}

impl SelfReferential {
    fn new(data: String) -> Pin<Box<Self>> {
        let mut boxed = Box::new(Self {
            data,
            ptr: std::ptr::null(),
        });
        let ptr = &boxed.data as *const String;
        boxed.ptr = ptr;
        Box::into_pin(boxed)
    }
    
    fn get_data(self: Pin<&Self>) -> &str {
        &self.data
    }
}

// Async function with pinned self
impl Stream for MyStream {
    type Item = i32;
    
    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        // Implementation
        Poll::Ready(Some(42))
    }
}
```
**What it means:**
- `Pin<P>` prevents the pointed-to value from moving in memory
- Required for self-referential structs and async/await
- `Unpin` trait marks types that are safe to move even when pinned
- Most types implement `Unpin` automatically

### Variance in Lifetimes

```rust
// Covariant lifetime (most common)
struct Covariant<'a> {
    data: &'a str,
}

// Contravariant lifetime (in function parameters)
struct Contravariant<'a> {
    func: fn(&'a str),
}

// Invariant lifetime (with &mut)
struct Invariant<'a> {
    data: &'a mut String,
}

// Variance in practice
fn covariant_example<'a, 'b: 'a>(longer: &'b str) -> Covariant<'a> {
    // 'b outlives 'a, so we can use it where 'a is expected
    Covariant { data: longer }
}

// Won't compile - invariant lifetime
// fn invariant_bad<'a, 'b: 'a>(longer: &'b mut String) -> Invariant<'a> {
//     Invariant { data: longer } // Error!
// }

// Subtyping with lifetimes
fn demonstrate_variance<'a>(x: &'a str) {
    // This works because &'static str is a subtype of &'a str
    let static_str: &'static str = "hello";
    let shorter_lifetime: &'a str = static_str;
}
```
**What it means:**
- **Covariant**: Can use a longer lifetime where shorter is expected
- **Contravariant**: Can use a shorter lifetime where longer is expected
- **Invariant**: Must use exact lifetime, no substitution
- `&T` is covariant over `T` and `'a`
- `&mut T` is invariant over `T` but covariant over `'a`
- Function parameters are contravariant

### Unsafe Function Signatures

```rust
// Basic unsafe function
unsafe fn raw_memory_access(ptr: *const u8, len: usize) -> &'static [u8] {
    std::slice::from_raw_parts(ptr, len)
}

// Unsafe trait
unsafe trait Zeroable {
    // Implementor guarantees all-zero bytes is valid
}

unsafe impl Zeroable for i32 {}
unsafe impl Zeroable for f64 {}

// Function requiring unsafe trait
fn create_zeroed<T: Zeroable>() -> T {
    unsafe { std::mem::zeroed() }
}

// Unsafe function pointer
type UnsafeFn = unsafe fn(*mut c_void) -> i32;

// Safe wrapper around unsafe operation
fn safe_wrapper<T>(slice: &mut [T], index: usize) -> Option<&mut T> {
    if index < slice.len() {
        // SAFETY: We checked bounds above
        Some(unsafe { slice.get_unchecked_mut(index) })
    } else {
        None
    }
}

// Unsafe async function
async unsafe fn async_unsafe_operation(ptr: *mut u8) {
    // SAFETY: Caller must ensure ptr is valid
    unsafe { *ptr = 42; }
}
```
**What it means:**
- `unsafe fn`: Caller must uphold safety invariants
- `unsafe trait`: Implementor must guarantee safety requirements
- Often used for FFI, raw memory access, or performance optimizations
- Safety comments (`// SAFETY:`) document invariants

### Macro-generated Functions

```rust
// Declarative macro generating functions
macro_rules! make_getter_setter {
    ($field:ident, $type:ty) => {
        paste::paste! {
            pub fn [<get_ $field>](&self) -> &$type {
                &self.$field
            }
            
            pub fn [<set_ $field>](&mut self, value: $type) {
                self.$field = value;
            }
        }
    };
}

struct Person {
    name: String,
    age: u32,
}

impl Person {
    make_getter_setter!(name, String);
    make_getter_setter!(age, u32);
}

// Procedural macro attributes (what they expand to)
// #[derive(Debug)] expands to:
impl std::fmt::Debug for Person {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Person")
            .field("name", &self.name)
            .field("age", &self.age)
            .finish()
    }
}

// Macro-generated async functions
macro_rules! async_endpoint {
    ($name:ident, $path:expr, $ret:ty) => {
        pub async fn $name(
            client: &Client,
            params: &HashMap<String, String>,
        ) -> Result<$ret, Error> {
            client.get($path).query(params).send().await?.json().await
        }
    };
}

// Generates multiple endpoints
async_endpoint!(get_user, "/api/user", User);
async_endpoint!(get_posts, "/api/posts", Vec<Post>);
```
**What it means:**
- Macros can generate complex function signatures
- Look for `!` to identify macro invocations
- Procedural macros can generate entire impl blocks
- IDE expansion helps understand generated code

### FFI Function Signatures

```rust
use std::os::raw::{c_char, c_int, c_void};
use std::ffi::{CStr, CString};

// External C function
extern "C" {
    fn strlen(s: *const c_char) -> usize;
    fn malloc(size: usize) -> *mut c_void;
    fn free(ptr: *mut c_void);
}

// Rust function exported to C
#[no_mangle]
pub extern "C" fn rust_function(x: c_int, y: c_int) -> c_int {
    x + y
}

// Callback function type
type Callback = unsafe extern "C" fn(data: *mut c_void, len: usize);

// Function accepting C callback
pub fn register_callback(cb: Callback, user_data: *mut c_void) {
    unsafe { cb(user_data, 42) };
}

// Opaque pointer pattern
#[repr(C)]
pub struct OpaqueHandle {
    _private: [u8; 0],
}

extern "C" {
    fn create_handle() -> *mut OpaqueHandle;
    fn destroy_handle(handle: *mut OpaqueHandle);
    fn use_handle(handle: *const OpaqueHandle, value: c_int) -> c_int;
}

// Safe wrapper around FFI
pub struct SafeHandle {
    raw: *mut OpaqueHandle,
}

impl SafeHandle {
    pub fn new() -> Option<Self> {
        let raw = unsafe { create_handle() };
        if raw.is_null() {
            None
        } else {
            Some(SafeHandle { raw })
        }
    }
    
    pub fn process(&self, value: i32) -> i32 {
        unsafe { use_handle(self.raw, value as c_int) as i32 }
    }
}

impl Drop for SafeHandle {
    fn drop(&mut self) {
        unsafe { destroy_handle(self.raw) };
    }
}

// Complex FFI with strings
#[no_mangle]
pub unsafe extern "C" fn process_string(
    input: *const c_char,
    output: *mut c_char,
    output_len: usize,
) -> c_int {
    if input.is_null() || output.is_null() {
        return -1;
    }
    
    let input_str = match CStr::from_ptr(input).to_str() {
        Ok(s) => s,
        Err(_) => return -2,
    };
    
    let processed = format!("Processed: {}", input_str);
    let processed_cstr = match CString::new(processed) {
        Ok(s) => s,
        Err(_) => return -3,
    };
    
    let bytes = processed_cstr.as_bytes_with_nul();
    if bytes.len() > output_len {
        return -4;
    }
    
    std::ptr::copy_nonoverlapping(
        bytes.as_ptr(),
        output as *mut u8,
        bytes.len(),
    );
    
    0 // Success
}
```
**What it means:**
- `extern "C"`: C calling convention for FFI
- `#[no_mangle]`: Prevents name mangling for C compatibility
- Raw pointers (`*const`, `*mut`) for C interop
- `#[repr(C)]`: C-compatible memory layout
- Always wrap unsafe FFI in safe Rust APIs

### Combining It All: Complex Real-World Example

```rust
use std::marker::PhantomData;
use std::pin::Pin;
use std::future::Future;

// Complex async trait with associated types, lifetimes, and unsafe
pub trait AsyncProcessor<'a>: Send + Sync + 'a {
    type Input: Send + 'a;
    type Output: Send + 'static;
    type Error: std::error::Error + Send + Sync + 'static;
    
    unsafe fn process_raw(
        self: Pin<&mut Self>,
        input: *const Self::Input,
        ctx: &mut Context<'a>,
    ) -> Poll<Result<*mut Self::Output, Self::Error>>;
    
    fn process<'b>(
        self: Pin<&'b mut Self>,
        input: &'b Self::Input,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send + 'b
    where
        'a: 'b,
    {
        async move {
            // Safe wrapper around unsafe method
            let mut ctx = Context::new();
            let output_ptr = unsafe {
                match self.process_raw(input as *const _, &mut ctx).await {
                    Poll::Ready(Ok(ptr)) => ptr,
                    Poll::Ready(Err(e)) => return Err(e),
                    Poll::Pending => unreachable!(),
                }
            };
            
            // SAFETY: process_raw guarantees valid pointer
            let output = unsafe { Box::from_raw(output_ptr) };
            Ok(*output)
        }
    }
}

// FFI wrapper with complex generics
#[repr(C)]
pub struct FfiWrapper<T, const N: usize>
where
    T: Zeroable + Send + Sync,
{
    data: [T; N],
    vtable: *const VTable<T>,
    _marker: PhantomData<&'static T>,
}

#[repr(C)]
struct VTable<T> {
    drop: unsafe extern "C" fn(*mut T),
    clone: unsafe extern "C" fn(*const T) -> *mut T,
    process: unsafe extern "C" fn(*mut T, c_int) -> c_int,
}

unsafe impl<T: Zeroable + Send + Sync, const N: usize> Send for FfiWrapper<T, N> {}
unsafe impl<T: Zeroable + Send + Sync, const N: usize> Sync for FfiWrapper<T, N> {}
```
**What it means:**
- Combines multiple advanced features
- Pin for self-referential async traits
- Unsafe for performance-critical paths
- FFI compatibility with repr(C)
- Const generics for compile-time sizes
- PhantomData for variance control

## Tips for Understanding Function Signatures

1. **Start with parameters and return type**: What goes in, what comes out
2. **Check trait bounds**: What capabilities are required
3. **Look at lifetimes**: How long do references need to live
4. **Identify patterns**: Many signatures follow common patterns
5. **Use IDE help**: Hover over types for expanded information
6. **For unsafe**: Look for SAFETY comments explaining invariants
7. **For macros**: Use cargo expand to see generated code
8. **For FFI**: Check for extern "C" and repr(C) attributes

## Common Trait Bound Combinations

```rust
// Thread-safe shared reference
T: Send + Sync

// Serializable
T: Serialize + Deserialize<'de>

// Comparable and displayable
T: PartialOrd + Display

// Cloneable error type
E: Clone + Error + Send + Sync + 'static

// Async-safe
T: Send + 'static

// Hash map key
K: Eq + Hash

// Numeric operations
T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>

// FFI-safe
T: Copy + 'static

// Pinned async
T: Future + Unpin

// Zero-cost wrapper
T: Deref<Target = U> + DerefMut
```

Remember: Complex signatures often arise from Rust's emphasis on:
- Zero-cost abstractions
- Memory safety
- Thread safety
- Explicit error handling
- FFI compatibility