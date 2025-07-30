# Tutorial 2: Ownership & Borrowing

## Understanding Ownership

Ownership is Rust's most unique feature, enabling memory safety without garbage collection.

### The Three Rules of Ownership

1. Each value has a single owner
2. There can only be one owner at a time
3. When the owner goes out of scope, the value is dropped

```rust
// src/main.rs
fn main() {
    // Rule 1: s owns the String value
    let s = String::from("hello");
    
    { // New scope
        // Rule 2: s2 owns a different String
        let s2 = String::from("world");
        println!("Inside scope: {}", s2);
    } // Rule 3: s2 goes out of scope and is dropped
    
    // s2 is no longer valid here
    println!("Outside scope: {}", s);
} // s goes out of scope and is dropped
```

## Move Semantics

```rust
// src/main.rs
fn main() {
    // Stack Data: Copy
    let x = 5;
    let y = x; // x is copied to y
    println!("x = {}, y = {}", x, y); // Both valid
    
    // Heap Data: Move
    let s1 = String::from("hello");
    let s2 = s1; // s1 is moved to s2
    
    // println!("{}", s1); // ERROR: s1 is no longer valid
    println!("s2 = {}", s2); // OK
    
    // Clone for deep copy
    let s3 = String::from("hello");
    let s4 = s3.clone();
    println!("s3 = {}, s4 = {}", s3, s4); // Both valid
}
```

## Ownership and Functions

```rust
// src/main.rs
fn main() {
    let s = String::from("hello");
    takes_ownership(s); // s is moved into the function
    // println!("{}", s); // ERROR: s is no longer valid
    
    let x = 5;
    makes_copy(x); // x is copied
    println!("x is still valid: {}", x); // OK
    
    let s1 = gives_ownership(); // Function gives ownership to s1
    println!("s1 = {}", s1);
    
    let s2 = String::from("hello");
    let s3 = takes_and_gives_back(s2); // s2 is moved, s3 receives ownership
    println!("s3 = {}", s3);
}

fn takes_ownership(some_string: String) {
    println!("Taking ownership of: {}", some_string);
} // some_string goes out of scope and is dropped

fn makes_copy(some_integer: i32) {
    println!("Making copy of: {}", some_integer);
}

fn gives_ownership() -> String {
    let some_string = String::from("created inside");
    some_string // Ownership is moved to caller
}

fn takes_and_gives_back(a_string: String) -> String {
    a_string // Ownership is returned
}
```

## References and Borrowing

References allow you to refer to a value without taking ownership.

```rust
// src/main.rs
fn main() {
    let s1 = String::from("hello");
    
    // Borrowing with immutable reference
    let len = calculate_length(&s1);
    println!("The length of '{}' is {}.", s1, len); // s1 is still valid
    
    // Multiple immutable references are allowed
    let r1 = &s1;
    let r2 = &s1;
    println!("r1: {}, r2: {}", r1, r2);
}

fn calculate_length(s: &String) -> usize {
    s.len()
} // s goes out of scope but doesn't drop the value (no ownership)
```

## Mutable References

```rust
// src/main.rs
fn main() {
    let mut s = String::from("hello");
    
    // Mutable reference
    change(&mut s);
    println!("Changed string: {}", s);
    
    // Only one mutable reference at a time
    let r1 = &mut s;
    // let r2 = &mut s; // ERROR: cannot borrow as mutable more than once
    println!("r1: {}", r1);
    
    // Mutable and immutable references cannot coexist
    let mut s2 = String::from("hello");
    let r3 = &s2; // immutable
    let r4 = &s2; // immutable
    // let r5 = &mut s2; // ERROR: cannot borrow as mutable when immutable refs exist
    println!("r3: {}, r4: {}", r3, r4);
    // r3 and r4 are no longer used after this point
    
    let r5 = &mut s2; // OK: previous refs are out of scope
    r5.push_str(" world");
    println!("r5: {}", r5);
}

fn change(some_string: &mut String) {
    some_string.push_str(", world");
}
```

## The Rules of References

1. You can have either:
   - One mutable reference, OR
   - Any number of immutable references
2. References must always be valid (no dangling references)

```rust
// src/main.rs
fn main() {
    // Example of reference scope
    let mut s = String::from("hello");
    
    let r1 = &s; // immutable borrow starts
    let r2 = &s; // immutable borrow starts
    println!("{} and {}", r1, r2);
    // r1 and r2 go out of scope here
    
    let r3 = &mut s; // mutable borrow starts (OK because r1, r2 are done)
    r3.push_str(" world");
    println!("{}", r3);
}

// This would create a dangling reference (Rust prevents this)
/*
fn dangle() -> &String { // ERROR
    let s = String::from("hello");
    &s // s goes out of scope, reference would be invalid
}
*/

// Correct way: return owned value
fn no_dangle() -> String {
    let s = String::from("hello");
    s // Ownership is moved out
}
```

## Slice Type

Slices let you reference a contiguous sequence of elements without ownership.

```rust
// src/main.rs
fn main() {
    // String slices
    let s = String::from("hello world");
    
    let hello = &s[0..5];  // or &s[..5]
    let world = &s[6..11]; // or &s[6..]
    let whole = &s[..];    // entire string
    
    println!("Slices: '{}', '{}', '{}'", hello, world, whole);
    
    // Using with functions
    let word = first_word(&s);
    println!("First word: {}", word);
    
    // Array slices
    let a = [1, 2, 3, 4, 5];
    let slice = &a[1..3];
    println!("Array slice: {:?}", slice);
}

fn first_word(s: &str) -> &str {
    let bytes = s.as_bytes();
    
    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            return &s[0..i];
        }
    }
    
    &s[..]
}
```

## Lifetimes Introduction

Lifetimes ensure references are valid for as long as we need them.

```rust
// src/main.rs
fn main() {
    let string1 = String::from("abcd");
    let string2 = "xyz";
    
    let result = longest(string1.as_str(), string2);
    println!("The longest string is {}", result);
    
    // Lifetime in struct
    let novel = String::from("Call me Ishmael. Some years ago...");
    let first_sentence = novel.split('.').next().expect("Could not find a '.'");
    let i = ImportantExcerpt {
        part: first_sentence,
    };
    println!("Excerpt: {}", i.part);
}

// Lifetime annotations
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

// Struct with lifetime
struct ImportantExcerpt<'a> {
    part: &'a str,
}
```

## Practical Example: Document Parser

```rust
// src/main.rs
#[derive(Debug)]
struct Document {
    title: String,
    content: String,
    word_count: usize,
}

impl Document {
    // Takes ownership of strings
    fn new(title: String, content: String) -> Self {
        let word_count = content.split_whitespace().count();
        Document {
            title,
            content,
            word_count,
        }
    }
    
    // Borrows self immutably
    fn summary(&self) -> String {
        format!("{} ({} words)", self.title, self.word_count)
    }
    
    // Borrows self mutably
    fn append_content(&mut self, text: &str) {
        self.content.push_str(text);
        self.word_count = self.content.split_whitespace().count();
    }
    
    // Returns a slice of the content
    fn preview(&self, max_len: usize) -> &str {
        let end = self.content.len().min(max_len);
        &self.content[..end]
    }
}

fn main() {
    // Create document (ownership transfer)
    let mut doc = Document::new(
        String::from("Rust Ownership"),
        String::from("Understanding ownership is crucial for Rust programming."),
    );
    
    // Immutable borrow
    println!("Summary: {}", doc.summary());
    
    // Another immutable borrow
    let preview = doc.preview(20);
    println!("Preview: {}...", preview);
    
    // Mutable borrow
    doc.append_content(" It ensures memory safety without garbage collection.");
    
    // Can use immutable borrow again
    println!("Updated: {}", doc.summary());
    
    // Transfer ownership to function
    process_document(doc);
    // doc is no longer available here
}

fn process_document(doc: Document) {
    println!("Processing: {:?}", doc);
    // doc is dropped at the end of this function
}
```

## Common Patterns

### Returning Multiple Values Without Losing Ownership

```rust
// src/main.rs
fn main() {
    let s = String::from("hello");
    let (s, len) = calculate_length_return_ownership(s);
    println!("'{}' has length {}", s, len);
    
    // Better way using references
    let s2 = String::from("world");
    let len2 = calculate_length_borrow(&s2);
    println!("'{}' has length {}", s2, len2);
}

// Awkward way
fn calculate_length_return_ownership(s: String) -> (String, usize) {
    let length = s.len();
    (s, length)
}

// Better way
fn calculate_length_borrow(s: &String) -> usize {
    s.len()
}
```

### Builder Pattern with Ownership

```rust
// src/main.rs
struct Config {
    name: String,
    value: i32,
}

struct ConfigBuilder {
    name: Option<String>,
    value: Option<i32>,
}

impl ConfigBuilder {
    fn new() -> Self {
        ConfigBuilder {
            name: None,
            value: None,
        }
    }
    
    fn name(mut self, name: String) -> Self {
        self.name = Some(name);
        self // Return ownership
    }
    
    fn value(mut self, value: i32) -> Self {
        self.value = Some(value);
        self // Return ownership
    }
    
    fn build(self) -> Result<Config, String> {
        let name = self.name.ok_or("Name is required")?;
        let value = self.value.ok_or("Value is required")?;
        Ok(Config { name, value })
    }
}

fn main() {
    let config = ConfigBuilder::new()
        .name(String::from("setting1"))
        .value(42)
        .build()
        .unwrap();
    
    println!("Config: {} = {}", config.name, config.value);
}
```

## Exercises

1. **Ownership Transfer**: Write a function that takes a `Vec<String>`, adds an element, and returns it. Then rewrite it using a mutable reference.

2. **String Manipulation**: Create a function that takes a string slice and returns the first and last words as a tuple of string slices.

3. **Lifetime Practice**: Write a function that takes two string slices and returns the one that comes first alphabetically, with proper lifetime annotations.

4. **Reference Rules**: Create examples that demonstrate compile errors for:
   - Using a moved value
   - Multiple mutable references
   - Dangling references

## Key Takeaways

- Every value has a single owner
- Values are dropped when their owner goes out of scope
- References borrow values without taking ownership
- Mutable references have exclusive access
- Lifetimes ensure references remain valid
- Prefer borrowing over taking ownership when possible

## Next Steps

In the next tutorial, we'll explore **Structs and Enums**, which are fundamental for domain modeling in Rust. These types, combined with ownership concepts, form the foundation for building robust, type-safe applications.