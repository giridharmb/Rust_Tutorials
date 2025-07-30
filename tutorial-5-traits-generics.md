# Tutorial 5: Traits & Generics

## Understanding Traits

Traits define shared behavior, similar to interfaces in other languages.

```rust
// src/main.rs
// Define a trait
trait Summary {
    fn summarize(&self) -> String;
    
    // Default implementation
    fn summarize_author(&self) -> String {
        String::from("(Anonymous)")
    }
}

// Implement trait for structs
struct Article {
    title: String,
    author: String,
    content: String,
}

impl Summary for Article {
    fn summarize(&self) -> String {
        format!("{} by {}", self.title, self.author)
    }
    
    fn summarize_author(&self) -> String {
        format!("@{}", self.author)
    }
}

struct Tweet {
    username: String,
    content: String,
    retweet: bool,
}

impl Summary for Tweet {
    fn summarize(&self) -> String {
        format!("{}: {}", self.username, self.content)
    }
    // Uses default implementation for summarize_author
}

fn main() {
    let article = Article {
        title: String::from("Rust Traits Explained"),
        author: String::from("Jane Doe"),
        content: String::from("Traits are powerful..."),
    };
    
    let tweet = Tweet {
        username: String::from("rustlang"),
        content: String::from("Check out the new features!"),
        retweet: false,
    };
    
    println!("Article: {}", article.summarize());
    println!("Tweet: {}", tweet.summarize());
    println!("Tweet author: {}", tweet.summarize_author());
}
```

## Trait Bounds

```rust
// src/main.rs
use std::fmt::Display;

// Function with trait bound
fn notify<T: Summary>(item: &T) {
    println!("Breaking news! {}", item.summarize());
}

// Multiple trait bounds
fn notify_and_display<T: Summary + Display>(item: &T) {
    println!("New item: {}", item);
    println!("Summary: {}", item.summarize());
}

// Where clause for complex bounds
fn some_function<T, U>(t: &T, u: &U) -> i32
where
    T: Display + Clone,
    U: Clone + Debug,
{
    // function body
    42
}

// Returning types that implement traits
fn returns_summarizable() -> impl Summary {
    Tweet {
        username: String::from("bot"),
        content: String::from("Hello world"),
        retweet: false,
    }
}

// Trait bounds with generics
struct Pair<T> {
    x: T,
    y: T,
}

impl<T> Pair<T> {
    fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

// Method only for types that implement Display + PartialOrd
impl<T: Display + PartialOrd> Pair<T> {
    fn cmp_display(&self) {
        if self.x >= self.y {
            println!("The largest member is x = {}", self.x);
        } else {
            println!("The largest member is y = {}", self.y);
        }
    }
}

trait Summary {
    fn summarize(&self) -> String;
}

struct Tweet {
    username: String,
    content: String,
    retweet: bool,
}

impl Summary for Tweet {
    fn summarize(&self) -> String {
        format!("{}: {}", self.username, self.content)
    }
}

impl Display for Tweet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "@{}", self.username)
    }
}

use std::fmt::Debug;

fn main() {
    let tweet = Tweet {
        username: String::from("rustlang"),
        content: String::from("Hello, world!"),
        retweet: false,
    };
    
    notify(&tweet);
    
    let pair = Pair::new(5, 10);
    pair.cmp_display();
}
```

## Generic Functions and Structs

```rust
// src/main.rs
// Generic function
fn largest<T: PartialOrd>(list: &[T]) -> &T {
    let mut largest = &list[0];
    
    for item in list {
        if item > largest {
            largest = item;
        }
    }
    
    largest
}

// Generic struct
#[derive(Debug)]
struct Point<T> {
    x: T,
    y: T,
}

// Multiple generic types
#[derive(Debug)]
struct Rectangle<T, U> {
    width: T,
    height: U,
}

impl<T> Point<T> {
    fn x(&self) -> &T {
        &self.x
    }
}

// Method with different generic type
impl<T> Point<T> {
    fn mixup<U>(self, other: Point<U>) -> Point<T> {
        Point {
            x: self.x,
            y: other.y,
        }
    }
}

// Generic enum (Option and Result are examples)
enum MyOption<T> {
    Some(T),
    None,
}

fn main() {
    // Using generic function
    let numbers = vec![34, 50, 25, 100, 65];
    let result = largest(&numbers);
    println!("The largest number is {}", result);
    
    let chars = vec!['y', 'm', 'a', 'q'];
    let result = largest(&chars);
    println!("The largest char is {}", result);
    
    // Generic structs
    let integer_point = Point { x: 5, y: 10 };
    let float_point = Point { x: 1.0, y: 4.0 };
    
    println!("integer_point.x = {}", integer_point.x());
    
    // Mixed types
    let rect = Rectangle {
        width: 30,
        height: 50.5,
    };
    println!("Rectangle: {:?}", rect);
}
```

## Common Traits

```rust
// src/main.rs
use std::fmt;

// Display trait for user-friendly output
struct User {
    name: String,
    age: u32,
}

impl fmt::Display for User {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} (age {})", self.name, self.age)
    }
}

// Debug trait for developer-friendly output
#[derive(Debug)]
struct Point {
    x: i32,
    y: i32,
}

// Clone trait for explicit copying
#[derive(Clone)]
struct Config {
    name: String,
    value: i32,
}

// PartialEq and Eq for equality comparisons
#[derive(PartialEq, Eq)]
struct Id(u32);

// PartialOrd and Ord for ordering
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Version {
    major: u32,
    minor: u32,
    patch: u32,
}

// Default trait
#[derive(Default)]
struct Settings {
    volume: u8,        // defaults to 0
    brightness: u8,    // defaults to 0
    name: String,      // defaults to empty string
}

// Custom Default implementation
impl Default for User {
    fn default() -> Self {
        User {
            name: String::from("Guest"),
            age: 0,
        }
    }
}

fn main() {
    // Display
    let user = User {
        name: String::from("Alice"),
        age: 30,
    };
    println!("User: {}", user);
    
    // Debug
    let point = Point { x: 3, y: 4 };
    println!("Point: {:?}", point);
    
    // Clone
    let config1 = Config {
        name: String::from("test"),
        value: 42,
    };
    let config2 = config1.clone();
    println!("Cloned config: {} = {}", config2.name, config2.value);
    
    // Equality
    let id1 = Id(1);
    let id2 = Id(1);
    let id3 = Id(2);
    println!("id1 == id2: {}", id1 == id2);
    println!("id1 == id3: {}", id1 == id3);
    
    // Ordering
    let v1 = Version { major: 1, minor: 0, patch: 0 };
    let v2 = Version { major: 1, minor: 2, patch: 0 };
    println!("v1 < v2: {}", v1 < v2);
    
    // Default
    let default_settings = Settings::default();
    let default_user = User::default();
    println!("Default user: {}", default_user);
}
```

## Advanced Trait Features

```rust
// src/main.rs
use std::ops::Add;

// Associated types
trait Iterator {
    type Item;
    
    fn next(&mut self) -> Option<Self::Item>;
}

struct Counter {
    count: u32,
    max: u32,
}

impl Iterator for Counter {
    type Item = u32;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.count < self.max {
            self.count += 1;
            Some(self.count)
        } else {
            None
        }
    }
}

// Operator overloading
#[derive(Debug, PartialEq)]
struct Point {
    x: i32,
    y: i32,
}

impl Add for Point {
    type Output = Point;
    
    fn add(self, other: Point) -> Point {
        Point {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

// Supertraits (traits that require other traits)
trait OutlinePrint: fmt::Display {
    fn outline_print(&self) {
        let output = self.to_string();
        let len = output.len();
        
        println!("{}", "*".repeat(len + 4));
        println!("* {} *", output);
        println!("{}", "*".repeat(len + 4));
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

impl OutlinePrint for Point {}

// Trait objects for dynamic dispatch
trait Draw {
    fn draw(&self);
}

struct Button {
    width: u32,
    height: u32,
    label: String,
}

impl Draw for Button {
    fn draw(&self) {
        println!("Drawing button: {} ({}x{})", self.label, self.width, self.height);
    }
}

struct SelectBox {
    width: u32,
    height: u32,
    options: Vec<String>,
}

impl Draw for SelectBox {
    fn draw(&self) {
        println!("Drawing select box with {} options", self.options.len());
    }
}

struct Screen {
    components: Vec<Box<dyn Draw>>,
}

impl Screen {
    fn run(&self) {
        for component in self.components.iter() {
            component.draw();
        }
    }
}

use std::fmt;

fn main() {
    // Using associated types
    let mut counter = Counter { count: 0, max: 5 };
    while let Some(value) = counter.next() {
        println!("Count: {}", value);
    }
    
    // Operator overloading
    let p1 = Point { x: 1, y: 0 };
    let p2 = Point { x: 2, y: 3 };
    let p3 = p1 + p2;
    println!("Point addition: {:?}", p3);
    
    // Supertrait
    let point = Point { x: 5, y: 10 };
    point.outline_print();
    
    // Trait objects
    let screen = Screen {
        components: vec![
            Box::new(Button {
                width: 50,
                height: 10,
                label: String::from("OK"),
            }),
            Box::new(SelectBox {
                width: 75,
                height: 10,
                options: vec![
                    String::from("Yes"),
                    String::from("No"),
                    String::from("Maybe"),
                ],
            }),
        ],
    };
    
    screen.run();
}
```

## Real-World Example: Database Abstraction

```rust
// src/main.rs
use std::collections::HashMap;
use std::fmt::Debug;

// Define traits for database operations
trait Entity: Clone + Debug {
    type Id: Clone + Eq + std::hash::Hash + Debug;
    
    fn id(&self) -> &Self::Id;
}

trait Repository<T: Entity> {
    fn find(&self, id: &T::Id) -> Option<T>;
    fn find_all(&self) -> Vec<T>;
    fn save(&mut self, entity: T) -> Result<(), String>;
    fn delete(&mut self, id: &T::Id) -> Result<(), String>;
}

// Domain models
#[derive(Clone, Debug)]
struct User {
    id: u64,
    username: String,
    email: String,
}

#[derive(Clone, Debug)]
struct Product {
    id: String,
    name: String,
    price: f64,
}

// Implement Entity trait
impl Entity for User {
    type Id = u64;
    
    fn id(&self) -> &Self::Id {
        &self.id
    }
}

impl Entity for Product {
    type Id = String;
    
    fn id(&self) -> &Self::Id {
        &self.id
    }
}

// Generic in-memory repository
struct InMemoryRepository<T: Entity> {
    storage: HashMap<T::Id, T>,
}

impl<T: Entity> InMemoryRepository<T> {
    fn new() -> Self {
        Self {
            storage: HashMap::new(),
        }
    }
}

impl<T: Entity> Repository<T> for InMemoryRepository<T> {
    fn find(&self, id: &T::Id) -> Option<T> {
        self.storage.get(id).cloned()
    }
    
    fn find_all(&self) -> Vec<T> {
        self.storage.values().cloned().collect()
    }
    
    fn save(&mut self, entity: T) -> Result<(), String> {
        self.storage.insert(entity.id().clone(), entity);
        Ok(())
    }
    
    fn delete(&mut self, id: &T::Id) -> Result<(), String> {
        self.storage.remove(id)
            .map(|_| ())
            .ok_or_else(|| format!("Entity with id {:?} not found", id))
    }
}

// Service layer using trait bounds
struct UserService<R: Repository<User>> {
    repository: R,
}

impl<R: Repository<User>> UserService<R> {
    fn new(repository: R) -> Self {
        Self { repository }
    }
    
    fn create_user(&mut self, username: String, email: String) -> Result<u64, String> {
        let id = self.repository.find_all().len() as u64 + 1;
        let user = User { id, username, email };
        self.repository.save(user.clone())?;
        Ok(id)
    }
    
    fn get_user(&self, id: u64) -> Option<User> {
        self.repository.find(&id)
    }
}

// Generic function working with any repository
fn count_entities<T, R>(repository: &R) -> usize
where
    T: Entity,
    R: Repository<T>,
{
    repository.find_all().len()
}

fn main() {
    // Create repositories
    let mut user_repo = InMemoryRepository::<User>::new();
    let mut product_repo = InMemoryRepository::<Product>::new();
    
    // Use repositories
    user_repo.save(User {
        id: 1,
        username: String::from("alice"),
        email: String::from("alice@example.com"),
    }).unwrap();
    
    product_repo.save(Product {
        id: String::from("PROD001"),
        name: String::from("Laptop"),
        price: 999.99,
    }).unwrap();
    
    // Use service
    let mut user_service = UserService::new(user_repo);
    match user_service.create_user(
        String::from("bob"),
        String::from("bob@example.com")
    ) {
        Ok(id) => println!("Created user with id: {}", id),
        Err(e) => println!("Error: {}", e),
    }
    
    // Generic function
    println!("Total users: {}", count_entities::<User, _>(&user_service.repository));
    println!("Total products: {}", count_entities::<Product, _>(&product_repo));
    
    // Find user
    if let Some(user) = user_service.get_user(1) {
        println!("Found user: {:?}", user);
    }
}
```

## Trait Bounds in Practice

```rust
// src/main.rs
use std::fmt::Debug;
use std::cmp::Ordering;

// Flexible API design with traits
trait Validator<T> {
    fn validate(&self, value: &T) -> Result<(), String>;
}

// Multiple validators
struct RangeValidator<T> {
    min: T,
    max: T,
}

impl<T: PartialOrd + Debug> Validator<T> for RangeValidator<T> {
    fn validate(&self, value: &T) -> Result<(), String> {
        if value < &self.min || value > &self.max {
            Err(format!("{:?} is out of range [{:?}, {:?}]", 
                       value, self.min, self.max))
        } else {
            Ok(())
        }
    }
}

struct LengthValidator {
    min_length: usize,
    max_length: usize,
}

impl Validator<String> for LengthValidator {
    fn validate(&self, value: &String) -> Result<(), String> {
        let len = value.len();
        if len < self.min_length || len > self.max_length {
            Err(format!("Length {} is out of range [{}, {}]", 
                       len, self.min_length, self.max_length))
        } else {
            Ok(())
        }
    }
}

// Composite validator
struct CompositeValidator<T> {
    validators: Vec<Box<dyn Validator<T>>>,
}

impl<T> CompositeValidator<T> {
    fn new() -> Self {
        Self {
            validators: Vec::new(),
        }
    }
    
    fn add(mut self, validator: Box<dyn Validator<T>>) -> Self {
        self.validators.push(validator);
        self
    }
}

impl<T> Validator<T> for CompositeValidator<T> {
    fn validate(&self, value: &T) -> Result<(), String> {
        for validator in &self.validators {
            validator.validate(value)?;
        }
        Ok(())
    }
}

// Type-safe builder pattern with traits
trait Builder {
    type Output;
    
    fn build(self) -> Result<Self::Output, String>;
}

#[derive(Debug)]
struct Server {
    host: String,
    port: u16,
    max_connections: usize,
}

struct ServerBuilder {
    host: Option<String>,
    port: Option<u16>,
    max_connections: Option<usize>,
}

impl ServerBuilder {
    fn new() -> Self {
        Self {
            host: None,
            port: None,
            max_connections: None,
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
    
    fn max_connections(mut self, max: usize) -> Self {
        self.max_connections = Some(max);
        self
    }
}

impl Builder for ServerBuilder {
    type Output = Server;
    
    fn build(self) -> Result<Self::Output, String> {
        Ok(Server {
            host: self.host.ok_or("Host is required")?,
            port: self.port.ok_or("Port is required")?,
            max_connections: self.max_connections.unwrap_or(100),
        })
    }
}

fn main() {
    // Using validators
    let age_validator = RangeValidator { min: 18, max: 100 };
    let name_validator = LengthValidator { min_length: 2, max_length: 50 };
    
    match age_validator.validate(&25) {
        Ok(_) => println!("Age is valid"),
        Err(e) => println!("Validation error: {}", e),
    }
    
    match name_validator.validate(&String::from("J")) {
        Ok(_) => println!("Name is valid"),
        Err(e) => println!("Validation error: {}", e),
    }
    
    // Composite validator
    let number_validator = CompositeValidator::new()
        .add(Box::new(RangeValidator { min: 0, max: 100 }));
    
    // Builder pattern
    let server = ServerBuilder::new()
        .host("localhost")
        .port(8080)
        .max_connections(1000)
        .build()
        .unwrap();
    
    println!("Server configured: {:?}", server);
}
```

## Exercises

1. **Generic Data Structure**: Implement a generic `Stack<T>` with push, pop, and peek methods. Add trait bounds to ensure it only works with `Clone` types.

2. **Custom Iterator**: Create a `Fibonacci` struct that implements the `Iterator` trait to generate Fibonacci numbers.

3. **Trait Composition**: Design a trait hierarchy for a game system with `Drawable`, `Updatable`, and `Collidable` traits. Create game objects that implement various combinations.

4. **Type-Safe State Machine**: Use traits and generics to implement a compile-time validated state machine (e.g., a door that can be Open, Closed, or Locked).

## Key Takeaways

- Traits define shared behavior across types
- Generics enable code reuse without sacrificing type safety
- Trait bounds restrict generic types to those implementing specific traits
- Associated types simplify trait definitions
- Trait objects enable runtime polymorphism
- Common traits like `Clone`, `Debug`, and `Display` provide standard functionality
- Traits can be composed and extended for complex abstractions

## Next Steps

In the next tutorial, we'll explore **Collections and Iterators**, building on traits and generics to understand Rust's powerful iteration patterns and data structures.