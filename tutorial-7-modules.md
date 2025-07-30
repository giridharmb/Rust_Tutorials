# Tutorial 7: Modules & Packages

## Understanding Rust's Module System

Rust organizes code using:
- **Packages**: Cargo feature that builds, tests, and shares crates
- **Crates**: Tree of modules that produces a library or executable
- **Modules**: Control organization, scope, and privacy of paths
- **Paths**: Way of naming items like structs, functions, or modules

## Creating a Package

```bash
# Create a new package with a binary crate
cargo new my_project

# Create a new package with a library crate
cargo new my_lib --lib

# Package structure
my_project/
├── Cargo.toml      # Package manifest
└── src/
    └── main.rs     # Binary crate root (for executables)
    └── lib.rs      # Library crate root (for libraries)
```

## Basic Module Structure

```rust
// src/lib.rs or src/main.rs

// Define a module
mod front_of_house {
    // Public module
    pub mod hosting {
        pub fn add_to_waitlist() {
            println!("Added to waitlist");
        }
        
        fn seat_at_table() {
            println!("Seated at table");
        }
    }
    
    // Private module (default)
    mod serving {
        fn take_order() {}
        fn serve_order() {}
        fn take_payment() {}
    }
}

// Using paths
pub fn eat_at_restaurant() {
    // Absolute path
    crate::front_of_house::hosting::add_to_waitlist();
    
    // Relative path
    front_of_house::hosting::add_to_waitlist();
}
```

## Module Tree and Privacy

```rust
// src/lib.rs
mod restaurant {
    // Making structs and enums public
    #[derive(Debug)]
    pub struct Breakfast {
        pub toast: String,
        seasonal_fruit: String, // Private field
    }
    
    impl Breakfast {
        // Public constructor needed because of private field
        pub fn summer(toast: &str) -> Breakfast {
            Breakfast {
                toast: String::from(toast),
                seasonal_fruit: String::from("peaches"),
            }
        }
    }
    
    // Enums: if public, all variants are public
    #[derive(Debug)]
    pub enum Appetizer {
        Soup,
        Salad,
    }
}

pub fn order_breakfast() {
    let mut meal = restaurant::Breakfast::summer("Rye");
    meal.toast = String::from("Wheat"); // Can modify public field
    // meal.seasonal_fruit = ... // Error: private field
    
    let appetizer = restaurant::Appetizer::Soup;
    println!("Ordered: {:?}", appetizer);
}
```

## Use Keyword and Re-exporting

```rust
// src/lib.rs
mod shapes {
    #[derive(Debug)]
    pub struct Rectangle {
        pub width: f64,
        pub height: f64,
    }
    
    impl Rectangle {
        pub fn new(width: f64, height: f64) -> Self {
            Rectangle { width, height }
        }
        
        pub fn area(&self) -> f64 {
            self.width * self.height
        }
    }
    
    #[derive(Debug)]
    pub struct Circle {
        pub radius: f64,
    }
}

// Bringing items into scope
use crate::shapes::Rectangle;
use crate::shapes::Circle;

// Or using nested paths
use crate::shapes::{Rectangle as Rect, Circle};

// Re-exporting
pub use crate::shapes::Rectangle;

// Using glob to bring all public items
use crate::shapes::*;

pub fn calculate_areas() {
    let rect = Rect::new(10.0, 20.0);
    let circle = Circle { radius: 5.0 };
    
    println!("Rectangle area: {}", rect.area());
}
```

## Separating Modules into Files

```rust
// src/lib.rs
mod config;
mod database;
mod api;

pub use config::Config;
pub use database::Database;
pub use api::ApiClient;

// src/config.rs
#[derive(Debug)]
pub struct Config {
    pub host: String,
    pub port: u16,
}

impl Config {
    pub fn new(host: String, port: u16) -> Self {
        Config { host, port }
    }
}

// src/database.rs
use crate::config::Config;

pub struct Database {
    config: Config,
}

impl Database {
    pub fn connect(config: Config) -> Self {
        println!("Connecting to {}:{}", config.host, config.port);
        Database { config }
    }
}

// src/api.rs
pub struct ApiClient {
    base_url: String,
}

impl ApiClient {
    pub fn new(base_url: String) -> Self {
        ApiClient { base_url }
    }
}
```

## Nested Modules in Directories

```bash
# File structure
src/
├── lib.rs
├── network/
│   ├── mod.rs
│   ├── client.rs
│   └── server.rs
└── utils/
    ├── mod.rs
    ├── logger.rs
    └── helpers.rs
```

```rust
// src/lib.rs
mod network;
mod utils;

pub use network::client::Client;
pub use network::server::Server;
pub use utils::logger::Logger;

// src/network/mod.rs
pub mod client;
pub mod server;

// Common functionality for the network module
pub fn initialize() {
    println!("Initializing network module");
}

// src/network/client.rs
pub struct Client {
    address: String,
}

impl Client {
    pub fn new(address: String) -> Self {
        Client { address }
    }
    
    pub fn connect(&self) {
        super::initialize(); // Access parent module
        println!("Client connecting to {}", self.address);
    }
}

// src/network/server.rs
pub struct Server {
    port: u16,
}

impl Server {
    pub fn new(port: u16) -> Self {
        Server { port }
    }
    
    pub fn start(&self) {
        println!("Server starting on port {}", self.port);
    }
}
```

## Creating a Library Crate

```rust
// Cargo.toml
[package]
name = "my_utils"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

// src/lib.rs
//! # My Utils
//! 
//! A collection of utility functions and types.

/// Mathematical utilities
pub mod math {
    /// Calculates the factorial of a number
    /// 
    /// # Examples
    /// 
    /// ```
    /// use my_utils::math::factorial;
    /// 
    /// assert_eq!(factorial(5), 120);
    /// ```
    pub fn factorial(n: u32) -> u32 {
        match n {
            0 => 1,
            _ => n * factorial(n - 1),
        }
    }
    
    /// Checks if a number is prime
    pub fn is_prime(n: u32) -> bool {
        if n <= 1 {
            return false;
        }
        for i in 2..=(n as f64).sqrt() as u32 {
            if n % i == 0 {
                return false;
            }
        }
        true
    }
}

/// String manipulation utilities
pub mod strings {
    /// Reverses a string
    pub fn reverse(s: &str) -> String {
        s.chars().rev().collect()
    }
    
    /// Checks if a string is a palindrome
    pub fn is_palindrome(s: &str) -> bool {
        let cleaned: String = s.chars()
            .filter(|c| c.is_alphanumeric())
            .map(|c| c.to_lowercase().to_string())
            .collect();
        
        cleaned == reverse(&cleaned)
    }
}

/// Data structures
pub mod collections {
    use std::collections::HashMap;
    
    /// A simple cache implementation
    pub struct Cache<K, V> {
        storage: HashMap<K, V>,
        capacity: usize,
    }
    
    impl<K: Eq + std::hash::Hash, V> Cache<K, V> {
        pub fn new(capacity: usize) -> Self {
            Cache {
                storage: HashMap::with_capacity(capacity),
                capacity,
            }
        }
        
        pub fn insert(&mut self, key: K, value: V) {
            if self.storage.len() >= self.capacity {
                // Simple eviction: clear all (not LRU)
                self.storage.clear();
            }
            self.storage.insert(key, value);
        }
        
        pub fn get(&self, key: &K) -> Option<&V> {
            self.storage.get(key)
        }
    }
}

// Re-export commonly used items at crate root
pub use collections::Cache;
pub use math::{factorial, is_prime};
pub use strings::{reverse, is_palindrome};

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(5), 120);
    }
    
    #[test]
    fn test_palindrome() {
        assert!(is_palindrome("A man a plan a canal Panama"));
        assert!(!is_palindrome("hello"));
    }
}
```

## Real-World Package Structure

```bash
# Complex project structure
my_app/
├── Cargo.toml
├── src/
│   ├── main.rs           # Application entry point
│   ├── lib.rs            # Library root (optional)
│   ├── config/
│   │   ├── mod.rs
│   │   ├── settings.rs
│   │   └── validation.rs
│   ├── models/
│   │   ├── mod.rs
│   │   ├── user.rs
│   │   ├── product.rs
│   │   └── order.rs
│   ├── services/
│   │   ├── mod.rs
│   │   ├── auth.rs
│   │   ├── database.rs
│   │   └── email.rs
│   ├── handlers/
│   │   ├── mod.rs
│   │   ├── user_handler.rs
│   │   └── product_handler.rs
│   └── utils/
│       ├── mod.rs
│       ├── errors.rs
│       └── helpers.rs
├── tests/
│   └── integration_test.rs
├── benches/
│   └── benchmark.rs
└── examples/
    └── example.rs
```

```rust
// src/main.rs
mod config;
mod models;
mod services;
mod handlers;
mod utils;

use config::Settings;
use services::{Database, AuthService};
use handlers::{UserHandler, ProductHandler};

fn main() {
    // Load configuration
    let settings = Settings::load().expect("Failed to load settings");
    
    // Initialize services
    let db = Database::connect(&settings.database_url);
    let auth = AuthService::new(&settings.secret_key);
    
    // Setup handlers
    let user_handler = UserHandler::new(db.clone(), auth.clone());
    let product_handler = ProductHandler::new(db.clone());
    
    println!("Application started!");
}

// src/models/mod.rs
pub mod user;
pub mod product;
pub mod order;

pub use user::User;
pub use product::Product;
pub use order::Order;

// src/models/user.rs
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct User {
    pub id: u64,
    pub username: String,
    pub email: String,
    #[serde(skip_serializing)]
    pub password_hash: String,
}

impl User {
    pub fn new(username: String, email: String, password: &str) -> Self {
        User {
            id: 0, // Will be set by database
            username,
            email,
            password_hash: hash_password(password),
        }
    }
}

fn hash_password(password: &str) -> String {
    // Simplified - use proper hashing in production
    format!("hashed_{}", password)
}
```

## Workspace for Multiple Packages

```toml
# Cargo.toml (workspace root)
[workspace]
members = [
    "server",
    "client",
    "shared",
]

# server/Cargo.toml
[package]
name = "server"
version = "0.1.0"
edition = "2021"

[dependencies]
shared = { path = "../shared" }
tokio = { version = "1", features = ["full"] }

# shared/Cargo.toml
[package]
name = "shared"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
```

```rust
// shared/src/lib.rs
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
    pub id: u64,
    pub content: String,
    pub timestamp: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Command {
    Connect { username: String },
    Disconnect,
    SendMessage(Message),
}

// server/src/main.rs
use shared::{Message, Command};

fn handle_command(cmd: Command) {
    match cmd {
        Command::Connect { username } => {
            println!("User {} connected", username);
        }
        Command::SendMessage(msg) => {
            println!("Message from {}: {}", msg.id, msg.content);
        }
        Command::Disconnect => {
            println!("User disconnected");
        }
    }
}

fn main() {
    println!("Server started");
}
```

## Best Practices and Patterns

```rust
// src/lib.rs

// 1. Use facade pattern for complex APIs
pub mod database {
    mod connection;
    mod query_builder;
    mod migrations;
    
    // Expose simplified interface
    pub use connection::Connection;
    
    pub fn connect(url: &str) -> Result<Connection, Error> {
        Connection::establish(url)
    }
    
    #[derive(Debug)]
    pub struct Error(String);
}

// 2. Builder pattern with modules
pub mod config {
    pub struct ConfigBuilder {
        settings: Settings,
    }
    
    #[derive(Default)]
    struct Settings {
        debug: bool,
        port: u16,
        host: String,
    }
    
    impl ConfigBuilder {
        pub fn new() -> Self {
            ConfigBuilder {
                settings: Settings::default(),
            }
        }
        
        pub fn debug(mut self, debug: bool) -> Self {
            self.settings.debug = debug;
            self
        }
        
        pub fn port(mut self, port: u16) -> Self {
            self.settings.port = port;
            self
        }
        
        pub fn build(self) -> Config {
            Config {
                settings: self.settings,
            }
        }
    }
    
    pub struct Config {
        settings: Settings,
    }
}

// 3. Feature flags for conditional compilation
#[cfg(feature = "json")]
pub mod json {
    pub fn parse(input: &str) -> Result<serde_json::Value, serde_json::Error> {
        serde_json::from_str(input)
    }
}

// 4. Platform-specific modules
#[cfg(target_os = "windows")]
mod platform {
    pub fn get_home_dir() -> String {
        std::env::var("USERPROFILE").unwrap_or_default()
    }
}

#[cfg(not(target_os = "windows"))]
mod platform {
    pub fn get_home_dir() -> String {
        std::env::var("HOME").unwrap_or_default()
    }
}

pub use platform::get_home_dir;
```

## Exercises

1. **Library Design**: Create a library crate for a simple task queue with modules for queue operations, task types, and worker management.

2. **Module Refactoring**: Take a large single-file program and refactor it into a well-organized module structure.

3. **Plugin System**: Design a module structure that supports plugins, where each plugin is a separate module with a common trait interface.

4. **Cross-Platform Library**: Build a library with platform-specific implementations hidden behind a common interface.

5. **Workspace Project**: Create a workspace with a shared library, CLI tool, and web server that all use the shared code.

## Key Takeaways

- Modules control code organization and privacy
- `pub` keyword makes items public
- `use` brings items into scope
- Modules can be in separate files or directories
- Re-exporting with `pub use` creates better APIs
- Workspaces enable multi-package projects
- Good module structure improves maintainability
- Documentation comments with `///` and `//!`

## Next Steps

In the next tutorial, we'll explore **Testing and Documentation**, learning how to write unit tests, integration tests, benchmarks, and create excellent documentation for your Rust code.