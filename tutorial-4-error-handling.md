# Tutorial 4: Error Handling

## Understanding Result<T, E>

Rust uses `Result<T, E>` for recoverable errors, making error handling explicit and type-safe.

```rust
// src/main.rs
use std::fs::File;
use std::io::ErrorKind;

fn main() {
    // Result is defined as:
    // enum Result<T, E> {
    //     Ok(T),
    //     Err(E),
    // }
    
    // Basic Result handling
    let f = File::open("hello.txt");
    
    let f = match f {
        Ok(file) => file,
        Err(error) => match error.kind() {
            ErrorKind::NotFound => {
                println!("File not found, creating it...");
                match File::create("hello.txt") {
                    Ok(fc) => fc,
                    Err(e) => panic!("Problem creating file: {:?}", e),
                }
            }
            other_error => {
                panic!("Problem opening file: {:?}", other_error)
            }
        },
    };
    
    println!("File opened successfully!");
}
```

## The ? Operator

The `?` operator provides a concise way to propagate errors.

```rust
// src/main.rs
use std::fs::File;
use std::io::{self, Read};

// Using ? operator for error propagation
fn read_username_from_file() -> Result<String, io::Error> {
    let mut f = File::open("username.txt")?;
    let mut s = String::new();
    f.read_to_string(&mut s)?;
    Ok(s)
}

// Even more concise
fn read_username_short() -> Result<String, io::Error> {
    let mut s = String::new();
    File::open("username.txt")?.read_to_string(&mut s)?;
    Ok(s)
}

// Using standard library convenience
fn read_username_shortest() -> Result<String, io::Error> {
    std::fs::read_to_string("username.txt")
}

fn main() {
    match read_username_from_file() {
        Ok(username) => println!("Username: {}", username),
        Err(e) => println!("Error reading username: {}", e),
    }
}
```

## Custom Error Types

```rust
// src/main.rs
use std::fmt;
use std::error::Error;

// Custom error type
#[derive(Debug)]
enum AppError {
    Io(std::io::Error),
    Parse(std::num::ParseIntError),
    Validation(String),
}

// Implement Display for user-friendly error messages
impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AppError::Io(e) => write!(f, "IO error: {}", e),
            AppError::Parse(e) => write!(f, "Parse error: {}", e),
            AppError::Validation(msg) => write!(f, "Validation error: {}", msg),
        }
    }
}

// Implement Error trait
impl Error for AppError {}

// Implement From for automatic conversion
impl From<std::io::Error> for AppError {
    fn from(error: std::io::Error) -> Self {
        AppError::Io(error)
    }
}

impl From<std::num::ParseIntError> for AppError {
    fn from(error: std::num::ParseIntError) -> Self {
        AppError::Parse(error)
    }
}

// Using custom error
fn process_file(filename: &str) -> Result<i32, AppError> {
    let contents = std::fs::read_to_string(filename)?; // Automatically converts io::Error
    let number: i32 = contents.trim().parse()?; // Automatically converts ParseIntError
    
    if number < 0 {
        return Err(AppError::Validation("Number must be positive".to_string()));
    }
    
    Ok(number * 2)
}

fn main() {
    // Create test file
    std::fs::write("number.txt", "42").expect("Unable to write file");
    
    match process_file("number.txt") {
        Ok(result) => println!("Processed result: {}", result),
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

## Error Handling Patterns

```rust
// src/main.rs
use std::collections::HashMap;

#[derive(Debug)]
struct Config {
    settings: HashMap<String, String>,
}

#[derive(Debug)]
enum ConfigError {
    Missing(String),
    Invalid(String),
}

impl Config {
    fn new() -> Self {
        Config {
            settings: HashMap::new(),
        }
    }
    
    fn set(&mut self, key: String, value: String) {
        self.settings.insert(key, value);
    }
    
    // Pattern 1: Return Result
    fn get(&self, key: &str) -> Result<&String, ConfigError> {
        self.settings
            .get(key)
            .ok_or_else(|| ConfigError::Missing(key.to_string()))
    }
    
    // Pattern 2: Return Option
    fn get_optional(&self, key: &str) -> Option<&String> {
        self.settings.get(key)
    }
    
    // Pattern 3: Provide default
    fn get_or_default(&self, key: &str, default: &str) -> String {
        self.settings
            .get(key)
            .cloned()
            .unwrap_or_else(|| default.to_string())
    }
    
    // Pattern 4: Panic with expect (for unrecoverable errors)
    fn get_required(&self, key: &str) -> &String {
        self.settings
            .get(key)
            .expect(&format!("Required configuration '{}' not found", key))
    }
}

fn main() {
    let mut config = Config::new();
    config.set("host".to_string(), "localhost".to_string());
    config.set("port".to_string(), "8080".to_string());
    
    // Using different patterns
    match config.get("host") {
        Ok(value) => println!("Host: {}", value),
        Err(ConfigError::Missing(key)) => println!("Missing key: {}", key),
        Err(ConfigError::Invalid(msg)) => println!("Invalid: {}", msg),
    }
    
    if let Some(port) = config.get_optional("port") {
        println!("Port: {}", port);
    }
    
    let timeout = config.get_or_default("timeout", "30");
    println!("Timeout: {}", timeout);
}
```

## Result Combinators

```rust
// src/main.rs
use std::num::ParseIntError;

fn parse_and_double(s: &str) -> Result<i32, ParseIntError> {
    s.parse::<i32>().map(|n| n * 2)
}

fn divide(numerator: f64, denominator: f64) -> Result<f64, String> {
    if denominator == 0.0 {
        Err("Cannot divide by zero".to_string())
    } else {
        Ok(numerator / denominator)
    }
}

fn main() {
    // map: Transform the Ok value
    let result = parse_and_double("21");
    println!("Doubled: {:?}", result);
    
    // map_err: Transform the error
    let result = divide(10.0, 0.0)
        .map_err(|e| format!("Division error: {}", e));
    println!("Division: {:?}", result);
    
    // and_then: Chain operations that return Result
    let result: Result<i32, ParseIntError> = "4"
        .parse::<i32>()
        .and_then(|n| Ok(n * 2))
        .and_then(|n| Ok(n + 1));
    println!("Chained: {:?}", result);
    
    // or_else: Provide alternative Result on error
    let result = "not_a_number"
        .parse::<i32>()
        .or_else(|_| Ok(0)); // Default to 0 on parse error
    println!("With default: {:?}", result);
    
    // unwrap_or: Extract value or provide default
    let value = "42".parse::<i32>().unwrap_or(0);
    println!("Value: {}", value);
    
    // ok(): Convert Result to Option
    let maybe_number = "123".parse::<i32>().ok();
    println!("As Option: {:?}", maybe_number);
}
```

## Real-World Example: File Processing System

```rust
// src/main.rs
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::fmt;

#[derive(Debug)]
enum ProcessError {
    Io(io::Error),
    Parse(String),
    Validation(String),
    NotFound(String),
}

impl fmt::Display for ProcessError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ProcessError::Io(e) => write!(f, "IO error: {}", e),
            ProcessError::Parse(msg) => write!(f, "Parse error: {}", msg),
            ProcessError::Validation(msg) => write!(f, "Validation error: {}", msg),
            ProcessError::NotFound(item) => write!(f, "Not found: {}", item),
        }
    }
}

impl From<io::Error> for ProcessError {
    fn from(error: io::Error) -> Self {
        ProcessError::Io(error)
    }
}

#[derive(Debug)]
struct DataRecord {
    id: u32,
    name: String,
    value: f64,
}

struct DataProcessor {
    records: Vec<DataRecord>,
}

impl DataProcessor {
    fn new() -> Self {
        DataProcessor {
            records: Vec::new(),
        }
    }
    
    // Load data from CSV-like file
    fn load_from_file(&mut self, path: &Path) -> Result<(), ProcessError> {
        let contents = fs::read_to_string(path)?;
        
        for (line_num, line) in contents.lines().enumerate() {
            // Skip header
            if line_num == 0 {
                continue;
            }
            
            let record = self.parse_line(line, line_num)?;
            self.validate_record(&record)?;
            self.records.push(record);
        }
        
        Ok(())
    }
    
    fn parse_line(&self, line: &str, line_num: usize) -> Result<DataRecord, ProcessError> {
        let parts: Vec<&str> = line.split(',').collect();
        
        if parts.len() != 3 {
            return Err(ProcessError::Parse(
                format!("Line {}: Expected 3 fields, found {}", line_num + 1, parts.len())
            ));
        }
        
        let id = parts[0].trim().parse::<u32>()
            .map_err(|_| ProcessError::Parse(format!("Line {}: Invalid ID", line_num + 1)))?;
        
        let name = parts[1].trim().to_string();
        
        let value = parts[2].trim().parse::<f64>()
            .map_err(|_| ProcessError::Parse(format!("Line {}: Invalid value", line_num + 1)))?;
        
        Ok(DataRecord { id, name, value })
    }
    
    fn validate_record(&self, record: &DataRecord) -> Result<(), ProcessError> {
        if record.name.is_empty() {
            return Err(ProcessError::Validation("Name cannot be empty".to_string()));
        }
        
        if record.value < 0.0 {
            return Err(ProcessError::Validation(
                format!("Value must be non-negative, got {}", record.value)
            ));
        }
        
        // Check for duplicate IDs
        if self.records.iter().any(|r| r.id == record.id) {
            return Err(ProcessError::Validation(
                format!("Duplicate ID: {}", record.id)
            ));
        }
        
        Ok(())
    }
    
    fn save_summary(&self, path: &Path) -> Result<(), ProcessError> {
        let mut file = fs::File::create(path)?;
        
        writeln!(file, "Data Summary")?;
        writeln!(file, "============")?;
        writeln!(file, "Total records: {}", self.records.len())?;
        
        if !self.records.is_empty() {
            let sum: f64 = self.records.iter().map(|r| r.value).sum();
            let avg = sum / self.records.len() as f64;
            
            writeln!(file, "Total value: {:.2}", sum)?;
            writeln!(file, "Average value: {:.2}", avg)?;
        }
        
        Ok(())
    }
    
    fn find_record(&self, id: u32) -> Result<&DataRecord, ProcessError> {
        self.records
            .iter()
            .find(|r| r.id == id)
            .ok_or_else(|| ProcessError::NotFound(format!("Record with ID {}", id)))
    }
}

fn main() -> Result<(), ProcessError> {
    // Create test data
    let csv_data = "id,name,value\n1,Item A,100.50\n2,Item B,200.75\n3,Item C,150.25";
    fs::write("data.csv", csv_data)?;
    
    let mut processor = DataProcessor::new();
    
    // Load and process data
    processor.load_from_file(Path::new("data.csv"))?;
    println!("Loaded {} records", processor.records.len());
    
    // Find specific record
    match processor.find_record(2) {
        Ok(record) => println!("Found: {:?}", record),
        Err(e) => println!("Error: {}", e),
    }
    
    // Save summary
    processor.save_summary(Path::new("summary.txt"))?;
    println!("Summary saved to summary.txt");
    
    // Cleanup
    fs::remove_file("data.csv").ok();
    fs::remove_file("summary.txt").ok();
    
    Ok(())
}
```

## Panic vs Result

```rust
// src/main.rs
use std::env;

// When to use panic! (unrecoverable errors)
fn get_config_value(key: &str) -> String {
    env::var(key).unwrap_or_else(|_| {
        panic!("Required environment variable {} not set", key)
    })
}

// When to use Result (recoverable errors)
fn parse_config_value(value: &str) -> Result<u16, String> {
    value.parse::<u16>()
        .map_err(|_| format!("'{}' is not a valid port number", value))
}

fn main() {
    // Set environment variable for demo
    env::set_var("PORT", "8080");
    
    // This would panic if PORT wasn't set
    let port_str = get_config_value("PORT");
    
    // This returns Result for handling
    match parse_config_value(&port_str) {
        Ok(port) => println!("Server will run on port {}", port),
        Err(e) => eprintln!("Configuration error: {}", e),
    }
    
    // Using expect for better panic messages
    let _important_file = std::fs::read_to_string("config.toml")
        .expect("config.toml must exist in the current directory");
}
```

## Error Handling Best Practices

```rust
// src/main.rs
use std::error::Error;
use std::fmt;

// 1. Use thiserror or similar for complex errors
#[derive(Debug)]
enum ApiError {
    Network(String),
    Authentication(String),
    RateLimit { retry_after: u64 },
    Server { code: u16, message: String },
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ApiError::Network(msg) => write!(f, "Network error: {}", msg),
            ApiError::Authentication(msg) => write!(f, "Authentication failed: {}", msg),
            ApiError::RateLimit { retry_after } => {
                write!(f, "Rate limited. Retry after {} seconds", retry_after)
            }
            ApiError::Server { code, message } => {
                write!(f, "Server error ({}): {}", code, message)
            }
        }
    }
}

impl Error for ApiError {}

// 2. Create type aliases for common Results
type ApiResult<T> = Result<T, ApiError>;

// 3. Use early returns with ?
fn make_api_request(endpoint: &str, token: &str) -> ApiResult<String> {
    validate_token(token)?;
    check_rate_limit()?;
    send_request(endpoint)
}

// 4. Provide context when converting errors
fn process_user_data(user_id: u64) -> ApiResult<String> {
    let data = fetch_user(user_id)
        .map_err(|e| ApiError::Network(format!("Failed to fetch user {}: {}", user_id, e)))?;
    
    Ok(data)
}

// Mock functions
fn validate_token(token: &str) -> ApiResult<()> {
    if token.len() < 10 {
        Err(ApiError::Authentication("Invalid token".to_string()))
    } else {
        Ok(())
    }
}

fn check_rate_limit() -> ApiResult<()> {
    Ok(())
}

fn send_request(_endpoint: &str) -> ApiResult<String> {
    Ok("Response data".to_string())
}

fn fetch_user(_user_id: u64) -> Result<String, Box<dyn Error>> {
    Ok("User data".to_string())
}

fn main() {
    // Example usage
    match make_api_request("/users", "valid_token_123") {
        Ok(response) => println!("Success: {}", response),
        Err(ApiError::RateLimit { retry_after }) => {
            println!("Rate limited. Waiting {} seconds...", retry_after);
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

## Exercises

1. **Custom Error Type**: Create a custom error type for a calculator that handles division by zero, overflow, and invalid operations.

2. **File Parser**: Build a configuration file parser that reads key-value pairs and handles various error cases (missing file, invalid format, duplicate keys).

3. **Result Chain**: Write a function that reads a number from a file, doubles it, and writes it to another file, using the ? operator throughout.

4. **Error Recovery**: Create a retry mechanism that attempts an operation up to 3 times before giving up, with exponential backoff.

## Key Takeaways

- Use `Result<T, E>` for recoverable errors
- The `?` operator simplifies error propagation
- Custom error types provide better error messages
- `panic!` is for unrecoverable errors
- Combinators like `map`, `and_then` enable functional error handling
- Type aliases simplify complex Result types
- Always provide context when converting errors

## Next Steps

In the next tutorial, we'll explore **Traits and Generics**, which enable polymorphism and code reuse in Rust. This will build on error handling by showing how to create generic functions that work with any Result type.