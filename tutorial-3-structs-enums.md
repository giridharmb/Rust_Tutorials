# Tutorial 3: Structs & Enums

## Defining Structs

Structs let you create custom types that group related data together.

```rust
// src/main.rs
// Classic struct with named fields
struct User {
    username: String,
    email: String,
    sign_in_count: u64,
    active: bool,
}

// Tuple struct
struct Point(i32, i32, i32);

// Unit struct (zero-sized)
struct AlwaysEqual;

fn main() {
    // Creating struct instances
    let user1 = User {
        email: String::from("someone@example.com"),
        username: String::from("someusername123"),
        active: true,
        sign_in_count: 1,
    };
    
    // Accessing fields
    println!("User: {}", user1.username);
    
    // Mutable struct
    let mut user2 = User {
        email: String::from("another@example.com"),
        username: String::from("anotheruser"),
        active: true,
        sign_in_count: 1,
    };
    user2.email = String::from("newemail@example.com");
    
    // Struct update syntax
    let user3 = User {
        email: String::from("third@example.com"),
        ..user1 // Copy remaining fields from user1
    };
    
    // Tuple structs
    let origin = Point(0, 0, 0);
    println!("Origin x: {}", origin.0);
    
    // Unit struct
    let _subject = AlwaysEqual;
}
```

## Field Init Shorthand and Builder Functions

```rust
// src/main.rs
struct User {
    username: String,
    email: String,
    active: bool,
}

// Builder function with field init shorthand
fn build_user(email: String, username: String) -> User {
    User {
        email,    // shorthand for email: email
        username, // shorthand for username: username
        active: true,
    }
}

fn main() {
    let user = build_user(
        String::from("user@example.com"),
        String::from("user123"),
    );
    println!("Built user: {}", user.username);
}
```

## Methods and Associated Functions

```rust
// src/main.rs
#[derive(Debug)]
struct Rectangle {
    width: u32,
    height: u32,
}

impl Rectangle {
    // Associated function (like static method)
    fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
    
    // Method (takes &self)
    fn area(&self) -> u32 {
        self.width * self.height
    }
    
    // Method that borrows mutably
    fn double_size(&mut self) {
        self.width *= 2;
        self.height *= 2;
    }
    
    // Method that takes ownership (rare)
    fn destroy(self) {
        println!("Rectangle destroyed: {:?}", self);
    }
    
    // Method with parameters
    fn can_hold(&self, other: &Rectangle) -> bool {
        self.width > other.width && self.height > other.height
    }
}

// Multiple impl blocks are allowed
impl Rectangle {
    fn perimeter(&self) -> u32 {
        2 * (self.width + self.height)
    }
}

fn main() {
    let mut rect1 = Rectangle::new(30, 50);
    let rect2 = Rectangle { width: 10, height: 40 };
    
    println!("Area: {}", rect1.area());
    println!("Perimeter: {}", rect1.perimeter());
    
    rect1.double_size();
    println!("After doubling: {:?}", rect1);
    
    println!("Can rect1 hold rect2? {}", rect1.can_hold(&rect2));
    
    // rect2.destroy(); // This would consume rect2
}
```

## Enums and Pattern Matching

Enums allow you to define a type by enumerating its possible variants.

```rust
// src/main.rs
// Simple enum
#[derive(Debug)]
enum IpAddrKind {
    V4,
    V6,
}

// Enum with data
#[derive(Debug)]
enum IpAddr {
    V4(u8, u8, u8, u8),
    V6(String),
}

// Enum with different variant types
#[derive(Debug)]
enum Message {
    Quit,                       // No data
    Move { x: i32, y: i32 },   // Named fields
    Write(String),              // Single value
    ChangeColor(i32, i32, i32), // Multiple values
}

impl Message {
    fn process(&self) {
        match self {
            Message::Quit => println!("Quit message"),
            Message::Move { x, y } => println!("Move to ({}, {})", x, y),
            Message::Write(text) => println!("Text message: {}", text),
            Message::ChangeColor(r, g, b) => println!("Change color to ({}, {}, {})", r, g, b),
        }
    }
}

fn main() {
    // Using simple enum
    let four = IpAddrKind::V4;
    let six = IpAddrKind::V6;
    
    // Enum with data
    let home = IpAddr::V4(127, 0, 0, 1);
    let loopback = IpAddr::V6(String::from("::1"));
    
    println!("Home: {:?}", home);
    println!("Loopback: {:?}", loopback);
    
    // Using Message enum
    let messages = vec![
        Message::Quit,
        Message::Move { x: 10, y: 20 },
        Message::Write(String::from("Hello")),
        Message::ChangeColor(255, 0, 0),
    ];
    
    for msg in messages {
        msg.process();
    }
}
```

## The Option Enum

Rust doesn't have null. Instead, it uses the `Option<T>` enum.

```rust
// src/main.rs
fn main() {
    // Option is defined in the standard library as:
    // enum Option<T> {
    //     None,
    //     Some(T),
    // }
    
    let some_number = Some(5);
    let some_string = Some("a string");
    let absent_number: Option<i32> = None;
    
    // Using Option with pattern matching
    let x: Option<i32> = Some(5);
    let y: Option<i32> = None;
    
    println!("x + 1 = {:?}", plus_one(x));
    println!("y + 1 = {:?}", plus_one(y));
    
    // Using if let for simple patterns
    let config_max = Some(3u8);
    if let Some(max) = config_max {
        println!("The maximum is configured to be {}", max);
    }
    
    // Option methods
    let text: Option<String> = Some(String::from("Hello"));
    
    // map: Transform the value inside
    let length = text.as_ref().map(|s| s.len());
    println!("Length: {:?}", length);
    
    // unwrap_or: Provide default value
    let value = absent_number.unwrap_or(0);
    println!("Value with default: {}", value);
}

fn plus_one(x: Option<i32>) -> Option<i32> {
    match x {
        None => None,
        Some(i) => Some(i + 1),
    }
}
```

## Pattern Matching in Depth

```rust
// src/main.rs
enum Coin {
    Penny,
    Nickel,
    Dime,
    Quarter(UsState),
}

#[derive(Debug)]
enum UsState {
    Alabama,
    Alaska,
    // ... etc
}

fn value_in_cents(coin: Coin) -> u8 {
    match coin {
        Coin::Penny => {
            println!("Lucky penny!");
            1
        }
        Coin::Nickel => 5,
        Coin::Dime => 10,
        Coin::Quarter(state) => {
            println!("State quarter from {:?}", state);
            25
        }
    }
}

fn main() {
    let coin = Coin::Quarter(UsState::Alaska);
    println!("Value: {} cents", value_in_cents(coin));
    
    // Match must be exhaustive
    let dice_roll = 9;
    match dice_roll {
        3 => println!("Fancy hat!"),
        7 => println!("Fancy hat removed!"),
        _ => println!("Move {} spaces", dice_roll), // Catch-all pattern
    }
    
    // Match with guards
    let num = Some(4);
    match num {
        Some(x) if x < 5 => println!("less than five: {}", x),
        Some(x) => println!("{}", x),
        None => (),
    }
    
    // Destructuring in patterns
    struct Point {
        x: i32,
        y: i32,
    }
    
    let p = Point { x: 0, y: 7 };
    match p {
        Point { x: 0, y } => println!("On the y axis at {}", y),
        Point { x, y: 0 } => println!("On the x axis at {}", x),
        Point { x, y } => println!("On neither axis: ({}, {})", x, y),
    }
}
```

## Real-World Example: Domain Modeling

```rust
// src/main.rs
use std::fmt;

// Domain model for an e-commerce system
#[derive(Debug, Clone)]
struct Product {
    id: u64,
    name: String,
    price: f64,
    stock: u32,
}

#[derive(Debug)]
enum OrderStatus {
    Pending,
    Processing,
    Shipped { tracking_number: String },
    Delivered,
    Cancelled { reason: String },
}

#[derive(Debug)]
struct Order {
    id: u64,
    items: Vec<OrderItem>,
    status: OrderStatus,
    total: f64,
}

#[derive(Debug)]
struct OrderItem {
    product: Product,
    quantity: u32,
}

impl Product {
    fn new(id: u64, name: String, price: f64, stock: u32) -> Self {
        Self { id, name, price, stock }
    }
    
    fn is_available(&self, quantity: u32) -> bool {
        self.stock >= quantity
    }
    
    fn reduce_stock(&mut self, quantity: u32) -> Result<(), String> {
        if self.is_available(quantity) {
            self.stock -= quantity;
            Ok(())
        } else {
            Err(format!("Insufficient stock. Available: {}, Requested: {}", 
                       self.stock, quantity))
        }
    }
}

impl OrderItem {
    fn new(product: Product, quantity: u32) -> Self {
        Self { product, quantity }
    }
    
    fn subtotal(&self) -> f64 {
        self.product.price * self.quantity as f64
    }
}

impl Order {
    fn new(id: u64) -> Self {
        Self {
            id,
            items: Vec::new(),
            status: OrderStatus::Pending,
            total: 0.0,
        }
    }
    
    fn add_item(&mut self, item: OrderItem) {
        self.total += item.subtotal();
        self.items.push(item);
    }
    
    fn process(&mut self) -> Result<(), String> {
        match &self.status {
            OrderStatus::Pending => {
                self.status = OrderStatus::Processing;
                Ok(())
            }
            _ => Err("Order cannot be processed in current status".to_string()),
        }
    }
    
    fn ship(&mut self, tracking_number: String) -> Result<(), String> {
        match &self.status {
            OrderStatus::Processing => {
                self.status = OrderStatus::Shipped { tracking_number };
                Ok(())
            }
            _ => Err("Order must be processing to ship".to_string()),
        }
    }
    
    fn cancel(&mut self, reason: String) -> Result<(), String> {
        match &self.status {
            OrderStatus::Pending | OrderStatus::Processing => {
                self.status = OrderStatus::Cancelled { reason };
                Ok(())
            }
            _ => Err("Order cannot be cancelled in current status".to_string()),
        }
    }
}

impl fmt::Display for Order {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Order #{} - Status: {:?} - Total: ${:.2}", 
               self.id, self.status, self.total)
    }
}

fn main() {
    // Create products
    let mut laptop = Product::new(1, "Laptop".to_string(), 999.99, 10);
    let mut mouse = Product::new(2, "Mouse".to_string(), 29.99, 50);
    
    // Create order
    let mut order = Order::new(1001);
    
    // Add items to order
    if laptop.is_available(1) && mouse.is_available(2) {
        order.add_item(OrderItem::new(laptop.clone(), 1));
        order.add_item(OrderItem::new(mouse.clone(), 2));
        
        // Update stock
        laptop.reduce_stock(1).unwrap();
        mouse.reduce_stock(2).unwrap();
    }
    
    println!("{}", order);
    
    // Process order workflow
    order.process().unwrap();
    println!("After processing: {}", order);
    
    order.ship("TRACK123".to_string()).unwrap();
    println!("After shipping: {}", order);
    
    // Pattern match on status
    match &order.status {
        OrderStatus::Shipped { tracking_number } => {
            println!("Your order has been shipped! Track it with: {}", tracking_number);
        }
        _ => println!("Order status: {:?}", order.status),
    }
}
```

## Generic Structs and Enums

```rust
// src/main.rs
// Generic struct
#[derive(Debug)]
struct Point<T> {
    x: T,
    y: T,
}

// Multiple generic types
#[derive(Debug)]
struct Pair<T, U> {
    first: T,
    second: U,
}

// Generic enum (like Result<T, E>)
#[derive(Debug)]
enum MyResult<T, E> {
    Ok(T),
    Err(E),
}

impl<T> Point<T> {
    fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

// Method only for specific type
impl Point<f64> {
    fn distance_from_origin(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }
}

fn main() {
    // Using generic structs
    let integer_point = Point::new(5, 10);
    let float_point = Point::new(1.0, 4.0);
    
    println!("Integer point: {:?}", integer_point);
    println!("Float point: {:?}", float_point);
    println!("Distance: {}", float_point.distance_from_origin());
    
    // Mixed types
    let pair = Pair { first: 5, second: "hello" };
    println!("Pair: {:?}", pair);
    
    // Using generic enum
    let success: MyResult<i32, String> = MyResult::Ok(42);
    let failure: MyResult<i32, String> = MyResult::Err("Error occurred".to_string());
    
    match success {
        MyResult::Ok(value) => println!("Success: {}", value),
        MyResult::Err(e) => println!("Error: {}", e),
    }
}
```

## Exercises

1. **Struct Design**: Create a `Library` system with `Book`, `Member`, and `Loan` structs. Implement methods for borrowing and returning books.

2. **Enum Patterns**: Design a `FileSystem` enum that can represent files and directories. Implement methods to calculate size and count items.

3. **Pattern Matching**: Create a `Calculator` enum with operations (Add, Subtract, etc.) and implement an evaluate method using pattern matching.

4. **Domain Model**: Design a domain model for a task management system with `Task`, `Priority`, and `Status` types. Include methods for state transitions.

## Key Takeaways

- Structs group related data with named fields
- Enums represent a value that can be one of several variants
- Methods are defined in `impl` blocks
- Pattern matching with `match` must be exhaustive
- `Option<T>` replaces null for optional values
- Generics allow structs and enums to work with multiple types
- Domain modeling in Rust leverages the type system for safety

## Next Steps

Next, we'll explore **Error Handling** in Rust, learning how to use `Result<T, E>` effectively and build robust applications that handle failures gracefully.