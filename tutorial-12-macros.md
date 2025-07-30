# Tutorial 12: Macros & Metaprogramming

## Declarative Macros (macro_rules!)

Declarative macros allow pattern matching on Rust syntax and code generation.

```rust
// src/main.rs

// Basic macro
macro_rules! say_hello {
    () => {
        println!("Hello, macro world!");
    };
}

// Macro with parameters
macro_rules! create_function {
    ($func_name:ident) => {
        fn $func_name() {
            println!("You called {:?}()", stringify!($func_name));
        }
    };
}

// Macro with multiple patterns
macro_rules! calculate {
    (add $x:expr, $y:expr) => {
        $x + $y
    };
    (multiply $x:expr, $y:expr) => {
        $x * $y
    };
    (square $x:expr) => {
        $x * $x
    };
}

// Variadic macro (accepting multiple arguments)
macro_rules! vec_of_strings {
    ($($x:expr),*) => {
        vec![$(String::from($x)),*]
    };
}

// Macro with repetition
macro_rules! create_struct {
    ($struct_name:ident { $($field_name:ident: $field_type:ty),* }) => {
        #[derive(Debug)]
        struct $struct_name {
            $($field_name: $field_type),*
        }
    };
}

fn main() {
    // Using basic macro
    say_hello!();
    
    // Creating functions with macros
    create_function!(foo);
    create_function!(bar);
    
    foo();
    bar();
    
    // Using calculate macro
    let sum = calculate!(add 5, 3);
    let product = calculate!(multiply 4, 7);
    let square = calculate!(square 6);
    
    println!("Sum: {}, Product: {}, Square: {}", sum, product, square);
    
    // Variadic macro
    let strings = vec_of_strings!["hello", "world", "from", "macro"];
    println!("Strings: {:?}", strings);
    
    // Creating struct with macro
    create_struct!(Person {
        name: String,
        age: u32,
        email: String
    });
    
    let person = Person {
        name: String::from("Alice"),
        age: 30,
        email: String::from("alice@example.com"),
    };
    
    println!("Person: {:?}", person);
}
```

## Advanced Macro Patterns

```rust
// src/main.rs

// Macro with different argument types
macro_rules! print_type {
    ($x:expr) => {
        println!("{} is {}", stringify!($x), type_name_of(&$x));
    };
}

fn type_name_of<T>(_: &T) -> &'static str {
    std::any::type_name::<T>()
}

// Recursive macros
macro_rules! count_items {
    () => (0);
    ($head:expr) => (1);
    ($head:expr, $($tail:expr),+) => (1 + count_items!($($tail),+));
}

// Pattern matching in macros
macro_rules! match_type {
    ($value:expr, {
        $($pattern:pat => $result:expr),+
        $(, _ => $default:expr)?
    }) => {
        match $value {
            $($pattern => $result),+
            $(, _ => $default)?
        }
    };
}

// Building complex expressions
macro_rules! build_map {
    ($($key:expr => $value:expr),*) => {
        {
            let mut map = std::collections::HashMap::new();
            $(
                map.insert($key, $value);
            )*
            map
        }
    };
}

// TT muncher pattern (token tree muncher)
macro_rules! replace_expr {
    ($_t:tt $sub:expr) => {$sub};
}

macro_rules! count_tts {
    () => {0};
    ($head:tt $($tail:tt)*) => {1 + count_tts!($($tail)*)};
}

// Macro generating macros
macro_rules! make_public {
    ($item:item) => {
        pub $item
    };
}

// DSL-like macro
macro_rules! html {
    (div { $($content:tt)* }) => {
        format!("<div>{}</div>", html!($($content)*))
    };
    (p { $($content:tt)* }) => {
        format!("<p>{}</p>", html!($($content)*))
    };
    ($text:literal) => {
        $text
    };
    ($($tag:ident { $($content:tt)* })*) => {
        {
            let mut result = String::new();
            $(
                result.push_str(&html!($tag { $($content)* }));
            )*
            result
        }
    };
}

fn main() {
    // Type printing
    print_type!(42);
    print_type!("hello");
    print_type!(vec![1, 2, 3]);
    
    // Counting items
    let count = count_items!(1, 2, 3, 4, 5);
    println!("Count: {}", count);
    
    // Pattern matching
    let result = match_type!(5, {
        0 => "zero",
        1..=10 => "one to ten",
        _ => "something else"
    });
    println!("Result: {}", result);
    
    // Building HashMap
    let map = build_map! {
        "one" => 1,
        "two" => 2,
        "three" => 3
    };
    println!("Map: {:?}", map);
    
    // TT muncher
    let value = replace_expr!((ignored) "actual value");
    println!("Value: {}", value);
    
    let tt_count = count_tts!(a b c d e);
    println!("Token count: {}", tt_count);
    
    // HTML DSL
    let html = html! {
        div {
            p { "Hello, world!" }
            p { "This is generated HTML" }
        }
    };
    println!("HTML: {}", html);
}

// Using macro to make items public
make_public! {
    struct PublicStruct {
        field: i32,
    }
}
```

## Procedural Macros

Procedural macros run code at compile time to generate code.

```rust
// In a separate crate (proc-macro crate)
// Cargo.toml:
// [package]
// name = "my_macros"
// version = "0.1.0"
// edition = "2021"
//
// [lib]
// proc-macro = true
//
// [dependencies]
// syn = "2.0"
// quote = "1.0"
// proc-macro2 = "1.0"

// src/lib.rs
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Data, Fields};

// Derive macro
#[proc_macro_derive(Builder)]
pub fn derive_builder(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let builder_name = syn::Ident::new(
        &format!("{}Builder", name),
        name.span()
    );
    
    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => panic!("Builder only supports structs with named fields"),
        },
        _ => panic!("Builder only supports structs"),
    };
    
    let field_names: Vec<_> = fields.iter()
        .map(|f| &f.ident)
        .collect();
    
    let field_types: Vec<_> = fields.iter()
        .map(|f| &f.ty)
        .collect();
    
    let builder_fields = field_names.iter().zip(field_types.iter())
        .map(|(name, ty)| {
            quote! {
                #name: Option<#ty>
            }
        });
    
    let builder_methods = field_names.iter().zip(field_types.iter())
        .map(|(name, ty)| {
            quote! {
                pub fn #name(mut self, value: #ty) -> Self {
                    self.#name = Some(value);
                    self
                }
            }
        });
    
    let build_fields = field_names.iter()
        .map(|name| {
            quote! {
                #name: self.#name.ok_or(concat!(stringify!(#name), " is required"))?
            }
        });
    
    let expanded = quote! {
        pub struct #builder_name {
            #(#builder_fields),*
        }
        
        impl #builder_name {
            pub fn new() -> Self {
                Self {
                    #(#field_names: None),*
                }
            }
            
            #(#builder_methods)*
            
            pub fn build(self) -> Result<#name, &'static str> {
                Ok(#name {
                    #(#build_fields),*
                })
            }
        }
        
        impl #name {
            pub fn builder() -> #builder_name {
                #builder_name::new()
            }
        }
    };
    
    TokenStream::from(expanded)
}

// Attribute macro
#[proc_macro_attribute]
pub fn timed(args: TokenStream, input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::ItemFn);
    let name = &input.sig.ident;
    let body = &input.block;
    let attrs = &input.attrs;
    let vis = &input.vis;
    let sig = &input.sig;
    
    let result = quote! {
        #(#attrs)*
        #vis #sig {
            let start = std::time::Instant::now();
            let result = #body;
            let duration = start.elapsed();
            println!("{} took {:?}", stringify!(#name), duration);
            result
        }
    };
    
    TokenStream::from(result)
}

// Function-like procedural macro
#[proc_macro]
pub fn sql(input: TokenStream) -> TokenStream {
    let input = input.to_string();
    let query = input.trim_matches('"');
    
    // In real implementation, you'd parse and validate SQL
    let result = quote! {
        {
            // This would connect to DB and execute query
            println!("Executing SQL: {}", #query);
            format!("Result of: {}", #query)
        }
    };
    
    TokenStream::from(result)
}
```

## Using Procedural Macros

```rust
// src/main.rs
// Using the procedural macros defined above

use my_macros::{Builder, timed, sql};

#[derive(Debug, Builder)]
struct User {
    id: u64,
    name: String,
    email: String,
    age: u32,
}

#[timed]
fn expensive_operation() -> u64 {
    let mut sum = 0;
    for i in 0..1_000_000 {
        sum += i;
    }
    sum
}

#[timed]
async fn async_operation() -> String {
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    String::from("Async result")
}

fn main() {
    // Using derive macro
    let user = User::builder()
        .id(1)
        .name(String::from("Alice"))
        .email(String::from("alice@example.com"))
        .age(30)
        .build()
        .unwrap();
    
    println!("User: {:?}", user);
    
    // Using attribute macro
    let result = expensive_operation();
    println!("Result: {}", result);
    
    // Using function-like macro
    let query_result = sql!("SELECT * FROM users WHERE age > 25");
    println!("Query result: {}", query_result);
}
```

## Real-World Macro Examples

```rust
// src/main.rs

// Configuration DSL
macro_rules! config {
    (
        $(
            $section:ident {
                $($key:ident : $value:expr),* $(,)?
            }
        )*
    ) => {
        {
            #[derive(Debug)]
            struct Config {
                $(
                    $section: config!(@section $section { $($key : $value),* }),
                )*
            }
            
            Config {
                $(
                    $section: config!(@section $section { $($key : $value),* }),
                )*
            }
        }
    };
    
    (@section $section:ident { $($key:ident : $value:expr),* }) => {
        {
            #[derive(Debug)]
            struct $section {
                $($key: config!(@type $value)),*
            }
            
            $section {
                $($key: $value),*
            }
        }
    };
    
    (@type $value:expr) => {
        std::convert::Into::into($value)
    };
}

// Test assertion macro
macro_rules! assert_matches {
    ($expression:expr, $pattern:pat $(if $guard:expr)? $(,)?) => {
        match $expression {
            $pattern $(if $guard)? => (),
            ref e => panic!(
                "assertion failed: `{:?}` does not match `{}`",
                e,
                stringify!($pattern $(if $guard)?)
            ),
        }
    };
}

// Lazy static initialization
macro_rules! lazy_static {
    ($(
        static ref $name:ident : $type:ty = $init:expr;
    )*) => {
        $(
            static $name: std::sync::OnceLock<$type> = std::sync::OnceLock::new();
            
            impl std::ops::Deref for $name {
                type Target = $type;
                
                fn deref(&self) -> &Self::Target {
                    self.get_or_init(|| $init)
                }
            }
        )*
    };
}

// Bitflags macro
macro_rules! bitflags {
    (
        struct $name:ident: $type:ty {
            $(const $flag:ident = $value:expr;)*
        }
    ) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        struct $name($type);
        
        impl $name {
            $(const $flag: Self = Self($value);)*
            
            pub fn contains(&self, other: Self) -> bool {
                (self.0 & other.0) == other.0
            }
            
            pub fn insert(&mut self, other: Self) {
                self.0 |= other.0;
            }
            
            pub fn remove(&mut self, other: Self) {
                self.0 &= !other.0;
            }
        }
        
        impl std::ops::BitOr for $name {
            type Output = Self;
            
            fn bitor(self, rhs: Self) -> Self::Output {
                Self(self.0 | rhs.0)
            }
        }
    };
}

// Enum dispatch macro
macro_rules! enum_dispatch {
    (
        enum $name:ident {
            $($variant:ident($type:ty)),* $(,)?
        }
        
        impl $trait:ident {
            $(fn $method:ident(&self $(, $arg:ident: $arg_type:ty)*) $(-> $ret:ty)?;)*
        }
    ) => {
        enum $name {
            $($variant($type)),*
        }
        
        impl $trait for $name {
            $(
                fn $method(&self $(, $arg: $arg_type)*) $(-> $ret)? {
                    match self {
                        $(Self::$variant(inner) => inner.$method($($arg),*)),*
                    }
                }
            )*
        }
    };
}

// Pipeline macro
macro_rules! pipe {
    ($value:expr => $($func:expr),+ $(,)?) => {
        {
            let mut result = $value;
            $(
                result = $func(result);
            )+
            result
        }
    };
}

fn main() {
    // Using config DSL
    let config = config! {
        server {
            host: "localhost",
            port: 8080,
            workers: 4,
        }
        database {
            url: "postgres://localhost/mydb",
            pool_size: 10,
        }
    };
    
    println!("Config: {:?}", config);
    
    // Using assert_matches
    let value = Some(42);
    assert_matches!(value, Some(x) if x > 40);
    
    // Using bitflags
    bitflags! {
        struct Permissions: u32 {
            const READ = 1 << 0;
            const WRITE = 1 << 1;
            const EXECUTE = 1 << 2;
        }
    }
    
    let mut perms = Permissions::READ | Permissions::WRITE;
    println!("Permissions: {:?}", perms);
    println!("Can execute: {}", perms.contains(Permissions::EXECUTE));
    
    perms.insert(Permissions::EXECUTE);
    println!("After insert: {:?}", perms);
    
    // Using pipeline
    let result = pipe! {
        vec![1, 2, 3, 4, 5]
        => |v: Vec<i32>| v.into_iter().map(|x| x * 2).collect::<Vec<_>>()
        => |v: Vec<i32>| v.into_iter().filter(|x| x % 3 == 0).collect::<Vec<_>>()
        => |v: Vec<i32>| v.into_iter().sum::<i32>()
    };
    
    println!("Pipeline result: {}", result);
}
```

## Macro Hygiene and Best Practices

```rust
// src/main.rs

// Hygienic macro - variables don't leak
macro_rules! hygienic {
    ($x:expr) => {
        {
            let temp = $x;
            temp * 2
        }
    };
}

// Using special identifiers to avoid conflicts
macro_rules! with_mutex {
    ($mutex:expr, $body:block) => {
        {
            let __guard = $mutex.lock().unwrap();
            let result = $body;
            drop(__guard);
            result
        }
    };
}

// Macro with proper error handling
macro_rules! try_or_return {
    ($expr:expr, $err:expr) => {
        match $expr {
            Ok(val) => val,
            Err(_) => return Err($err),
        }
    };
}

// Macro generating documentation
macro_rules! define_constants {
    ($(
        $(#[$meta:meta])*
        $name:ident = $value:expr;
    )*) => {
        $(
            $(#[$meta])*
            pub const $name: u32 = $value;
        )*
    };
}

// Debugging macro
macro_rules! dbg_vars {
    ($($var:ident),* $(,)?) => {
        {
            eprintln!("[{}:{}]", file!(), line!());
            $(
                eprintln!("  {} = {:?}", stringify!($var), $var);
            )*
        }
    };
}

// Compile-time assertions
macro_rules! const_assert {
    ($condition:expr) => {
        const _: () = assert!($condition);
    };
}

// Feature-gated macro
macro_rules! debug_only {
    ($($body:tt)*) => {
        #[cfg(debug_assertions)]
        {
            $($body)*
        }
    };
}

// Main function demonstrating usage
fn main() {
    // Hygienic macro
    let temp = 5;
    let result = hygienic!(10);
    println!("temp: {}, result: {}", temp, result); // temp is not affected
    
    // Mutex helper
    use std::sync::Mutex;
    let data = Mutex::new(vec![1, 2, 3]);
    
    let sum = with_mutex!(data, {
        data.iter().sum::<i32>()
    });
    println!("Sum: {}", sum);
    
    // Constants with documentation
    define_constants! {
        /// Maximum number of connections
        MAX_CONNECTIONS = 100;
        
        /// Default timeout in seconds
        DEFAULT_TIMEOUT = 30;
    }
    
    println!("Max connections: {}", MAX_CONNECTIONS);
    
    // Debug variables
    let x = 42;
    let y = "hello";
    let z = vec![1, 2, 3];
    
    debug_only! {
        dbg_vars!(x, y, z);
    }
    
    // Compile-time assertion
    const_assert!(std::mem::size_of::<usize>() >= 4);
    
    println!("All macros executed successfully!");
}

// Advanced: Macro to implement a trait for multiple types
macro_rules! impl_display_for {
    ($($type:ty),* $(,)?) => {
        $(
            impl std::fmt::Display for $type {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, "{:?}", self)
                }
            }
        )*
    };
}

#[derive(Debug)]
struct Point { x: i32, y: i32 }

#[derive(Debug)]
struct Color { r: u8, g: u8, b: u8 }

impl_display_for!(Point, Color);
```

## Exercises

1. **Custom Derive Macro**: Create a derive macro `Validate` that generates a `validate()` method checking field constraints.

2. **DSL Macro**: Build a macro-based DSL for defining state machines with transitions and guards.

3. **Test Framework**: Implement a simple test framework using macros that supports test registration and custom assertions.

4. **Code Generator**: Create a macro that generates CRUD operations for a struct (create, read, update, delete methods).

5. **Async Macro**: Write a macro that transforms synchronous code into async code by wrapping operations in async blocks.

## Key Takeaways

- Declarative macros use pattern matching and are defined with `macro_rules!`
- Procedural macros run Rust code at compile time to generate code
- Three types of proc macros: derive, attribute, and function-like
- Macros are hygienic by default (variables don't leak)
- Use `$crate` for macro imports to avoid naming conflicts
- Macros can create DSLs for more expressive code
- Token tree munchers process tokens recursively
- Procedural macros require a separate crate with `proc-macro = true`
- Macros are expanded before type checking
- Use macros to reduce boilerplate and enforce patterns

## Next Steps

In Tutorial 13, we'll explore **Database Integration**, learning how to work with PostgreSQL using diesel and sqlx, which aligns with your database-focused work.