# Tutorial 6: Collections & Iterators

## Vectors

Vectors are resizable arrays, the most common collection type.

```rust
// src/main.rs
fn main() {
    // Creating vectors
    let mut v1: Vec<i32> = Vec::new();
    let v2 = vec![1, 2, 3]; // Using the vec! macro
    
    // Adding elements
    v1.push(5);
    v1.push(6);
    v1.push(7);
    
    // Accessing elements
    let third: &i32 = &v2[2]; // Panics if out of bounds
    println!("The third element is {}", third);
    
    // Safe access with get()
    match v2.get(2) {
        Some(value) => println!("The third element is {}", value),
        None => println!("There is no third element"),
    }
    
    // Iterating
    for i in &v1 {
        println!("Value: {}", i);
    }
    
    // Iterating with mutable references
    for i in &mut v1 {
        *i += 50;
    }
    
    // Using enum to store different types
    #[derive(Debug)]
    enum SpreadsheetCell {
        Int(i32),
        Float(f64),
        Text(String),
    }
    
    let row = vec![
        SpreadsheetCell::Int(3),
        SpreadsheetCell::Text(String::from("blue")),
        SpreadsheetCell::Float(10.12),
    ];
    
    for cell in &row {
        println!("Cell: {:?}", cell);
    }
}
```

## Strings

Strings in Rust are UTF-8 encoded and growable.

```rust
// src/main.rs
fn main() {
    // Creating strings
    let mut s1 = String::new();
    let s2 = "initial contents".to_string();
    let s3 = String::from("initial contents");
    
    // Updating strings
    s1.push_str("foo");
    s1.push(' '); // Single character
    s1.push_str("bar");
    println!("s1: {}", s1);
    
    // Concatenation
    let s4 = String::from("Hello, ");
    let s5 = String::from("world!");
    let s6 = s4 + &s5; // s4 is moved here
    println!("s6: {}", s6);
    
    // Format macro (doesn't take ownership)
    let s7 = String::from("tic");
    let s8 = String::from("tac");
    let s9 = String::from("toe");
    let s10 = format!("{}-{}-{}", s7, s8, s9);
    println!("s10: {}", s10);
    
    // String slicing (be careful with UTF-8!)
    let hello = "Здравствуйте";
    let s = &hello[0..4]; // Takes first 4 bytes
    println!("Slice: {}", s);
    
    // Iterating over strings
    // By characters
    for c in "नमस्ते".chars() {
        println!("Char: {}", c);
    }
    
    // By bytes
    for b in "नमस्ते".bytes() {
        println!("Byte: {}", b);
    }
    
    // Useful string methods
    let text = "  hello world  ";
    println!("Trimmed: '{}'", text.trim());
    println!("Contains 'world': {}", text.contains("world"));
    println!("Replaced: '{}'", text.replace("world", "Rust"));
    
    // Splitting
    let words: Vec<&str> = text.split_whitespace().collect();
    println!("Words: {:?}", words);
}
```

## HashMaps

HashMaps store key-value pairs with O(1) average access time.

```rust
// src/main.rs
use std::collections::HashMap;

fn main() {
    // Creating HashMaps
    let mut scores = HashMap::new();
    
    // Inserting values
    scores.insert(String::from("Blue"), 10);
    scores.insert(String::from("Yellow"), 50);
    
    // Creating from iterators
    let teams = vec![String::from("Blue"), String::from("Yellow")];
    let initial_scores = vec![10, 50];
    let scores: HashMap<_, _> = teams.into_iter()
        .zip(initial_scores.into_iter())
        .collect();
    
    // Accessing values
    let team_name = String::from("Blue");
    let score = scores.get(&team_name);
    match score {
        Some(s) => println!("Score: {}", s),
        None => println!("No score found"),
    }
    
    // Iterating
    for (key, value) in &scores {
        println!("{}: {}", key, value);
    }
    
    // Updating values
    let mut scores = HashMap::new();
    scores.insert(String::from("Blue"), 10);
    
    // Overwriting
    scores.insert(String::from("Blue"), 25);
    
    // Only insert if key doesn't exist
    scores.entry(String::from("Yellow")).or_insert(50);
    scores.entry(String::from("Blue")).or_insert(50); // Won't overwrite
    
    println!("Scores: {:?}", scores);
    
    // Updating based on old value
    let text = "hello world wonderful world";
    let mut map = HashMap::new();
    
    for word in text.split_whitespace() {
        let count = map.entry(word).or_insert(0);
        *count += 1;
    }
    
    println!("Word count: {:?}", map);
}
```

## Other Collections

```rust
// src/main.rs
use std::collections::{HashSet, VecDeque, BinaryHeap};

fn main() {
    // HashSet - unique values only
    let mut books = HashSet::new();
    books.insert("The Great Gatsby");
    books.insert("To Kill a Mockingbird");
    books.insert("The Great Gatsby"); // Duplicate, won't be added
    
    println!("Number of unique books: {}", books.len());
    
    // Set operations
    let a: HashSet<i32> = [1, 2, 3].iter().cloned().collect();
    let b: HashSet<i32> = [2, 3, 4].iter().cloned().collect();
    
    let union: HashSet<_> = a.union(&b).cloned().collect();
    let intersection: HashSet<_> = a.intersection(&b).cloned().collect();
    let difference: HashSet<_> = a.difference(&b).cloned().collect();
    
    println!("Union: {:?}", union);
    println!("Intersection: {:?}", intersection);
    println!("Difference (a - b): {:?}", difference);
    
    // VecDeque - double-ended queue
    let mut deque = VecDeque::new();
    deque.push_back(1);
    deque.push_back(2);
    deque.push_front(0);
    
    println!("Deque: {:?}", deque);
    println!("Front: {:?}", deque.pop_front());
    println!("Back: {:?}", deque.pop_back());
    
    // BinaryHeap - priority queue (max-heap)
    let mut heap = BinaryHeap::new();
    heap.push(1);
    heap.push(5);
    heap.push(2);
    heap.push(3);
    
    // Elements come out in descending order
    while let Some(top) = heap.pop() {
        println!("Heap pop: {}", top);
    }
}
```

## Iterator Trait

```rust
// src/main.rs
// Understanding the Iterator trait
trait MyIterator {
    type Item;
    
    fn next(&mut self) -> Option<Self::Item>;
    
    // Many default methods like map, filter, etc.
}

// Custom iterator
struct Counter {
    count: u32,
    max: u32,
}

impl Counter {
    fn new(max: u32) -> Self {
        Counter { count: 0, max }
    }
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

fn main() {
    let counter = Counter::new(5);
    
    // Using iterator methods
    let sum: u32 = counter.sum();
    println!("Sum of 1 to 5: {}", sum);
    
    // Chaining iterator methods
    let counter = Counter::new(10);
    let even_squares: Vec<u32> = counter
        .filter(|x| x % 2 == 0)
        .map(|x| x * x)
        .collect();
    
    println!("Even squares: {:?}", even_squares);
}
```

## Iterator Methods

```rust
// src/main.rs
fn main() {
    let v = vec![1, 2, 3, 4, 5];
    
    // map - transform each element
    let v2: Vec<i32> = v.iter().map(|x| x + 1).collect();
    println!("Mapped: {:?}", v2);
    
    // filter - keep elements matching predicate
    let evens: Vec<&i32> = v.iter().filter(|x| *x % 2 == 0).collect();
    println!("Evens: {:?}", evens);
    
    // fold - reduce to single value
    let sum = v.iter().fold(0, |acc, x| acc + x);
    println!("Sum: {}", sum);
    
    // find - first element matching predicate
    let first_even = v.iter().find(|x| *x % 2 == 0);
    println!("First even: {:?}", first_even);
    
    // any/all - test predicates
    let has_negative = v.iter().any(|x| *x < 0);
    let all_positive = v.iter().all(|x| *x > 0);
    println!("Has negative: {}, All positive: {}", has_negative, all_positive);
    
    // zip - combine two iterators
    let v1 = vec![1, 2, 3];
    let v2 = vec!["a", "b", "c"];
    let zipped: Vec<_> = v1.iter().zip(v2.iter()).collect();
    println!("Zipped: {:?}", zipped);
    
    // enumerate - add index
    for (i, value) in v.iter().enumerate() {
        println!("Index {}: {}", i, value);
    }
    
    // skip/take
    let subset: Vec<&i32> = v.iter().skip(1).take(3).collect();
    println!("Skip 1, take 3: {:?}", subset);
    
    // partition - split into two collections
    let (evens, odds): (Vec<i32>, Vec<i32>) = v.into_iter()
        .partition(|x| x % 2 == 0);
    println!("Evens: {:?}, Odds: {:?}", evens, odds);
}
```

## Functional Programming with Iterators

```rust
// src/main.rs
#[derive(Debug)]
struct Person {
    name: String,
    age: u32,
}

fn main() {
    let people = vec![
        Person { name: String::from("Alice"), age: 25 },
        Person { name: String::from("Bob"), age: 30 },
        Person { name: String::from("Charlie"), age: 35 },
        Person { name: String::from("Diana"), age: 28 },
    ];
    
    // Complex iterator chains
    let names_over_30: Vec<String> = people
        .iter()
        .filter(|p| p.age > 30)
        .map(|p| p.name.clone())
        .collect();
    
    println!("People over 30: {:?}", names_over_30);
    
    // flat_map - flatten nested structures
    let sentences = vec![
        "Hello world",
        "Rust is awesome",
        "Iterators are powerful",
    ];
    
    let words: Vec<&str> = sentences
        .iter()
        .flat_map(|s| s.split_whitespace())
        .collect();
    
    println!("All words: {:?}", words);
    
    // Group by using fold
    use std::collections::HashMap;
    
    let age_groups: HashMap<String, Vec<&Person>> = people
        .iter()
        .fold(HashMap::new(), |mut acc, person| {
            let group = match person.age {
                0..=29 => "20s",
                30..=39 => "30s",
                _ => "40+",
            };
            acc.entry(group.to_string()).or_insert(Vec::new()).push(person);
            acc
        });
    
    for (group, people) in age_groups {
        println!("{}: {:?}", group, people.iter().map(|p| &p.name).collect::<Vec<_>>());
    }
}
```

## Real-World Example: Data Processing Pipeline

```rust
// src/main.rs
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct Transaction {
    id: u64,
    user_id: u64,
    amount: f64,
    category: String,
    timestamp: u64,
}

#[derive(Debug)]
struct Summary {
    total: f64,
    count: usize,
    average: f64,
    categories: HashMap<String, f64>,
}

struct TransactionProcessor {
    transactions: Vec<Transaction>,
}

impl TransactionProcessor {
    fn new(transactions: Vec<Transaction>) -> Self {
        Self { transactions }
    }
    
    // Filter transactions by time range
    fn in_time_range(&self, start: u64, end: u64) -> Vec<&Transaction> {
        self.transactions
            .iter()
            .filter(|t| t.timestamp >= start && t.timestamp <= end)
            .collect()
    }
    
    // Get transactions for a specific user
    fn for_user(&self, user_id: u64) -> Vec<&Transaction> {
        self.transactions
            .iter()
            .filter(|t| t.user_id == user_id)
            .collect()
    }
    
    // Calculate summary statistics
    fn summarize(&self, transactions: &[&Transaction]) -> Summary {
        let total = transactions.iter().map(|t| t.amount).sum();
        let count = transactions.len();
        let average = if count > 0 { total / count as f64 } else { 0.0 };
        
        let mut categories = HashMap::new();
        for transaction in transactions {
            *categories.entry(transaction.category.clone()).or_insert(0.0) += transaction.amount;
        }
        
        Summary {
            total,
            count,
            average,
            categories,
        }
    }
    
    // Find top spenders
    fn top_spenders(&self, n: usize) -> Vec<(u64, f64)> {
        let mut user_totals: HashMap<u64, f64> = HashMap::new();
        
        for transaction in &self.transactions {
            *user_totals.entry(transaction.user_id).or_insert(0.0) += transaction.amount;
        }
        
        let mut totals: Vec<(u64, f64)> = user_totals.into_iter().collect();
        totals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        totals.into_iter().take(n).collect()
    }
    
    // Advanced: Stream processing with iterator
    fn process_stream<F>(&self, mut processor: F)
    where
        F: FnMut(&Transaction),
    {
        self.transactions.iter().for_each(|t| processor(t));
    }
}

// Custom iterator for batching
struct BatchIterator<'a, T> {
    items: &'a [T],
    batch_size: usize,
    current: usize,
}

impl<'a, T> BatchIterator<'a, T> {
    fn new(items: &'a [T], batch_size: usize) -> Self {
        Self {
            items,
            batch_size,
            current: 0,
        }
    }
}

impl<'a, T> Iterator for BatchIterator<'a, T> {
    type Item = &'a [T];
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.items.len() {
            return None;
        }
        
        let start = self.current;
        let end = (self.current + self.batch_size).min(self.items.len());
        self.current = end;
        
        Some(&self.items[start..end])
    }
}

fn main() {
    // Sample data
    let transactions = vec![
        Transaction { id: 1, user_id: 100, amount: 50.0, category: "Food".to_string(), timestamp: 1000 },
        Transaction { id: 2, user_id: 101, amount: 120.0, category: "Electronics".to_string(), timestamp: 1100 },
        Transaction { id: 3, user_id: 100, amount: 30.0, category: "Food".to_string(), timestamp: 1200 },
        Transaction { id: 4, user_id: 102, amount: 200.0, category: "Travel".to_string(), timestamp: 1300 },
        Transaction { id: 5, user_id: 101, amount: 80.0, category: "Food".to_string(), timestamp: 1400 },
    ];
    
    let processor = TransactionProcessor::new(transactions.clone());
    
    // Time range analysis
    let recent = processor.in_time_range(1100, 1400);
    let summary = processor.summarize(&recent);
    println!("Recent transactions summary: {:?}", summary);
    
    // User analysis
    let user_100_transactions = processor.for_user(100);
    let user_summary = processor.summarize(&user_100_transactions);
    println!("User 100 summary: {:?}", user_summary);
    
    // Top spenders
    let top = processor.top_spenders(2);
    println!("Top 2 spenders: {:?}", top);
    
    // Stream processing
    let mut category_counter = HashMap::new();
    processor.process_stream(|t| {
        *category_counter.entry(t.category.clone()).or_insert(0) += 1;
    });
    println!("Category counts: {:?}", category_counter);
    
    // Batch processing
    let batch_iter = BatchIterator::new(&transactions, 2);
    for (i, batch) in batch_iter.enumerate() {
        println!("Batch {}: {} transactions", i, batch.len());
        for t in batch {
            println!("  Transaction {}: ${}", t.id, t.amount);
        }
    }
}
```

## Performance Considerations

```rust
// src/main.rs
use std::time::Instant;

fn main() {
    let data: Vec<i32> = (0..1_000_000).collect();
    
    // Iterator vs loop performance
    let start = Instant::now();
    let sum1: i32 = data.iter().sum();
    let iterator_time = start.elapsed();
    
    let start = Instant::now();
    let mut sum2 = 0;
    for &x in &data {
        sum2 += x;
    }
    let loop_time = start.elapsed();
    
    println!("Iterator sum: {} in {:?}", sum1, iterator_time);
    println!("Loop sum: {} in {:?}", sum2, loop_time);
    
    // Lazy evaluation
    let result: Vec<i32> = data.iter()
        .map(|&x| {
            // This computation only happens for taken elements
            x * 2
        })
        .filter(|&x| x % 4 == 0)
        .take(10)  // Only processes enough elements to get 10 results
        .collect();
    
    println!("First 10 results: {:?}", result);
    
    // Collecting into different types
    use std::collections::HashSet;
    
    let unique: HashSet<i32> = vec![1, 2, 2, 3, 3, 3]
        .into_iter()
        .collect();
    println!("Unique values: {:?}", unique);
    
    // extend vs collect
    let mut vec1 = vec![1, 2, 3];
    let vec2 = vec![4, 5, 6];
    
    // More efficient than collecting and appending
    vec1.extend(vec2.iter().cloned());
    println!("Extended vector: {:?}", vec1);
}
```

## Advanced Iterator Patterns

```rust
// src/main.rs
use std::collections::HashMap;

// Window iterator for sliding window operations
fn moving_average(data: &[f64], window_size: usize) -> Vec<f64> {
    data.windows(window_size)
        .map(|window| window.iter().sum::<f64>() / window_size as f64)
        .collect()
}

// Chunks for batch processing
fn process_in_chunks<T, F>(data: &[T], chunk_size: usize, mut processor: F)
where
    F: FnMut(&[T]),
{
    for chunk in data.chunks(chunk_size) {
        processor(chunk);
    }
}

// Custom iterator adapter
struct Unique<I: Iterator> {
    iter: I,
    seen: HashMap<I::Item, ()>,
}

impl<I> Unique<I>
where
    I: Iterator,
    I::Item: Eq + std::hash::Hash + Clone,
{
    fn new(iter: I) -> Self {
        Self {
            iter,
            seen: HashMap::new(),
        }
    }
}

impl<I> Iterator for Unique<I>
where
    I: Iterator,
    I::Item: Eq + std::hash::Hash + Clone,
{
    type Item = I::Item;
    
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(item) = self.iter.next() {
            if self.seen.insert(item.clone(), ()).is_none() {
                return Some(item);
            }
        }
        None
    }
}

// Extension trait for convenience
trait IteratorExt: Iterator {
    fn unique(self) -> Unique<Self>
    where
        Self: Sized,
        Self::Item: Eq + std::hash::Hash + Clone,
    {
        Unique::new(self)
    }
}

impl<I: Iterator> IteratorExt for I {}

fn main() {
    // Moving average
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let avg = moving_average(&data, 3);
    println!("Moving average: {:?}", avg);
    
    // Chunk processing
    let numbers = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    process_in_chunks(&numbers, 3, |chunk| {
        let sum: i32 = chunk.iter().sum();
        println!("Chunk sum: {}", sum);
    });
    
    // Using custom iterator
    let duplicates = vec![1, 2, 2, 3, 3, 3, 4, 1];
    let unique: Vec<i32> = duplicates.into_iter().unique().collect();
    println!("Unique values: {:?}", unique);
    
    // step_by for sampling
    let large_data: Vec<i32> = (0..100).collect();
    let sample: Vec<i32> = large_data.iter()
        .step_by(10)
        .cloned()
        .collect();
    println!("Every 10th element: {:?}", sample);
    
    // scan for running computation
    let numbers = vec![1, 2, 3, 4, 5];
    let running_sum: Vec<i32> = numbers
        .iter()
        .scan(0, |state, &x| {
            *state += x;
            Some(*state)
        })
        .collect();
    println!("Running sum: {:?}", running_sum);
}
```

## Collection Best Practices

```rust
// src/main.rs
use std::collections::{HashMap, BTreeMap};

#[derive(Debug)]
struct Cache<K, V> {
    map: HashMap<K, V>,
    max_size: usize,
}

impl<K: Eq + std::hash::Hash, V> Cache<K, V> {
    fn new(max_size: usize) -> Self {
        Self {
            map: HashMap::with_capacity(max_size),
            max_size,
        }
    }
    
    fn insert(&mut self, key: K, value: V) -> Option<V> {
        if self.map.len() >= self.max_size && !self.map.contains_key(&key) {
            // Simple eviction: remove first item (not LRU)
            if let Some(first_key) = self.map.keys().next().cloned() {
                self.map.remove(&first_key);
            }
        }
        self.map.insert(key, value)
    }
    
    fn get(&self, key: &K) -> Option<&V> {
        self.map.get(key)
    }
}

// Choosing the right collection
fn collection_examples() {
    // Vec for ordered, indexed access
    let mut list: Vec<i32> = Vec::new();
    list.push(1);
    list.push(2);
    
    // HashMap for fast key-value lookup
    let mut cache: HashMap<String, String> = HashMap::new();
    cache.insert("key".to_string(), "value".to_string());
    
    // BTreeMap for sorted keys
    let mut sorted: BTreeMap<i32, &str> = BTreeMap::new();
    sorted.insert(3, "three");
    sorted.insert(1, "one");
    sorted.insert(2, "two");
    
    for (k, v) in &sorted {
        println!("{}: {}", k, v); // Prints in sorted order
    }
    
    // String for owned, mutable text
    let mut text = String::from("Hello");
    text.push_str(", world!");
    
    // &str for borrowed text
    let slice: &str = &text[0..5];
    println!("Slice: {}", slice);
}

// Efficient collection operations
fn efficient_operations() {
    // Pre-allocate capacity
    let mut vec = Vec::with_capacity(1000);
    for i in 0..1000 {
        vec.push(i); // No reallocation needed
    }
    
    // Use entry API for HashMap
    let mut word_count: HashMap<&str, i32> = HashMap::new();
    let text = "the quick brown fox jumps over the lazy dog the fox";
    
    for word in text.split_whitespace() {
        word_count.entry(word).and_modify(|e| *e += 1).or_insert(1);
    }
    
    // Drain for moving elements
    let mut source = vec![1, 2, 3, 4, 5];
    let drained: Vec<i32> = source.drain(1..4).collect();
    println!("Source after drain: {:?}", source); // [1, 5]
    println!("Drained: {:?}", drained); // [2, 3, 4]
}

fn main() {
    // Using cache
    let mut cache = Cache::new(3);
    cache.insert("a", 1);
    cache.insert("b", 2);
    cache.insert("c", 3);
    cache.insert("d", 4); // Should evict one entry
    
    println!("Cache size: {}", cache.map.len());
    
    collection_examples();
    efficient_operations();
}
```

## Exercises

1. **Custom Collection**: Implement a `CircularBuffer<T>` that overwrites old elements when full. Include iterator support.

2. **Data Pipeline**: Create a data processing pipeline that reads CSV data, filters records, groups by category, and calculates statistics using iterator chains.

3. **Frequency Counter**: Build a generic frequency counter that can count occurrences of any hashable type and return the top N most frequent items.

4. **Iterator Combinator**: Implement a custom iterator combinator `intersperse` that inserts a separator between elements.

5. **Performance Comparison**: Compare the performance of different collection types (Vec, LinkedList, VecDeque) for various operations (push, pop, insert at middle).

## Key Takeaways

- Vectors are the go-to collection for ordered data
- Strings are UTF-8 encoded and require careful handling
- HashMap provides O(1) average-case lookup
- Iterators enable functional programming patterns
- Iterator chains are lazy and efficient
- Choose collections based on access patterns
- Pre-allocate capacity when size is known
- Use entry API for HashMap updates
- Iterator adapters can be composed for complex operations

## Next Steps

In the next tutorial, we'll explore **Modules and Packages**, learning how to organize code into reusable components and create libraries that can be shared across projects.