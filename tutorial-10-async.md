# Tutorial 10: Async Programming

## Understanding Async/Await

Async programming allows concurrent operations without the overhead of OS threads.

```rust
// Cargo.toml dependencies
// [dependencies]
// tokio = { version = "1", features = ["full"] }
// futures = "0.3"

// src/main.rs
use std::time::Duration;
use tokio::time::sleep;

// Async function
async fn hello_world() {
    println!("Hello");
    sleep(Duration::from_secs(1)).await;
    println!("World!");
}

// Async functions return Futures
async fn compute_value() -> i32 {
    sleep(Duration::from_millis(100)).await;
    42
}

#[tokio::main]
async fn main() {
    // Calling async functions
    hello_world().await;
    
    let value = compute_value().await;
    println!("Computed value: {}", value);
    
    // Running multiple async operations concurrently
    let future1 = async {
        sleep(Duration::from_secs(1)).await;
        println!("Future 1 complete");
        1
    };
    
    let future2 = async {
        sleep(Duration::from_millis(500)).await;
        println!("Future 2 complete");
        2
    };
    
    // Join futures
    let (result1, result2) = tokio::join!(future1, future2);
    println!("Results: {} and {}", result1, result2);
}
```

## Futures and Tasks

```rust
// src/main.rs
use futures::future::{self, FutureExt};
use tokio::task;
use std::time::Duration;

async fn task_example(id: u64) -> String {
    println!("Task {} started", id);
    tokio::time::sleep(Duration::from_millis(100 * id)).await;
    format!("Task {} completed", id)
}

#[tokio::main]
async fn main() {
    // Spawning tasks
    let handle = task::spawn(async {
        println!("Running in a separate task");
        tokio::time::sleep(Duration::from_secs(1)).await;
        "Task result"
    });
    
    let result = handle.await.unwrap();
    println!("Task returned: {}", result);
    
    // Multiple tasks
    let mut handles = vec![];
    
    for i in 0..5 {
        let handle = task::spawn(task_example(i));
        handles.push(handle);
    }
    
    // Wait for all tasks
    for handle in handles {
        let result = handle.await.unwrap();
        println!("{}", result);
    }
    
    // Select - first future to complete wins
    let future1 = async {
        tokio::time::sleep(Duration::from_millis(200)).await;
        "Future 1"
    };
    
    let future2 = async {
        tokio::time::sleep(Duration::from_millis(100)).await;
        "Future 2"
    };
    
    let result = tokio::select! {
        val = future1 => val,
        val = future2 => val,
    };
    
    println!("First to complete: {}", result);
    
    // Timeout
    let slow_operation = async {
        tokio::time::sleep(Duration::from_secs(5)).await;
        "Finally done"
    };
    
    match tokio::time::timeout(Duration::from_secs(1), slow_operation).await {
        Ok(result) => println!("Operation completed: {}", result),
        Err(_) => println!("Operation timed out"),
    }
}
```

## Async I/O

```rust
// src/main.rs
use tokio::fs::File;
use tokio::io::{self, AsyncReadExt, AsyncWriteExt, AsyncBufReadExt, BufReader};
use tokio::net::{TcpListener, TcpStream};

async fn file_operations() -> io::Result<()> {
    // Writing to a file
    let mut file = File::create("async_test.txt").await?;
    file.write_all(b"Hello from async Rust!\n").await?;
    file.write_all(b"This is line 2\n").await?;
    file.sync_all().await?;
    
    // Reading from a file
    let mut file = File::open("async_test.txt").await?;
    let mut contents = String::new();
    file.read_to_string(&mut contents).await?;
    println!("File contents:\n{}", contents);
    
    // Reading line by line
    let file = File::open("async_test.txt").await?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    
    while let Some(line) = lines.next_line().await? {
        println!("Line: {}", line);
    }
    
    // Cleanup
    tokio::fs::remove_file("async_test.txt").await?;
    
    Ok(())
}

async fn handle_connection(mut stream: TcpStream, id: usize) -> io::Result<()> {
    println!("Client {} connected", id);
    
    let mut buffer = [0; 1024];
    
    loop {
        let n = stream.read(&mut buffer).await?;
        if n == 0 {
            println!("Client {} disconnected", id);
            return Ok(());
        }
        
        // Echo back
        stream.write_all(&buffer[..n]).await?;
        stream.flush().await?;
    }
}

async fn tcp_server() -> io::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:8080").await?;
    println!("Server listening on 127.0.0.1:8080");
    
    let mut client_id = 0;
    
    loop {
        let (stream, addr) = listener.accept().await?;
        println!("New connection from {}", addr);
        
        client_id += 1;
        tokio::spawn(handle_connection(stream, client_id));
    }
}

#[tokio::main]
async fn main() -> io::Result<()> {
    // File operations
    file_operations().await?;
    
    // Note: TCP server example would run forever
    // Uncomment to test:
    // tcp_server().await?;
    
    Ok(())
}
```

## Channels and Message Passing

```rust
// src/main.rs
use tokio::sync::{mpsc, oneshot, broadcast};
use tokio::time::{sleep, Duration};

#[derive(Debug, Clone)]
struct Message {
    id: u64,
    content: String,
}

async fn mpsc_example() {
    // Multi-producer, single-consumer channel
    let (tx, mut rx) = mpsc::channel::<Message>(100);
    
    // Spawn multiple producers
    for i in 0..3 {
        let tx = tx.clone();
        tokio::spawn(async move {
            for j in 0..3 {
                let msg = Message {
                    id: i * 10 + j,
                    content: format!("Message from producer {}", i),
                };
                tx.send(msg).await.unwrap();
                sleep(Duration::from_millis(100)).await;
            }
        });
    }
    
    // Drop original sender so receiver knows when all producers are done
    drop(tx);
    
    // Consumer
    while let Some(msg) = rx.recv().await {
        println!("Received: {:?}", msg);
    }
}

async fn oneshot_example() {
    // One-shot channel for single value
    let (tx, rx) = oneshot::channel::<String>();
    
    tokio::spawn(async move {
        sleep(Duration::from_millis(100)).await;
        tx.send("Computation result".to_string()).unwrap();
    });
    
    match rx.await {
        Ok(value) => println!("Received oneshot: {}", value),
        Err(_) => println!("Sender dropped"),
    }
}

async fn broadcast_example() {
    // Broadcast channel - multiple receivers
    let (tx, _) = broadcast::channel::<Message>(16);
    
    // Create multiple receivers
    let mut rx1 = tx.subscribe();
    let mut rx2 = tx.subscribe();
    
    // Receiver 1
    tokio::spawn(async move {
        while let Ok(msg) = rx1.recv().await {
            println!("Receiver 1 got: {:?}", msg);
        }
    });
    
    // Receiver 2
    tokio::spawn(async move {
        while let Ok(msg) = rx2.recv().await {
            println!("Receiver 2 got: {:?}", msg);
        }
    });
    
    // Send messages
    for i in 0..5 {
        let msg = Message {
            id: i,
            content: format!("Broadcast message {}", i),
        };
        tx.send(msg).unwrap();
        sleep(Duration::from_millis(50)).await;
    }
    
    sleep(Duration::from_millis(100)).await;
}

#[tokio::main]
async fn main() {
    println!("=== MPSC Example ===");
    mpsc_example().await;
    
    println!("\n=== Oneshot Example ===");
    oneshot_example().await;
    
    println!("\n=== Broadcast Example ===");
    broadcast_example().await;
}
```

## Async Synchronization

```rust
// src/main.rs
use tokio::sync::{Mutex, RwLock, Semaphore, Notify};
use std::sync::Arc;
use tokio::time::{sleep, Duration};

async fn mutex_example() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];
    
    for i in 0..5 {
        let counter = Arc::clone(&counter);
        let handle = tokio::spawn(async move {
            for _ in 0..100 {
                let mut num = counter.lock().await;
                *num += 1;
                // Lock is automatically released here
            }
            println!("Task {} completed", i);
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.await.unwrap();
    }
    
    println!("Final counter: {}", *counter.lock().await);
}

async fn rwlock_example() {
    let data = Arc::new(RwLock::new(vec![1, 2, 3]));
    
    // Multiple readers
    let mut reader_handles = vec![];
    for i in 0..3 {
        let data = Arc::clone(&data);
        let handle = tokio::spawn(async move {
            let values = data.read().await;
            println!("Reader {} sees: {:?}", i, *values);
            sleep(Duration::from_millis(100)).await;
        });
        reader_handles.push(handle);
    }
    
    // One writer
    let data_clone = Arc::clone(&data);
    let writer = tokio::spawn(async move {
        sleep(Duration::from_millis(50)).await;
        let mut values = data_clone.write().await;
        values.push(4);
        println!("Writer added 4");
    });
    
    for handle in reader_handles {
        handle.await.unwrap();
    }
    writer.await.unwrap();
    
    println!("Final data: {:?}", *data.read().await);
}

async fn semaphore_example() {
    // Limit concurrent access
    let semaphore = Arc::new(Semaphore::new(3));
    let mut handles = vec![];
    
    for i in 0..10 {
        let permit = semaphore.clone();
        let handle = tokio::spawn(async move {
            let _permit = permit.acquire().await.unwrap();
            println!("Task {} has permit", i);
            sleep(Duration::from_millis(100)).await;
            println!("Task {} releasing permit", i);
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.await.unwrap();
    }
}

async fn notify_example() {
    let notify = Arc::new(Notify::new());
    let notify2 = notify.clone();
    
    tokio::spawn(async move {
        sleep(Duration::from_millis(100)).await;
        println!("Sending notification");
        notify2.notify_one();
    });
    
    println!("Waiting for notification");
    notify.notified().await;
    println!("Received notification!");
}

#[tokio::main]
async fn main() {
    println!("=== Mutex Example ===");
    mutex_example().await;
    
    println!("\n=== RwLock Example ===");
    rwlock_example().await;
    
    println!("\n=== Semaphore Example ===");
    semaphore_example().await;
    
    println!("\n=== Notify Example ===");
    notify_example().await;
}
```

## Real-World Example: Async Web Scraper

```rust
// src/main.rs
use futures::stream::{self, StreamExt};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, Semaphore};
use tokio::time::{sleep, Duration, Instant};

#[derive(Debug, Clone)]
struct Page {
    url: String,
    content: String,
}

#[derive(Debug)]
struct ScraperStats {
    pages_scraped: usize,
    total_bytes: usize,
    errors: usize,
}

struct WebScraper {
    client: reqwest::Client,
    semaphore: Arc<Semaphore>,
    stats: Arc<Mutex<ScraperStats>>,
}

impl WebScraper {
    fn new(max_concurrent: usize) -> Self {
        WebScraper {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap(),
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            stats: Arc::new(Mutex::new(ScraperStats {
                pages_scraped: 0,
                total_bytes: 0,
                errors: 0,
            })),
        }
    }
    
    async fn fetch_page(&self, url: String) -> Result<Page, String> {
        let _permit = self.semaphore.acquire().await.unwrap();
        
        println!("Fetching: {}", url);
        
        // Simulate network request (replace with actual HTTP request)
        sleep(Duration::from_millis(100)).await;
        
        // Simulated response
        let content = format!("Content of {} (simulated)", url);
        
        Ok(Page {
            url,
            content,
        })
    }
    
    async fn scrape_pages(&self, urls: Vec<String>) -> Vec<Page> {
        let futures = urls.into_iter().map(|url| {
            let scraper = self.clone();
            async move {
                match scraper.fetch_page(url).await {
                    Ok(page) => {
                        let mut stats = scraper.stats.lock().await;
                        stats.pages_scraped += 1;
                        stats.total_bytes += page.content.len();
                        Some(page)
                    }
                    Err(e) => {
                        eprintln!("Error fetching page: {}", e);
                        let mut stats = scraper.stats.lock().await;
                        stats.errors += 1;
                        None
                    }
                }
            }
        });
        
        stream::iter(futures)
            .buffer_unordered(10)
            .filter_map(|x| async { x })
            .collect()
            .await
    }
    
    async fn get_stats(&self) -> ScraperStats {
        let stats = self.stats.lock().await;
        ScraperStats {
            pages_scraped: stats.pages_scraped,
            total_bytes: stats.total_bytes,
            errors: stats.errors,
        }
    }
}

impl Clone for WebScraper {
    fn clone(&self) -> Self {
        WebScraper {
            client: self.client.clone(),
            semaphore: Arc::clone(&self.semaphore),
            stats: Arc::clone(&self.stats),
        }
    }
}

// Async task processing pipeline
struct TaskPipeline<T> {
    sender: tokio::sync::mpsc::Sender<T>,
}

impl<T: Send + 'static> TaskPipeline<T> {
    fn new<F, Fut>(buffer: usize, handler: F) -> Self
    where
        F: Fn(T) -> Fut + Send + 'static + Clone,
        Fut: std::future::Future<Output = ()> + Send,
    {
        let (sender, mut receiver) = tokio::sync::mpsc::channel(buffer);
        
        tokio::spawn(async move {
            while let Some(item) = receiver.recv().await {
                let handler = handler.clone();
                tokio::spawn(handler(item));
            }
        });
        
        TaskPipeline { sender }
    }
    
    async fn submit(&self, item: T) -> Result<(), String> {
        self.sender
            .send(item)
            .await
            .map_err(|_| "Pipeline closed".to_string())
    }
}

// Rate limiter
struct RateLimiter {
    semaphore: Arc<Semaphore>,
    interval: Duration,
}

impl RateLimiter {
    fn new(rate: usize, interval: Duration) -> Self {
        let semaphore = Arc::new(Semaphore::new(rate));
        let sem_clone = Arc::clone(&semaphore);
        
        // Replenish permits periodically
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(interval);
            loop {
                interval.tick().await;
                // Reset to full capacity
                while sem_clone.available_permits() < rate {
                    sem_clone.add_permits(1);
                }
            }
        });
        
        RateLimiter { semaphore, interval }
    }
    
    async fn acquire(&self) {
        let _permit = self.semaphore.acquire().await.unwrap();
    }
}

#[tokio::main]
async fn main() {
    // Web scraper example
    let scraper = WebScraper::new(3);
    
    let urls: Vec<String> = (1..=20)
        .map(|i| format!("https://example.com/page{}", i))
        .collect();
    
    let start = Instant::now();
    let pages = scraper.scrape_pages(urls).await;
    let duration = start.elapsed();
    
    println!("\n=== Scraping Results ===");
    println!("Pages scraped: {}", pages.len());
    println!("Time taken: {:?}", duration);
    
    let stats = scraper.get_stats().await;
    println!("Stats: {:?}", stats);
    
    // Task pipeline example
    println!("\n=== Task Pipeline ===");
    let pipeline = TaskPipeline::new(10, |task: String| async move {
        println!("Processing task: {}", task);
        sleep(Duration::from_millis(50)).await;
        println!("Completed task: {}", task);
    });
    
    for i in 0..5 {
        pipeline.submit(format!("Task {}", i)).await.unwrap();
    }
    
    // Rate limiter example
    println!("\n=== Rate Limiter ===");
    let rate_limiter = RateLimiter::new(3, Duration::from_secs(1));
    
    for i in 0..10 {
        rate_limiter.acquire().await;
        println!("Request {} at {:?}", i, Instant::now());
    }
    
    sleep(Duration::from_secs(1)).await;
}
```

## Stream Processing

```rust
// src/main.rs
use futures::stream::{self, Stream, StreamExt};
use tokio::time::{interval, Duration};
use std::pin::Pin;

// Create a stream that yields values
fn number_stream() -> impl Stream<Item = i32> {
    stream::iter(1..=10)
}

// Create an infinite stream with intervals
fn interval_stream() -> impl Stream<Item = i32> {
    let mut counter = 0;
    stream::unfold(counter, |state| async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        let current = state;
        Some((current, state + 1))
    })
}

// Custom stream implementation
struct CounterStream {
    count: i32,
    max: i32,
}

impl CounterStream {
    fn new(max: i32) -> Self {
        CounterStream { count: 0, max }
    }
}

impl Stream for CounterStream {
    type Item = i32;
    
    fn poll_next(
        mut self: Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        if self.count < self.max {
            let current = self.count;
            self.count += 1;
            std::task::Poll::Ready(Some(current))
        } else {
            std::task::Poll::Ready(None)
        }
    }
}

async fn process_stream<S>(mut stream: S) 
where
    S: Stream<Item = i32> + Unpin,
{
    while let Some(value) = stream.next().await {
        println!("Stream value: {}", value);
    }
}

#[tokio::main]
async fn main() {
    // Basic stream operations
    println!("=== Number Stream ===");
    let mut stream = number_stream();
    while let Some(n) = stream.next().await {
        println!("Number: {}", n);
    }
    
    // Stream combinators
    println!("\n=== Stream Combinators ===");
    let processed = number_stream()
        .map(|n| n * 2)
        .filter(|n| n % 3 == 0)
        .take(3);
    
    let results: Vec<i32> = processed.collect().await;
    println!("Processed results: {:?}", results);
    
    // Merging streams
    println!("\n=== Merged Streams ===");
    let stream1 = stream::iter(vec![1, 3, 5]);
    let stream2 = stream::iter(vec![2, 4, 6]);
    
    let merged = stream::select(stream1, stream2);
    let merged_results: Vec<i32> = merged.collect().await;
    println!("Merged: {:?}", merged_results);
    
    // Buffered processing
    println!("\n=== Buffered Stream ===");
    let futures = (1..=10).map(|i| async move {
        tokio::time::sleep(Duration::from_millis(100 - i * 10)).await;
        i
    });
    
    let buffered = stream::iter(futures).buffered(3);
    let buffered_results: Vec<i32> = buffered.collect().await;
    println!("Buffered results: {:?}", buffered_results);
    
    // Custom stream
    println!("\n=== Custom Stream ===");
    let counter = CounterStream::new(5);
    process_stream(counter).await;
}
```

## Error Handling in Async

```rust
// src/main.rs
use std::error::Error;
use std::fmt;
use tokio::time::{sleep, Duration};

#[derive(Debug)]
enum ServiceError {
    Timeout,
    Connection(String),
    Parse(String),
}

impl fmt::Display for ServiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ServiceError::Timeout => write!(f, "Operation timed out"),
            ServiceError::Connection(msg) => write!(f, "Connection error: {}", msg),
            ServiceError::Parse(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl Error for ServiceError {}

async fn flaky_service(id: u32) -> Result<String, ServiceError> {
    if id % 3 == 0 {
        Err(ServiceError::Connection("Failed to connect".to_string()))
    } else if id % 5 == 0 {
        Err(ServiceError::Parse("Invalid response".to_string()))
    } else {
        sleep(Duration::from_millis(50)).await;
        Ok(format!("Success: {}", id))
    }
}

async fn retry_with_backoff<F, Fut, T, E>(
    mut f: F,
    max_retries: u32,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
    E: fmt::Display,
{
    let mut retries = 0;
    let mut delay = Duration::from_millis(100);
    
    loop {
        match f().await {
            Ok(result) => return Ok(result),
            Err(e) if retries < max_retries => {
                eprintln!("Attempt {} failed: {}. Retrying in {:?}", retries + 1, e, delay);
                sleep(delay).await;
                delay *= 2; // Exponential backoff
                retries += 1;
            }
            Err(e) => return Err(e),
        }
    }
}

async fn process_batch(ids: Vec<u32>) -> Vec<Result<String, ServiceError>> {
    let futures = ids.into_iter().map(|id| async move {
        retry_with_backoff(|| flaky_service(id), 3).await
    });
    
    futures::future::join_all(futures).await
}

#[tokio::main]
async fn main() {
    // Single operation with error handling
    match flaky_service(1).await {
        Ok(result) => println!("Result: {}", result),
        Err(e) => eprintln!("Error: {}", e),
    }
    
    // Batch processing with mixed results
    let ids: Vec<u32> = (1..=20).collect();
    let results = process_batch(ids).await;
    
    let (successes, failures): (Vec<_>, Vec<_>) = results
        .into_iter()
        .partition(Result::is_ok);
    
    println!("\nSuccesses: {}", successes.len());
    for result in successes {
        println!("  {}", result.unwrap());
    }
    
    println!("\nFailures: {}", failures.len());
    for result in failures {
        println!("  {}", result.unwrap_err());
    }
    
    // Using ? operator in async context
    async fn complex_operation() -> Result<String, Box<dyn Error>> {
        let result1 = flaky_service(1).await?;
        let result2 = flaky_service(2).await?;
        Ok(format!("{} + {}", result1, result2))
    }
    
    match complex_operation().await {
        Ok(result) => println!("\nComplex operation: {}", result),
        Err(e) => eprintln!("Complex operation failed: {}", e),
    }
}
```

## Exercises

1. **Async HTTP Server**: Build a simple HTTP server using Tokio that handles multiple concurrent requests and implements rate limiting.

2. **Parallel Downloads**: Create a program that downloads multiple files concurrently with progress reporting and retry logic.

3. **Chat Server**: Implement a WebSocket chat server where clients can join rooms and broadcast messages to all room members.

4. **Stream Processing**: Build a log processing pipeline that reads log files, parses entries, filters by criteria, and aggregates statistics using async streams.

5. **Database Connection Pool**: Create an async connection pool that manages database connections with configurable min/max connections and idle timeouts.

## Key Takeaways

- `async`/`await` enables concurrent programming without threads
- Futures represent values that will be available in the future
- `tokio::spawn` creates concurrent tasks
- `select!` races multiple futures
- Async channels enable communication between tasks
- Streams provide async iteration
- Async mutex/rwlock prevent data races in async code
- Error handling uses the same `Result` type with `?` operator
- Retry logic and timeouts are essential for robust async code
- Always prefer async-aware synchronization primitives

## Next Steps

In Tutorial 11, we'll explore **Smart Pointers & Interior Mutability**, learning about Box, Rc, Arc, RefCell, and advanced memory management patterns in Rust.