# Tutorial 9: Concurrency & Threading

## Creating Threads

Rust's ownership system enables fearless concurrency by preventing data races at compile time.

```rust
// src/main.rs
use std::thread;
use std::time::Duration;

fn main() {
    // Spawning a simple thread
    let handle = thread::spawn(|| {
        for i in 1..10 {
            println!("hi number {} from the spawned thread!", i);
            thread::sleep(Duration::from_millis(1));
        }
    });
    
    // Main thread work
    for i in 1..5 {
        println!("hi number {} from the main thread!", i);
        thread::sleep(Duration::from_millis(1));
    }
    
    // Wait for spawned thread to finish
    handle.join().unwrap();
    
    // Using move to transfer ownership
    let v = vec![1, 2, 3];
    
    let handle = thread::spawn(move || {
        println!("Here's a vector: {:?}", v);
        // v is now owned by this thread
    });
    
    // println!("{:?}", v); // Error: v was moved
    
    handle.join().unwrap();
}
```

## Message Passing with Channels

Channels allow threads to communicate by sending messages.

```rust
// src/main.rs
use std::sync::mpsc; // Multiple Producer, Single Consumer
use std::thread;
use std::time::Duration;

fn main() {
    // Creating a channel
    let (tx, rx) = mpsc::channel();
    
    // Single producer
    thread::spawn(move || {
        let val = String::from("hi");
        tx.send(val).unwrap();
        // println!("val is {}", val); // Error: val was moved
    });
    
    let received = rx.recv().unwrap();
    println!("Got: {}", received);
    
    // Multiple values
    let (tx, rx) = mpsc::channel();
    
    thread::spawn(move || {
        let vals = vec![
            String::from("hi"),
            String::from("from"),
            String::from("the"),
            String::from("thread"),
        ];
        
        for val in vals {
            tx.send(val).unwrap();
            thread::sleep(Duration::from_millis(500));
        }
    });
    
    // Receiving as iterator
    for received in rx {
        println!("Got: {}", received);
    }
    
    // Multiple producers
    let (tx, rx) = mpsc::channel();
    let tx1 = tx.clone();
    
    thread::spawn(move || {
        let vals = vec!["hi", "from", "thread", "one"];
        for val in vals {
            tx1.send(format!("1: {}", val)).unwrap();
            thread::sleep(Duration::from_millis(100));
        }
    });
    
    thread::spawn(move || {
        let vals = vec!["more", "messages", "from", "two"];
        for val in vals {
            tx.send(format!("2: {}", val)).unwrap();
            thread::sleep(Duration::from_millis(100));
        }
    });
    
    for received in rx {
        println!("Got: {}", received);
    }
}
```

## Shared-State Concurrency

Using Mutex (mutual exclusion) for safe shared memory access.

```rust
// src/main.rs
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    // Mutex basics
    let m = Mutex::new(5);
    
    {
        let mut num = m.lock().unwrap();
        *num = 6;
    } // lock is released here
    
    println!("m = {:?}", m);
    
    // Sharing Mutex between threads with Arc
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];
    
    for i in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
            println!("Thread {} incremented counter", i);
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    println!("Final counter: {}", *counter.lock().unwrap());
}
```

## Thread Pools and Scoped Threads

```rust
// src/main.rs
use std::thread;
use std::sync::{Arc, Mutex};

// Simple thread pool implementation
struct ThreadPool {
    workers: Vec<Worker>,
    sender: std::sync::mpsc::Sender<Job>,
}

type Job = Box<dyn FnOnce() + Send + 'static>;

struct Worker {
    id: usize,
    thread: Option<thread::JoinHandle<()>>,
}

impl ThreadPool {
    fn new(size: usize) -> ThreadPool {
        assert!(size > 0);
        
        let (sender, receiver) = std::sync::mpsc::channel();
        let receiver = Arc::new(Mutex::new(receiver));
        
        let mut workers = Vec::with_capacity(size);
        
        for id in 0..size {
            workers.push(Worker::new(id, Arc::clone(&receiver)));
        }
        
        ThreadPool { workers, sender }
    }
    
    fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let job = Box::new(f);
        self.sender.send(job).unwrap();
    }
}

impl Worker {
    fn new(id: usize, receiver: Arc<Mutex<std::sync::mpsc::Receiver<Job>>>) -> Worker {
        let thread = thread::spawn(move || loop {
            let job = receiver.lock().unwrap().recv().unwrap();
            println!("Worker {} got a job; executing.", id);
            job();
        });
        
        Worker {
            id,
            thread: Some(thread),
        }
    }
}

fn main() {
    let pool = ThreadPool::new(4);
    
    for i in 0..8 {
        pool.execute(move || {
            println!("Task {} is running", i);
            thread::sleep(std::time::Duration::from_millis(1000));
            println!("Task {} is done", i);
        });
    }
    
    thread::sleep(std::time::Duration::from_secs(3));
    
    // Scoped threads (requires crossbeam crate in real usage)
    let mut vec = vec![1, 2, 3, 4];
    let mut x = 0;
    
    // This demonstrates the concept - in practice use crossbeam::scope
    {
        let handles: Vec<_> = vec.chunks_mut(2)
            .map(|chunk| {
                thread::spawn(move || {
                    chunk[0] *= 2;
                    if chunk.len() > 1 {
                        chunk[1] *= 2;
                    }
                })
            })
            .collect();
        
        for handle in handles {
            handle.join().unwrap();
        }
    }
    
    // Note: In real code, use crossbeam for scoped threads
    // crossbeam::scope(|s| {
    //     s.spawn(|_| {
    //         vec.push(5);
    //     });
    // }).unwrap();
}
```

## Sync Primitives

```rust
// src/main.rs
use std::sync::{Arc, Mutex, RwLock, Condvar, Barrier};
use std::thread;
use std::time::Duration;

fn main() {
    // RwLock - Multiple readers OR one writer
    let lock = Arc::new(RwLock::new(vec![1, 2, 3]));
    
    // Multiple readers
    let mut handles = vec![];
    
    for i in 0..3 {
        let lock = Arc::clone(&lock);
        let handle = thread::spawn(move || {
            let data = lock.read().unwrap();
            println!("Reader {} sees: {:?}", i, *data);
        });
        handles.push(handle);
    }
    
    // One writer
    let lock_clone = Arc::clone(&lock);
    let writer = thread::spawn(move || {
        let mut data = lock_clone.write().unwrap();
        data.push(4);
        println!("Writer added 4");
    });
    
    for handle in handles {
        handle.join().unwrap();
    }
    writer.join().unwrap();
    
    // Condition Variable
    let pair = Arc::new((Mutex::new(false), Condvar::new()));
    let pair2 = Arc::clone(&pair);
    
    thread::spawn(move || {
        let (lock, cvar) = &*pair2;
        let mut started = lock.lock().unwrap();
        *started = true;
        cvar.notify_one();
        println!("Notified!");
    });
    
    let (lock, cvar) = &*pair;
    let mut started = lock.lock().unwrap();
    while !*started {
        println!("Waiting...");
        started = cvar.wait(started).unwrap();
    }
    println!("Received notification!");
    
    // Barrier - Synchronization point
    let mut handles = vec![];
    let barrier = Arc::new(Barrier::new(3));
    
    for i in 0..3 {
        let barrier = Arc::clone(&barrier);
        let handle = thread::spawn(move || {
            println!("Thread {} before barrier", i);
            barrier.wait();
            println!("Thread {} after barrier", i);
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
}
```

## Atomic Operations

```rust
// src/main.rs
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

fn main() {
    // Atomic counter
    let counter = Arc::new(AtomicUsize::new(0));
    let mut handles = vec![];
    
    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            for _ in 0..1000 {
                counter.fetch_add(1, Ordering::SeqCst);
            }
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    println!("Counter: {}", counter.load(Ordering::SeqCst));
    
    // Atomic flag for coordination
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = Arc::clone(&running);
    
    let worker = thread::spawn(move || {
        let mut count = 0;
        while running_clone.load(Ordering::Relaxed) {
            count += 1;
            if count % 1_000_000 == 0 {
                println!("Still working... {}", count);
            }
        }
        println!("Worker stopped at count: {}", count);
    });
    
    thread::sleep(std::time::Duration::from_millis(100));
    running.store(false, Ordering::Relaxed);
    println!("Stopping worker...");
    
    worker.join().unwrap();
}
```

## Real-World Example: Parallel Data Processing

```rust
// src/main.rs
use std::sync::{Arc, Mutex};
use std::thread;
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct DataPoint {
    id: u64,
    value: f64,
    category: String,
}

struct ParallelProcessor {
    thread_count: usize,
}

impl ParallelProcessor {
    fn new(thread_count: usize) -> Self {
        ParallelProcessor { thread_count }
    }
    
    fn process_batch(&self, data: Vec<DataPoint>) -> ProcessingResult {
        let chunk_size = (data.len() + self.thread_count - 1) / self.thread_count;
        let chunks: Vec<Vec<DataPoint>> = data
            .chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();
        
        let results = Arc::new(Mutex::new(Vec::new()));
        let mut handles = vec![];
        
        for (i, chunk) in chunks.into_iter().enumerate() {
            let results = Arc::clone(&results);
            let handle = thread::spawn(move || {
                println!("Thread {} processing {} items", i, chunk.len());
                let local_result = process_chunk(chunk);
                results.lock().unwrap().push(local_result);
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Merge results
        let results = results.lock().unwrap();
        merge_results(results.clone())
    }
}

#[derive(Debug)]
struct ChunkResult {
    sum: f64,
    count: usize,
    categories: HashMap<String, usize>,
    max_value: f64,
    min_value: f64,
}

#[derive(Debug)]
struct ProcessingResult {
    total_sum: f64,
    total_count: usize,
    average: f64,
    categories: HashMap<String, usize>,
    max_value: f64,
    min_value: f64,
}

fn process_chunk(data: Vec<DataPoint>) -> ChunkResult {
    let mut sum = 0.0;
    let mut categories = HashMap::new();
    let mut max_value = f64::MIN;
    let mut min_value = f64::MAX;
    
    for point in &data {
        sum += point.value;
        *categories.entry(point.category.clone()).or_insert(0) += 1;
        max_value = max_value.max(point.value);
        min_value = min_value.min(point.value);
    }
    
    ChunkResult {
        sum,
        count: data.len(),
        categories,
        max_value,
        min_value,
    }
}

fn merge_results(results: Vec<ChunkResult>) -> ProcessingResult {
    let mut total_sum = 0.0;
    let mut total_count = 0;
    let mut categories = HashMap::new();
    let mut max_value = f64::MIN;
    let mut min_value = f64::MAX;
    
    for result in results {
        total_sum += result.sum;
        total_count += result.count;
        max_value = max_value.max(result.max_value);
        min_value = min_value.min(result.min_value);
        
        for (category, count) in result.categories {
            *categories.entry(category).or_insert(0) += count;
        }
    }
    
    ProcessingResult {
        total_sum,
        total_count,
        average: if total_count > 0 { total_sum / total_count as f64 } else { 0.0 },
        categories,
        max_value,
        min_value,
    }
}

// Producer-Consumer Pattern
struct Pipeline<T> {
    sender: std::sync::mpsc::Sender<Option<T>>,
    receiver: Arc<Mutex<std::sync::mpsc::Receiver<Option<T>>>>,
}

impl<T> Pipeline<T> {
    fn new() -> Self {
        let (sender, receiver) = std::sync::mpsc::channel();
        Pipeline {
            sender,
            receiver: Arc::new(Mutex::new(receiver)),
        }
    }
    
    fn sender(&self) -> std::sync::mpsc::Sender<Option<T>> {
        self.sender.clone()
    }
    
    fn spawn_consumer<F>(&self, id: usize, mut processor: F) -> thread::JoinHandle<()>
    where
        F: FnMut(T) + Send + 'static,
        T: Send + 'static,
    {
        let receiver = Arc::clone(&self.receiver);
        thread::spawn(move || {
            loop {
                let item = receiver.lock().unwrap().recv().unwrap();
                match item {
                    Some(data) => {
                        println!("Consumer {} processing", id);
                        processor(data);
                    }
                    None => {
                        println!("Consumer {} shutting down", id);
                        break;
                    }
                }
            }
        })
    }
}

fn main() {
    // Generate test data
    let mut data = Vec::new();
    for i in 0..10000 {
        data.push(DataPoint {
            id: i,
            value: (i as f64 * 0.1).sin() * 100.0,
            category: match i % 3 {
                0 => "A",
                1 => "B",
                _ => "C",
            }.to_string(),
        });
    }
    
    // Parallel processing
    let processor = ParallelProcessor::new(4);
    let start = std::time::Instant::now();
    let result = processor.process_batch(data.clone());
    let duration = start.elapsed();
    
    println!("Parallel processing took: {:?}", duration);
    println!("Results: {:?}", result);
    
    // Pipeline example
    let pipeline: Pipeline<Vec<DataPoint>> = Pipeline::new();
    let mut consumers = vec![];
    
    // Spawn consumers
    for i in 0..3 {
        let consumer = pipeline.spawn_consumer(i, move |batch| {
            let sum: f64 = batch.iter().map(|p| p.value).sum();
            println!("Consumer {} processed batch, sum: {}", i, sum);
            thread::sleep(std::time::Duration::from_millis(100));
        });
        consumers.push(consumer);
    }
    
    // Producer
    let sender = pipeline.sender();
    thread::spawn(move || {
        for chunk in data.chunks(1000) {
            sender.send(Some(chunk.to_vec())).unwrap();
        }
        // Send termination signal to all consumers
        for _ in 0..3 {
            sender.send(None).unwrap();
        }
    });
    
    // Wait for consumers
    for consumer in consumers {
        consumer.join().unwrap();
    }
}
```

## Thread Safety Patterns

```rust
// src/main.rs
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::collections::HashMap;
use std::cell::RefCell;

// Thread-safe cache implementation
struct ThreadSafeCache<K, V> {
    data: Arc<RwLock<HashMap<K, V>>>,
}

impl<K, V> ThreadSafeCache<K, V>
where
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
{
    fn new() -> Self {
        ThreadSafeCache {
            data: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    fn get(&self, key: &K) -> Option<V> {
        self.data.read().unwrap().get(key).cloned()
    }
    
    fn insert(&self, key: K, value: V) {
        self.data.write().unwrap().insert(key, value);
    }
    
    fn get_or_compute<F>(&self, key: K, compute: F) -> V
    where
        F: FnOnce() -> V,
    {
        // Try to get with read lock first
        {
            let cache = self.data.read().unwrap();
            if let Some(value) = cache.get(&key) {
                return value.clone();
            }
        }
        
        // Compute and insert with write lock
        let value = compute();
        self.data.write().unwrap().insert(key.clone(), value.clone());
        value
    }
}

// Message passing actor pattern
enum ActorMessage {
    Increment,
    Decrement,
    Get(std::sync::mpsc::Sender<i32>),
}

struct CounterActor {
    receiver: std::sync::mpsc::Receiver<ActorMessage>,
    count: i32,
}

impl CounterActor {
    fn new() -> (Self, std::sync::mpsc::Sender<ActorMessage>) {
        let (sender, receiver) = std::sync::mpsc::channel();
        let actor = CounterActor {
            receiver,
            count: 0,
        };
        (actor, sender)
    }
    
    fn run(mut self) {
        while let Ok(msg) = self.receiver.recv() {
            match msg {
                ActorMessage::Increment => self.count += 1,
                ActorMessage::Decrement => self.count -= 1,
                ActorMessage::Get(sender) => {
                    sender.send(self.count).unwrap();
                }
            }
        }
    }
}

// Thread-local storage example
thread_local! {
    static THREAD_ID: RefCell<Option<usize>> = RefCell::new(None);
}

fn set_thread_id(id: usize) {
    THREAD_ID.with(|f| {
        *f.borrow_mut() = Some(id);
    });
}

fn get_thread_id() -> Option<usize> {
    THREAD_ID.with(|f| *f.borrow())
}

fn main() {
    // Using thread-safe cache
    let cache = ThreadSafeCache::new();
    let cache_clone = cache.data.clone();
    
    let handles: Vec<_> = (0..5)
        .map(|i| {
            let cache = ThreadSafeCache { data: cache_clone.clone() };
            thread::spawn(move || {
                let value = cache.get_or_compute(i, || {
                    println!("Computing value for key {}", i);
                    thread::sleep(std::time::Duration::from_millis(100));
                    i * i
                });
                println!("Thread got value {} for key {}", value, i);
            })
        })
        .collect();
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Actor pattern
    let (actor, sender) = CounterActor::new();
    
    thread::spawn(move || {
        actor.run();
    });
    
    // Send messages to actor
    sender.send(ActorMessage::Increment).unwrap();
    sender.send(ActorMessage::Increment).unwrap();
    sender.send(ActorMessage::Decrement).unwrap();
    
    let (tx, rx) = std::sync::mpsc::channel();
    sender.send(ActorMessage::Get(tx)).unwrap();
    println!("Counter value: {}", rx.recv().unwrap());
    
    // Thread-local storage
    let handles: Vec<_> = (0..3)
        .map(|i| {
            thread::spawn(move || {
                set_thread_id(i);
                println!("Thread {} has ID: {:?}", i, get_thread_id());
            })
        })
        .collect();
    
    for handle in handles {
        handle.join().unwrap();
    }
}
```

## Exercises

1. **Parallel Computation**: Implement a parallel matrix multiplication algorithm using threads. Compare performance with single-threaded version.

2. **Thread Pool**: Build a more complete thread pool that handles panics gracefully and can be shut down cleanly.

3. **Producer-Consumer**: Create a multi-producer, multi-consumer queue with backpressure (blocking when queue is full).

4. **Concurrent Data Structure**: Implement a thread-safe LRU (Least Recently Used) cache with a maximum size.

5. **Deadlock Detection**: Create a scenario that would deadlock and then fix it using lock ordering or other techniques.

## Key Takeaways

- Rust prevents data races at compile time through ownership
- Use `thread::spawn` to create threads and `join()` to wait
- Channels enable safe message passing between threads
- `Arc<Mutex<T>>` allows shared mutable state
- `RwLock` enables multiple readers or one writer
- Atomic types provide lock-free operations
- Choose between shared state and message passing based on the problem
- Thread pools amortize thread creation costs
- Actor pattern provides isolated mutable state
- Always prefer immutability and message passing when possible

## Next Steps

In Tutorial 10, we'll explore **Async Programming**, learning about futures, async/await syntax, and building high-performance concurrent applications without the overhead of threads.