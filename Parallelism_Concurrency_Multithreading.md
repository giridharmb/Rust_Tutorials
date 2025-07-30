# Rust Parallelism & Concurrency: Complete Guide

A comprehensive guide to all forms of parallelism, concurrency, multi-threading, and multi-processing in Rust.

## Table of Contents

- [Overview](#overview)
- [Comparison Table](#comparison-table)
- [Threading Approaches](#threading-approaches)
  - [1. std::thread - Native OS Threads](#1-stdthread---native-os-threads)
  - [2. Rayon - Data Parallelism](#2-rayon---data-parallelism)
  - [3. Crossbeam - Scoped Threads](#3-crossbeam---scoped-threads)
- [Async Approaches](#async-approaches)
  - [4. Tokio - Async Runtime](#4-tokio---async-runtime)
  - [5. async-std - Alternative Async Runtime](#5-async-std---alternative-async-runtime)
  - [6. Futures - Low-level Async](#6-futures---low-level-async)
- [Channel-based Concurrency](#channel-based-concurrency)
  - [7. std::sync::mpsc - Multi-producer Single-consumer](#7-stdsyncmpsc---multi-producer-single-consumer)
  - [8. Crossbeam Channels - Multi-producer Multi-consumer](#8-crossbeam-channels---multi-producer-multi-consumer)
  - [9. Tokio Channels - Async Channels](#9-tokio-channels---async-channels)
- [Shared State Concurrency](#shared-state-concurrency)
  - [10. Arc + Mutex - Shared Ownership](#10-arc--mutex---shared-ownership)
  - [11. RwLock - Read-Write Locks](#11-rwlock---read-write-locks)
  - [12. Atomic Types - Lock-free Programming](#12-atomic-types---lock-free-programming)
- [Actor Model](#actor-model)
  - [13. Actix - Actor Framework](#13-actix---actor-framework)
- [Process-based Parallelism](#process-based-parallelism)
  - [14. std::process - Multi-processing](#14-stdprocess---multi-processing)
- [GPU Computing](#gpu-computing)
  - [15. wgpu - GPU Parallelism](#15-wgpu---gpu-parallelism)
- [Hybrid Approaches](#hybrid-approaches)
- [Best Practices](#best-practices)

## Overview

Rust provides multiple paradigms for concurrent and parallel programming, each with different trade-offs in terms of performance, complexity, and use cases.

## Comparison Table

| Approach | Type | Best For | Pros | Cons | Overhead |
|----------|------|----------|------|------|----------|
| **std::thread** | OS Threads | CPU-bound tasks, true parallelism | • True parallelism<br>• Simple API<br>• No runtime needed | • High memory overhead<br>• Expensive context switching<br>• Limited by OS thread count | High |
| **Rayon** | Work-stealing threads | Data parallelism, batch processing | • Automatic work distribution<br>• Great for iterators<br>• Excellent performance | • Not for I/O tasks<br>• All-or-nothing parallelism<br>• Learning curve | Medium |
| **Crossbeam** | Scoped threads | Short-lived parallel tasks | • Borrow checker friendly<br>• No Arc needed<br>• Efficient channels | • Manual thread management<br>• Not for async I/O | Medium |
| **Tokio** | Async tasks | I/O-bound tasks, web servers | • Massive concurrency<br>• Low memory per task<br>• Rich ecosystem | • Complex error handling<br>• Async infection<br>• Runtime overhead | Low |
| **async-std** | Async tasks | I/O-bound tasks, simpler API | • Familiar std-like API<br>• Good for beginners<br>• Lightweight | • Smaller ecosystem<br>• Less features than Tokio | Low |
| **Futures** | Async primitives | Custom executors, libraries | • Maximum control<br>• No runtime lock-in<br>• Composable | • Low-level<br>• Requires executor<br>• Complex | Very Low |
| **std::sync::mpsc** | Channel messaging | Producer-consumer patterns | • Built into std<br>• Simple API<br>• Type safe | • Single consumer only<br>• Can't be cloned<br>• Blocking | Low |
| **Crossbeam channels** | Advanced channels | Complex messaging patterns | • Multi-consumer<br>• Select operation<br>• Better performance | • External dependency<br>• More complex API | Low |
| **Arc + Mutex** | Shared memory | Shared state between threads | • Familiar pattern<br>• Fine-grained control<br>• Works everywhere | • Deadlock potential<br>• Contention issues<br>• Error prone | Medium |
| **RwLock** | Read-heavy workloads | Many readers, few writers | • Concurrent reads<br>• Better than Mutex for reads<br>• Fair scheduling | • Writer starvation<br>• More overhead than Mutex<br>• Complexity | Medium |
| **Atomics** | Lock-free | High-performance counters | • No locks<br>• Fastest option<br>• Wait-free | • Very complex<br>• Limited use cases<br>• Architecture specific | Very Low |
| **Actix** | Actor model | Complex state machines | • Message passing<br>• Fault tolerance<br>• Supervision trees | • Learning curve<br>• Overhead<br>• Ecosystem lock-in | Medium |
| **std::process** | Multi-processing | Process isolation, fault tolerance | • True isolation<br>• Crash resilience<br>• Security boundaries | • High overhead<br>• Complex IPC<br>• Serialization costs | Very High |
| **wgpu** | GPU compute | Massive data parallelism | • Thousands of threads<br>• SIMD operations<br>• Graphics + compute | • GPU required<br>• Complex programming<br>• Limited algorithms | N/A |

## Threading Approaches

### 1. std::thread - Native OS Threads

Basic thread spawning for CPU-bound parallel tasks.

```rust
use std::thread;
use std::sync::{Arc, Mutex};
use std::time::Duration;

// Basic thread spawning
fn basic_threads() {
    let mut handles = vec![];
    
    for i in 0..4 {
        let handle = thread::spawn(move || {
            println!("Thread {} starting", i);
            thread::sleep(Duration::from_millis(100));
            i * i
        });
        handles.push(handle);
    }
    
    for handle in handles {
        let result = handle.join().unwrap();
        println!("Thread result: {}", result);
    }
}

// Shared state with Arc + Mutex
fn shared_state_threads() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];
    
    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            for _ in 0..1000 {
                let mut num = counter.lock().unwrap();
                *num += 1;
            }
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    println!("Final count: {}", *counter.lock().unwrap());
}

// Thread pool pattern
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
            println!("Worker {} executing job", id);
            job();
        });
        
        Worker {
            id,
            thread: Some(thread),
        }
    }
}
```

### 2. Rayon - Data Parallelism

Work-stealing thread pool for parallel iterators and divide-and-conquer algorithms.

```rust
use rayon::prelude::*;

// Parallel iterator processing
fn parallel_map() {
    let numbers: Vec<i32> = (0..1_000_000).collect();
    
    let squares: Vec<i32> = numbers
        .par_iter()
        .map(|&x| x * x)
        .collect();
    
    println!("Computed {} squares", squares.len());
}

// Parallel sorting
fn parallel_sort() {
    let mut data = vec![5, 2, 8, 1, 9, 3, 7, 4, 6];
    data.par_sort();
    println!("Sorted: {:?}", data);
}

// Parallel reduce
fn parallel_sum() {
    let sum: i64 = (0..1_000_000)
        .into_par_iter()
        .map(|x| x as i64)
        .sum();
    
    println!("Sum: {}", sum);
}

// Custom thread pool configuration
use rayon::ThreadPoolBuilder;

fn configured_thread_pool() -> Result<(), rayon::ThreadPoolBuildError> {
    let pool = ThreadPoolBuilder::new()
        .num_threads(4)
        .thread_name(|index| format!("rayon-worker-{}", index))
        .build()?;
    
    pool.install(|| {
        let result: Vec<_> = (0..100)
            .into_par_iter()
            .map(|i| i * i)
            .collect();
        println!("Computed {} results", result.len());
    });
    
    Ok(())
}

// Parallel divide and conquer
fn quicksort<T: Ord + Send>(v: &mut [T]) {
    if v.len() <= 1 {
        return;
    }
    
    let mid = partition(v);
    let (left, right) = v.split_at_mut(mid);
    
    rayon::join(
        || quicksort(left),
        || quicksort(&mut right[1..]),
    );
}

fn partition<T: Ord>(v: &mut [T]) -> usize {
    let pivot = v.len() - 1;
    let mut i = 0;
    for j in 0..pivot {
        if v[j] <= v[pivot] {
            v.swap(i, j);
            i += 1;
        }
    }
    v.swap(i, pivot);
    i
}

// Parallel try_fold for error handling
fn parallel_validation(items: Vec<String>) -> Result<Vec<i32>, String> {
    items
        .par_iter()
        .map(|s| {
            s.parse::<i32>()
                .map_err(|_| format!("Failed to parse: {}", s))
        })
        .collect()
}
```

### 3. Crossbeam - Scoped Threads

Scoped threads that can borrow from the stack safely.

```rust
use crossbeam::thread;
use crossbeam::channel::{bounded, unbounded};

// Scoped threads - borrow without Arc
fn scoped_threads() {
    let data = vec![1, 2, 3, 4, 5];
    let mut results = vec![0; 5];
    
    thread::scope(|s| {
        for (i, n) in data.iter().enumerate() {
            let result_ref = &mut results[i];
            s.spawn(move |_| {
                *result_ref = n * n;
            });
        }
    }).unwrap();
    
    println!("Results: {:?}", results);
}

// Producer-consumer with crossbeam channels
fn producer_consumer() {
    let (tx, rx) = bounded(10);
    
    thread::scope(|s| {
        // Producer
        s.spawn(|_| {
            for i in 0..20 {
                tx.send(i).unwrap();
                thread::sleep(Duration::from_millis(50));
            }
        });
        
        // Multiple consumers
        for id in 0..3 {
            let rx = rx.clone();
            s.spawn(move |_| {
                while let Ok(item) = rx.recv() {
                    println!("Consumer {} got: {}", id, item);
                }
            });
        }
    }).unwrap();
}

// Select operation on multiple channels
use crossbeam::select;

fn channel_select() {
    let (tx1, rx1) = unbounded();
    let (tx2, rx2) = unbounded();
    
    thread::scope(|s| {
        s.spawn(|_| {
            for i in 0..10 {
                tx1.send(format!("First: {}", i)).unwrap();
                thread::sleep(Duration::from_millis(100));
            }
        });
        
        s.spawn(|_| {
            for i in 0..10 {
                tx2.send(format!("Second: {}", i)).unwrap();
                thread::sleep(Duration::from_millis(150));
            }
        });
        
        s.spawn(|_| {
            loop {
                select! {
                    recv(rx1) -> msg => {
                        if let Ok(msg) = msg {
                            println!("Received: {}", msg);
                        } else {
                            break;
                        }
                    }
                    recv(rx2) -> msg => {
                        if let Ok(msg) = msg {
                            println!("Received: {}", msg);
                        } else {
                            break;
                        }
                    }
                }
            }
        });
    }).unwrap();
}
```

## Async Approaches

### 4. Tokio - Async Runtime

Full-featured async runtime for I/O-bound concurrency.

```rust
use tokio::task;
use tokio::time::{sleep, Duration};
use tokio::sync::{mpsc, Mutex, RwLock, Semaphore};
use std::sync::Arc;

// Basic async tasks
#[tokio::main]
async fn basic_async() {
    let task1 = task::spawn(async {
        sleep(Duration::from_millis(100)).await;
        "Task 1 complete"
    });
    
    let task2 = task::spawn(async {
        sleep(Duration::from_millis(50)).await;
        "Task 2 complete"
    });
    
    let (result1, result2) = tokio::join!(task1, task2);
    println!("{}, {}", result1.unwrap(), result2.unwrap());
}

// Concurrent HTTP requests
use reqwest;

async fn concurrent_requests() -> Result<(), Box<dyn std::error::Error>> {
    let urls = vec![
        "https://api.github.com/users/rust-lang",
        "https://api.github.com/users/tokio-rs",
        "https://api.github.com/users/hyperium",
    ];
    
    let mut tasks = vec![];
    
    for url in urls {
        tasks.push(task::spawn(async move {
            let response = reqwest::get(url).await?;
            let text = response.text().await?;
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(text.len())
        }));
    }
    
    for task in tasks {
        match task.await? {
            Ok(len) => println!("Response length: {}", len),
            Err(e) => eprintln!("Error: {}", e),
        }
    }
    
    Ok(())
}

// Async mutex for shared state
async fn async_shared_state() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];
    
    for i in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = task::spawn(async move {
            for _ in 0..100 {
                let mut lock = counter.lock().await;
                *lock += 1;
            }
            println!("Task {} done", i);
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.await.unwrap();
    }
    
    println!("Final count: {}", *counter.lock().await);
}

// Semaphore for rate limiting
async fn rate_limited_tasks() {
    let semaphore = Arc::new(Semaphore::new(3)); // Max 3 concurrent
    let mut handles = vec![];
    
    for i in 0..10 {
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let handle = task::spawn(async move {
            println!("Task {} starting", i);
            sleep(Duration::from_secs(1)).await;
            println!("Task {} done", i);
            drop(permit);
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.await.unwrap();
    }
}

// Async channels
async fn async_channels() {
    let (tx, mut rx) = mpsc::channel(32);
    
    // Producer
    let producer = task::spawn(async move {
        for i in 0..10 {
            tx.send(i).await.unwrap();
            sleep(Duration::from_millis(100)).await;
        }
    });
    
    // Consumer
    let consumer = task::spawn(async move {
        while let Some(value) = rx.recv().await {
            println!("Received: {}", value);
        }
    });
    
    tokio::join!(producer, consumer);
}

// CPU-bound tasks in async context
async fn cpu_bound_async() {
    let result = task::spawn_blocking(|| {
        // CPU-intensive work
        let mut sum = 0u64;
        for i in 0..1_000_000 {
            sum += i;
        }
        sum
    }).await.unwrap();
    
    println!("Sum: {}", result);
}
```

### 5. async-std - Alternative Async Runtime

Standard library-like async runtime.

```rust
use async_std::task;
use async_std::sync::{Arc, Mutex};
use async_std::channel;
use async_std::prelude::*;

// Basic async-std usage
#[async_std::main]
async fn async_std_example() -> Result<(), Box<dyn std::error::Error>> {
    let task1 = task::spawn(async {
        task::sleep(std::time::Duration::from_millis(100)).await;
        42
    });
    
    let task2 = task::spawn(async {
        task::sleep(std::time::Duration::from_millis(50)).await;
        84
    });
    
    let sum = task1.await + task2.await;
    println!("Sum: {}", sum);
    
    Ok(())
}

// Parallel stream processing
async fn stream_processing() {
    use async_std::stream;
    
    let sum = stream::from_iter(0..1000)
        .map(|x| x * 2)
        .filter(|x| x % 3 == 0)
        .fold(0, |acc, x| acc + x)
        .await;
    
    println!("Sum of multiples of 3: {}", sum);
}

// Concurrent file I/O
use async_std::fs;
use async_std::io;

async fn concurrent_file_ops() -> io::Result<()> {
    let tasks: Vec<_> = (0..5)
        .map(|i| {
            task::spawn(async move {
                let filename = format!("file_{}.txt", i);
                fs::write(&filename, format!("Content {}", i)).await?;
                let content = fs::read_to_string(&filename).await?;
                fs::remove_file(&filename).await?;
                Ok::<_, io::Error>(content)
            })
        })
        .collect();
    
    for task in tasks {
        match task.await {
            Ok(content) => println!("Processed: {}", content),
            Err(e) => eprintln!("Error: {}", e),
        }
    }
    
    Ok(())
}
```

### 6. Futures - Low-level Async

Building blocks for async programming without runtime.

```rust
use futures::future::{self, Future, FutureExt};
use futures::stream::{self, StreamExt};
use futures::executor;

// Custom future implementation
use std::pin::Pin;
use std::task::{Context, Poll};

struct TimerFuture {
    when: std::time::Instant,
}

impl Future for TimerFuture {
    type Output = ();
    
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if std::time::Instant::now() >= self.when {
            Poll::Ready(())
        } else {
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

// Combinators
fn future_combinators() {
    executor::block_on(async {
        // Join multiple futures
        let (a, b, c) = future::join3(
            async { 1 },
            async { 2 },
            async { 3 },
        ).await;
        println!("Results: {}, {}, {}", a, b, c);
        
        // Select first completed
        let either = future::select(
            future::ready(1),
            future::pending::<i32>(),
        ).await;
        
        match either {
            future::Either::Left((val, _)) => println!("Left: {}", val),
            future::Either::Right((val, _)) => println!("Right: {}", val),
        }
    });
}

// Stream processing
fn stream_example() {
    executor::block_on(async {
        let mut stream = stream::iter(vec![1, 2, 3, 4, 5])
            .map(|x| x * 2)
            .filter(|x| future::ready(x % 3 != 0));
        
        while let Some(value) = stream.next().await {
            println!("Value: {}", value);
        }
    });
}

// Custom executor
use futures::task::{ArcWake, waker_ref};
use std::sync::Arc;
use std::collections::VecDeque;

struct SimpleExecutor {
    ready_queue: VecDeque<Arc<Task>>,
}

struct Task {
    future: Mutex<Option<Pin<Box<dyn Future<Output = ()> + Send>>>>,
    executor: Arc<Mutex<SimpleExecutor>>,
}

impl ArcWake for Task {
    fn wake_by_ref(arc_self: &Arc<Self>) {
        let cloned = arc_self.clone();
        arc_self.executor.lock().unwrap().ready_queue.push_back(cloned);
    }
}
```

## Channel-based Concurrency

### 7. std::sync::mpsc - Multi-producer Single-consumer

Standard library channels for thread communication.

```rust
use std::sync::mpsc;
use std::thread;

// Basic channel usage
fn basic_channel() {
    let (tx, rx) = mpsc::channel();
    
    thread::spawn(move || {
        tx.send("Hello from thread").unwrap();
    });
    
    let msg = rx.recv().unwrap();
    println!("Received: {}", msg);
}

// Multiple producers
fn multiple_producers() {
    let (tx, rx) = mpsc::channel();
    let mut handles = vec![];
    
    for i in 0..5 {
        let tx_clone = tx.clone();
        let handle = thread::spawn(move || {
            tx_clone.send(format!("Message from thread {}", i)).unwrap();
        });
        handles.push(handle);
    }
    
    drop(tx); // Close original sender
    
    for received in rx {
        println!("Got: {}", received);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
}

// Synchronous channel with backpressure
fn sync_channel() {
    let (tx, rx) = mpsc::sync_channel(3); // Buffer size 3
    
    thread::spawn(move || {
        for i in 0..10 {
            println!("Sending {}", i);
            tx.send(i).unwrap();
        }
    });
    
    thread::sleep(Duration::from_secs(1));
    
    for received in rx {
        println!("Received: {}", received);
        thread::sleep(Duration::from_millis(200));
    }
}
```

### 8. Crossbeam Channels - Multi-producer Multi-consumer

More flexible channels with select support.

```rust
use crossbeam::channel::{unbounded, bounded, select, Receiver};

// Multi-consumer pattern
fn multi_consumer() {
    let (tx, rx) = unbounded();
    
    // Multiple consumers
    for i in 0..3 {
        let rx = rx.clone();
        thread::spawn(move || {
            while let Ok(msg) = rx.recv() {
                println!("Consumer {} got: {}", i, msg);
            }
        });
    }
    
    // Producer
    for i in 0..10 {
        tx.send(i).unwrap();
    }
}

// Complex select with timeout
fn select_with_timeout() {
    let (tx1, rx1) = bounded::<i32>(1);
    let (tx2, rx2) = bounded::<String>(1);
    
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(100));
        tx1.send(42).unwrap();
    });
    
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(200));
        tx2.send("Hello".to_string()).unwrap();
    });
    
    loop {
        select! {
            recv(rx1) -> msg => {
                if let Ok(num) = msg {
                    println!("Number: {}", num);
                }
            }
            recv(rx2) -> msg => {
                if let Ok(text) = msg {
                    println!("Text: {}", text);
                }
            }
            default(Duration::from_millis(50)) => {
                println!("Timeout!");
                break;
            }
        }
    }
}
```

### 9. Tokio Channels - Async Channels

Async-aware channels for tokio runtime.

```rust
use tokio::sync::{mpsc, broadcast, watch, oneshot};

// Bounded async channel
async fn tokio_mpsc() {
    let (tx, mut rx) = mpsc::channel(10);
    
    tokio::spawn(async move {
        for i in 0..20 {
            tx.send(i).await.unwrap();
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    });
    
    while let Some(value) = rx.recv().await {
        println!("Received: {}", value);
    }
}

// Broadcast channel - multiple consumers get all messages
async fn broadcast_channel() {
    let (tx, _) = broadcast::channel(16);
    let mut rx1 = tx.subscribe();
    let mut rx2 = tx.subscribe();
    
    tokio::spawn(async move {
        while let Ok(value) = rx1.recv().await {
            println!("Subscriber 1: {}", value);
        }
    });
    
    tokio::spawn(async move {
        while let Ok(value) = rx2.recv().await {
            println!("Subscriber 2: {}", value);
        }
    });
    
    for i in 0..5 {
        tx.send(i).unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

// Watch channel - single value, multiple readers
async fn watch_channel() {
    let (tx, rx) = watch::channel("initial");
    
    for i in 0..3 {
        let mut rx = rx.clone();
        tokio::spawn(async move {
            while rx.changed().await.is_ok() {
                println!("Watcher {} saw: {}", i, *rx.borrow());
            }
        });
    }
    
    for i in 0..5 {
        tx.send(format!("update {}", i)).unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

// Oneshot - single use channel
async fn oneshot_channel() {
    let (tx, rx) = oneshot::channel();
    
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        tx.send("Result").unwrap();
    });
    
    match rx.await {
        Ok(value) => println!("Got: {}", value),
        Err(_) => println!("Sender dropped"),
    }
}
```

## Shared State Concurrency

### 10. Arc + Mutex - Shared Ownership

Thread-safe reference counting with mutual exclusion.

```rust
use std::sync::{Arc, Mutex};

// Basic Arc + Mutex
fn arc_mutex_basic() {
    let data = Arc::new(Mutex::new(vec![1, 2, 3]));
    let mut handles = vec![];
    
    for i in 0..3 {
        let data = Arc::clone(&data);
        let handle = thread::spawn(move || {
            let mut vec = data.lock().unwrap();
            vec.push(i);
            println!("Thread {} pushed {}", i, i);
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    println!("Final data: {:?}", *data.lock().unwrap());
}

// Avoiding deadlocks with lock ordering
struct BankAccount {
    id: u32,
    balance: Mutex<f64>,
}

fn transfer(from: &BankAccount, to: &BankAccount, amount: f64) {
    // Always lock in consistent order to avoid deadlock
    let (first, second) = if from.id < to.id {
        (from, to)
    } else {
        (to, from)
    };
    
    let mut balance1 = first.balance.lock().unwrap();
    let mut balance2 = second.balance.lock().unwrap();
    
    if from.id == first.id {
        *balance1 -= amount;
        *balance2 += amount;
    } else {
        *balance2 -= amount;
        *balance1 += amount;
    }
}
```

### 11. RwLock - Read-Write Locks

Multiple readers or single writer pattern.

```rust
use std::sync::{Arc, RwLock};

// Read-heavy workload
fn rwlock_example() {
    let data = Arc::new(RwLock::new(HashMap::new()));
    
    // Initialize
    {
        let mut write = data.write().unwrap();
        write.insert("key1", "value1");
        write.insert("key2", "value2");
    }
    
    let mut handles = vec![];
    
    // Many readers
    for i in 0..5 {
        let data = Arc::clone(&data);
        let handle = thread::spawn(move || {
            let read = data.read().unwrap();
            println!("Reader {} sees {} entries", i, read.len());
            thread::sleep(Duration::from_millis(100));
        });
        handles.push(handle);
    }
    
    // Occasional writer
    let data_write = Arc::clone(&data);
    let writer = thread::spawn(move || {
        thread::sleep(Duration::from_millis(50));
        let mut write = data_write.write().unwrap();
        write.insert("key3", "value3");
        println!("Writer added key3");
    });
    
    for handle in handles {
        handle.join().unwrap();
    }
    writer.join().unwrap();
}
```

### 12. Atomic Types - Lock-free Programming

Lock-free concurrent programming with atomic operations.

```rust
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::sync::Arc;

// Atomic counter
fn atomic_counter() {
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
    
    println!("Final count: {}", counter.load(Ordering::SeqCst));
}

// Spinlock implementation
struct SpinLock {
    locked: AtomicBool,
}

impl SpinLock {
    fn new() -> Self {
        SpinLock {
            locked: AtomicBool::new(false),
        }
    }
    
    fn lock(&self) {
        while self.locked.compare_exchange_weak(
            false,
            true,
            Ordering::Acquire,
            Ordering::Relaxed,
        ).is_err() {
            std::hint::spin_loop();
        }
    }
    
    fn unlock(&self) {
        self.locked.store(false, Ordering::Release);
    }
}

// Lock-free stack
use std::ptr;

struct Node<T> {
    data: T,
    next: *mut Node<T>,
}

struct LockFreeStack<T> {
    head: AtomicPtr<Node<T>>,
}

impl<T> LockFreeStack<T> {
    fn new() -> Self {
        LockFreeStack {
            head: AtomicPtr::new(ptr::null_mut()),
        }
    }
    
    fn push(&self, data: T) {
        let new_node = Box::into_raw(Box::new(Node {
            data,
            next: ptr::null_mut(),
        }));
        
        loop {
            let head = self.head.load(Ordering::Acquire);
            unsafe { (*new_node).next = head; }
            
            match self.head.compare_exchange_weak(
                head,
                new_node,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(_) => continue,
            }
        }
    }
}
```

## Actor Model

### 13. Actix - Actor Framework

Message-passing concurrency with actors.

```rust
use actix::prelude::*;

// Define messages
#[derive(Message)]
#[rtype(result = "usize")]
struct GetCount;

#[derive(Message)]
#[rtype(result = "()")]
struct Increment;

// Define actor
struct Counter {
    count: usize,
}

impl Actor for Counter {
    type Context = Context<Self>;
}

// Handle messages
impl Handler<GetCount> for Counter {
    type Result = usize;
    
    fn handle(&mut self, _msg: GetCount, _ctx: &mut Context<Self>) -> Self::Result {
        self.count
    }
}

impl Handler<Increment> for Counter {
    type Result = ();
    
    fn handle(&mut self, _msg: Increment, _ctx: &mut Context<Self>) {
        self.count += 1;
    }
}

// Using actors
#[actix::main]
async fn actor_example() {
    let addr = Counter { count: 0 }.start();
    
    // Send messages
    for _ in 0..10 {
        addr.send(Increment).await.unwrap();
    }
    
    let count = addr.send(GetCount).await.unwrap();
    println!("Final count: {}", count);
}

// Supervisor pattern
struct Supervisor;

impl Actor for Supervisor {
    type Context = Context<Self>;
    
    fn started(&mut self, ctx: &mut Context<Self>) {
        // Start child actors
        let child = Counter { count: 0 }.start();
        ctx.run_later(Duration::from_secs(1), move |_, _| {
            child.do_send(Increment);
        });
    }
}

impl Supervised for Counter {}

impl Supervisor<Counter> for Supervisor {
    fn restarting(&mut self, _addr: &mut <Counter as Actor>::Context) {
        println!("Restarting counter actor");
    }
}
```

## Process-based Parallelism

### 14. std::process - Multi-processing

Spawn separate OS processes for true isolation.

```rust
use std::process::{Command, Stdio};
use std::io::{Write, BufReader, BufRead};

// Basic process spawning
fn spawn_process() -> std::io::Result<()> {
    let output = Command::new("echo")
        .arg("Hello from subprocess")
        .output()?;
    
    println!("Output: {}", String::from_utf8_lossy(&output.stdout));
    Ok(())
}

// Piped communication
fn process_pipeline() -> std::io::Result<()> {
    let mut child = Command::new("python3")
        .arg("-c")
        .arg("
import sys
for line in sys.stdin:
    print(f'Processed: {line.strip()}')
")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    
    let mut stdin = child.stdin.take().unwrap();
    
    // Send data to child process
    thread::spawn(move || {
        for i in 0..5 {
            writeln!(stdin, "Message {}", i).unwrap();
        }
    });
    
    // Read output
    let stdout = child.stdout.take().unwrap();
    let reader = BufReader::new(stdout);
    
    for line in reader.lines() {
        println!("Got: {}", line?);
    }
    
    child.wait()?;
    Ok(())
}

// Parallel processing with multiple processes
fn multiprocess_parallel() -> Result<(), Box<dyn std::error::Error>> {
    let data_chunks = vec![
        vec![1, 2, 3, 4, 5],
        vec![6, 7, 8, 9, 10],
        vec![11, 12, 13, 14, 15],
    ];
    
    let mut children = vec![];
    
    for chunk in data_chunks {
        let child = Command::new("python3")
            .arg("-c")
            .arg(format!(
                "print(sum({:?}))",
                chunk
            ))
            .output()?;
        children.push(child);
    }
    
    for output in children {
        let result = String::from_utf8_lossy(&output.stdout);
        println!("Chunk sum: {}", result.trim());
    }
    
    Ok(())
}

// IPC with shared memory (using memmap2)
use memmap2::MmapMut;
use std::fs::OpenOptions;

fn shared_memory_ipc() -> Result<(), Box<dyn std::error::Error>> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open("shared.dat")?;
    
    file.set_len(1024)?;
    
    let mut mmap = unsafe { MmapMut::map_mut(&file)? };
    
    // Parent writes
    mmap[..5].copy_from_slice(b"Hello");
    
    let child = Command::new("cat")
        .arg("shared.dat")
        .output()?;
    
    println!("Child read: {:?}", String::from_utf8_lossy(&child.stdout));
    
    Ok(())
}
```

## GPU Computing

### 15. wgpu - GPU Parallelism

Massive parallelism on GPU for compute workloads.

```rust
use wgpu;

// GPU compute shader for parallel addition
const COMPUTE_SHADER: &str = r#"
@group(0) @binding(0)
var<storage, read> input_a: array<f32>;

@group(0) @binding(1)
var<storage, read> input_b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index < arrayLength(&input_a)) {
        output[index] = input_a[index] + input_b[index];
    }
}
"#;

async fn gpu_compute() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GPU
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await?;
    
    // Create compute pipeline
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Compute Shader"),
        source: wgpu::ShaderSource::Wgsl(COMPUTE_SHADER.into()),
    });
    
    // Create buffers and run computation
    let size = 1000;
    let input_a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let input_b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
    
    // ... (buffer creation and execution code)
    
    Ok(())
}
```

## Hybrid Approaches

### Combining Different Paradigms

```rust
// Async + Threading hybrid
async fn hybrid_compute() {
    let (tx, mut rx) = tokio::sync::mpsc::channel(10);
    
    // CPU-bound work in thread pool
    for i in 0..4 {
        let tx = tx.clone();
        std::thread::spawn(move || {
            let result = expensive_computation(i);
            tokio::runtime::Handle::current().block_on(async {
                tx.send(result).await.unwrap();
            });
        });
    }
    
    drop(tx);
    
    // Async I/O processing of results
    while let Some(result) = rx.recv().await {
        save_to_database(result).await;
    }
}

// Rayon + Tokio combination
async fn parallel_async_processing(urls: Vec<String>) -> Vec<String> {
    let client = reqwest::Client::new();
    let runtime = tokio::runtime::Handle::current();
    
    urls.into_par_iter()
        .map(|url| {
            runtime.block_on(async {
                client.get(&url)
                    .send()
                    .await
                    .ok()?
                    .text()
                    .await
                    .ok()
            })
        })
        .flatten()
        .collect()
}
```

## Best Practices

### 1. **Choosing the Right Approach**

```rust
// I/O-bound → Async
async fn handle_web_request() { /* ... */ }

// CPU-bound → Threads/Rayon
fn compute_mandelbrot() { /* ... */ }

// Mixed → Hybrid
async fn process_uploaded_image() {
    let image = download_image().await;
    let processed = tokio::task::spawn_blocking(|| {
        apply_filters(image)
    }).await;
    upload_result(processed).await;
}
```

### 2. **Error Handling**

```rust
// Always handle panics in threads
let handle = thread::spawn(|| {
    panic::catch_unwind(|| {
        // Potentially panicking code
    })
});

// Use proper error types
type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
```

### 3. **Resource Management**

```rust
// Limit concurrency
let semaphore = Arc::new(Semaphore::new(100));

// Use connection pools
let pool = sqlx::PgPool::builder()
    .max_connections(5)
    .build("postgresql://...").await?;
```

### 4. **Testing Concurrent Code**

```rust
#[tokio::test]
async fn test_concurrent_operation() {
    let result = timeout(Duration::from_secs(5), async {
        // Test code
    }).await;
    
    assert!(result.is_ok());
}

// Use loom for concurrency testing
#[cfg(loom)]
#[test]
fn test_concurrent_data_structure() {
    loom::model(|| {
        // Test implementation
    });
}
```

## Performance Considerations

1. **Context Switching**: OS threads have high overhead
2. **Memory Usage**: Each OS thread uses ~2MB stack
3. **Cache Locality**: Keep related data together
4. **False Sharing**: Pad data to cache line boundaries
5. **Lock Contention**: Prefer message passing over shared state

## Common Pitfalls

1. **Deadlocks**: Always acquire locks in consistent order
2. **Data Races**: Use proper synchronization primitives
3. **Async Infection**: Once async, always async
4. **Blocking in Async**: Never block the async runtime
5. **Unbounded Channels**: Always use bounded channels in production