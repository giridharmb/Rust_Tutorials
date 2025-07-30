# Rust Async Worker Pool Tutorial

A comprehensive guide to implementing worker pools in Rust using async/await, demonstrating multiple approaches for concurrent task execution.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Project Setup](#project-setup)
- [Implementation Approaches](#implementation-approaches)
  - [Approach 1: Channel-based Worker Pool](#approach-1-channel-based-worker-pool)
  - [Approach 2: Semaphore-based Concurrency Control](#approach-2-semaphore-based-concurrency-control)
  - [Approach 3: Actor Model with tokio::task](#approach-3-actor-model-with-tokiotask)
  - [Approach 4: Stream-based Processing](#approach-4-stream-based-processing)
  - [Approach 5: Rayon with Async Bridge](#approach-5-rayon-with-async-bridge)
- [Performance Comparison](#performance-comparison)
- [Best Practices](#best-practices)
- [Running the Examples](#running-the-examples)

## Overview

This tutorial explores different patterns for implementing worker pools in Rust that can:
- Execute async functions concurrently
- Process various inputs in parallel
- Capture and aggregate results
- Handle errors gracefully
- Control concurrency levels

## Prerequisites

- Rust 1.75+ (for async traits stabilization)
- Basic understanding of Rust async/await
- Familiarity with tokio runtime

## Project Setup

Create a new Rust project:

```bash
cargo new rust-async-worker-pool
cd rust-async-worker-pool
```

Add dependencies to `Cargo.toml`:

```toml
[dependencies]
tokio = { version = "1.35", features = ["full"] }
async-trait = "0.1"
futures = "0.3"
crossbeam = "0.8"
rayon = "1.8"
anyhow = "1.0"
thiserror = "1.0"
tracing = "0.1"
tracing-subscriber = "0.3"

[dev-dependencies]
criterion = "0.5"
```

## Implementation Approaches

### Approach 1: Channel-based Worker Pool

This approach uses channels to distribute work among a fixed number of worker tasks.

```rust
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use anyhow::Result;

#[derive(Debug, Clone)]
pub struct WorkItem<T: Send + 'static> {
    pub id: usize,
    pub data: T,
}

#[derive(Debug)]
pub struct WorkResult<T, R> {
    pub id: usize,
    pub input: T,
    pub result: Result<R>,
}

pub struct ChannelWorkerPool<T, R> 
where
    T: Send + 'static,
    R: Send + 'static,
{
    tx: mpsc::Sender<WorkItem<T>>,
    results_rx: Arc<Mutex<mpsc::Receiver<WorkResult<T, R>>>>,
    workers: Vec<JoinHandle<()>>,
}

impl<T, R> ChannelWorkerPool<T, R>
where
    T: Send + Clone + 'static,
    R: Send + 'static,
{
    pub fn new<F, Fut>(worker_count: usize, process_fn: F) -> Self
    where
        F: Fn(T) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<R>> + Send + 'static,
    {
        let (tx, mut rx) = mpsc::channel::<WorkItem<T>>(100);
        let (results_tx, results_rx) = mpsc::channel::<WorkResult<T, R>>(100);
        
        let mut workers = Vec::with_capacity(worker_count);
        
        for _ in 0..worker_count {
            let mut worker_rx = rx.clone();
            let worker_results_tx = results_tx.clone();
            let worker_process_fn = process_fn.clone();
            
            let handle = tokio::spawn(async move {
                while let Some(work_item) = worker_rx.recv().await {
                    let result = worker_process_fn(work_item.data.clone()).await;
                    
                    let work_result = WorkResult {
                        id: work_item.id,
                        input: work_item.data,
                        result,
                    };
                    
                    if worker_results_tx.send(work_result).await.is_err() {
                        break;
                    }
                }
            });
            
            workers.push(handle);
        }
        
        // Close the original receiver
        drop(rx);
        drop(results_tx);
        
        Self {
            tx,
            results_rx: Arc::new(Mutex::new(results_rx)),
            workers,
        }
    }
    
    pub async fn submit(&self, id: usize, data: T) -> Result<()> {
        self.tx.send(WorkItem { id, data }).await
            .map_err(|_| anyhow::anyhow!("Failed to submit work"))
    }
    
    pub async fn get_result(&self) -> Option<WorkResult<T, R>> {
        let mut rx = self.results_rx.lock().await;
        rx.recv().await
    }
    
    pub async fn shutdown(self) -> Result<Vec<WorkResult<T, R>>> {
        drop(self.tx);
        
        // Wait for all workers to finish
        for worker in self.workers {
            worker.await?;
        }
        
        // Collect remaining results
        let mut results = Vec::new();
        let mut rx = self.results_rx.lock().await;
        while let Some(result) = rx.recv().await {
            results.push(result);
        }
        
        Ok(results)
    }
}

// Example usage
#[tokio::main]
async fn main() -> Result<()> {
    // Define a CPU-intensive async function
    async fn process_data(n: u64) -> Result<u64> {
        // Simulate async work
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        Ok(n * n)
    }
    
    let pool = ChannelWorkerPool::new(4, process_data);
    
    // Submit work
    for i in 0..10 {
        pool.submit(i, i as u64).await?;
    }
    
    // Collect results as they complete
    let mut results = Vec::new();
    for _ in 0..10 {
        if let Some(result) = pool.get_result().await {
            results.push(result);
        }
    }
    
    // Sort by ID to verify all work was completed
    results.sort_by_key(|r| r.id);
    
    for result in results {
        match result.result {
            Ok(value) => println!("Task {} completed: {} -> {}", result.id, result.input, value),
            Err(e) => println!("Task {} failed: {}", result.id, e),
        }
    }
    
    Ok(())
}
```

### Approach 2: Semaphore-based Concurrency Control

This approach uses a semaphore to limit concurrent executions without pre-spawning workers.

```rust
use tokio::sync::{Semaphore, Mutex};
use futures::future::join_all;
use std::sync::Arc;

pub struct SemaphoreWorkerPool<T, R> {
    semaphore: Arc<Semaphore>,
    results: Arc<Mutex<Vec<WorkResult<T, R>>>>,
}

impl<T, R> SemaphoreWorkerPool<T, R>
where
    T: Send + Clone + 'static,
    R: Send + 'static,
{
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            results: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    pub async fn process_batch<F, Fut>(&self, items: Vec<(usize, T)>, process_fn: F) -> Vec<WorkResult<T, R>>
    where
        F: Fn(T) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<R>> + Send + 'static,
    {
        let tasks: Vec<_> = items
            .into_iter()
            .map(|(id, data)| {
                let sem = self.semaphore.clone();
                let results = self.results.clone();
                let process_fn = process_fn.clone();
                let data_clone = data.clone();
                
                tokio::spawn(async move {
                    let _permit = sem.acquire().await.unwrap();
                    
                    let result = process_fn(data_clone.clone()).await;
                    
                    let work_result = WorkResult {
                        id,
                        input: data_clone,
                        result,
                    };
                    
                    results.lock().await.push(work_result);
                })
            })
            .collect();
        
        join_all(tasks).await;
        
        let mut results = self.results.lock().await;
        std::mem::take(&mut *results)
    }
}

// Example usage
#[tokio::test]
async fn test_semaphore_pool() {
    async fn expensive_computation(n: u64) -> Result<u64> {
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        Ok(n.pow(3))
    }
    
    let pool = SemaphoreWorkerPool::new(3);
    
    let items: Vec<_> = (0..20).map(|i| (i, i as u64)).collect();
    let results = pool.process_batch(items, expensive_computation).await;
    
    assert_eq!(results.len(), 20);
}
```

### Approach 3: Actor Model with tokio::task

This approach implements an actor-based pattern where each worker is an independent actor.

```rust
use tokio::sync::mpsc;
use tokio::task::JoinSet;

#[derive(Debug)]
enum WorkerMessage<T, R> {
    Process { id: usize, data: T, response_tx: mpsc::Sender<WorkResult<T, R>> },
    Shutdown,
}

pub struct ActorWorkerPool<T, R> {
    workers: Vec<mpsc::Sender<WorkerMessage<T, R>>>,
    join_set: JoinSet<()>,
    results_tx: mpsc::Sender<WorkResult<T, R>>,
    results_rx: mpsc::Receiver<WorkResult<T, R>>,
}

impl<T, R> ActorWorkerPool<T, R>
where
    T: Send + Clone + 'static,
    R: Send + 'static,
{
    pub fn new<F, Fut>(worker_count: usize, process_fn: F) -> Self
    where
        F: Fn(T) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<R>> + Send + 'static,
    {
        let mut workers = Vec::with_capacity(worker_count);
        let mut join_set = JoinSet::new();
        let (results_tx, results_rx) = mpsc::channel(100);
        
        for worker_id in 0..worker_count {
            let (tx, mut rx) = mpsc::channel::<WorkerMessage<T, R>>(10);
            let process_fn = process_fn.clone();
            
            join_set.spawn(async move {
                while let Some(msg) = rx.recv().await {
                    match msg {
                        WorkerMessage::Process { id, data, response_tx } => {
                            let result = process_fn(data.clone()).await;
                            let work_result = WorkResult { id, input: data, result };
                            let _ = response_tx.send(work_result).await;
                        }
                        WorkerMessage::Shutdown => break,
                    }
                }
                tracing::debug!("Worker {} shutting down", worker_id);
            });
            
            workers.push(tx);
        }
        
        Self {
            workers,
            join_set,
            results_tx,
            results_rx,
        }
    }
    
    pub async fn submit(&self, id: usize, data: T) -> Result<()> {
        let worker_idx = id % self.workers.len();
        let msg = WorkerMessage::Process {
            id,
            data,
            response_tx: self.results_tx.clone(),
        };
        
        self.workers[worker_idx].send(msg).await
            .map_err(|_| anyhow::anyhow!("Worker unavailable"))
    }
    
    pub async fn collect_results(&mut self) -> Vec<WorkResult<T, R>> {
        let mut results = Vec::new();
        while let Some(result) = self.results_rx.recv().await {
            results.push(result);
        }
        results
    }
    
    pub async fn shutdown(mut self) -> Result<()> {
        for worker in &self.workers {
            let _ = worker.send(WorkerMessage::Shutdown).await;
        }
        
        while let Some(res) = self.join_set.join_next().await {
            res?;
        }
        
        Ok(())
    }
}
```

### Approach 4: Stream-based Processing

This approach uses futures streams for elegant composition and processing.

```rust
use futures::stream::{self, StreamExt, TryStreamExt};
use futures::Stream;

pub struct StreamWorkerPool;

impl StreamWorkerPool {
    pub fn process_stream<T, R, F, Fut>(
        items: impl Stream<Item = (usize, T)> + Send + 'static,
        concurrency: usize,
        process_fn: F,
    ) -> impl Stream<Item = WorkResult<T, R>>
    where
        T: Send + Clone + 'static,
        R: Send + 'static,
        F: Fn(T) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<R>> + Send + 'static,
    {
        items
            .map(move |(id, data)| {
                let process_fn = process_fn.clone();
                let data_clone = data.clone();
                
                async move {
                    let result = process_fn(data_clone.clone()).await;
                    WorkResult {
                        id,
                        input: data_clone,
                        result,
                    }
                }
            })
            .buffer_unordered(concurrency)
    }
    
    pub async fn process_vec<T, R, F, Fut>(
        items: Vec<(usize, T)>,
        concurrency: usize,
        process_fn: F,
    ) -> Result<Vec<WorkResult<T, R>>>
    where
        T: Send + Clone + 'static,
        R: Send + 'static,
        F: Fn(T) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<R>> + Send + 'static,
    {
        let stream = stream::iter(items);
        let results: Vec<_> = Self::process_stream(stream, concurrency, process_fn)
            .collect()
            .await;
        
        Ok(results)
    }
}

// Example with error handling and retry logic
pub struct RetryStreamWorkerPool;

impl RetryStreamWorkerPool {
    pub fn process_with_retry<T, R, F, Fut>(
        items: impl Stream<Item = (usize, T)> + Send + 'static,
        concurrency: usize,
        max_retries: usize,
        process_fn: F,
    ) -> impl Stream<Item = WorkResult<T, R>>
    where
        T: Send + Clone + 'static,
        R: Send + 'static,
        F: Fn(T) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<R>> + Send + 'static,
    {
        items
            .map(move |(id, data)| {
                let process_fn = process_fn.clone();
                let data_clone = data.clone();
                
                async move {
                    let mut last_error = None;
                    
                    for attempt in 0..=max_retries {
                        match process_fn(data_clone.clone()).await {
                            Ok(result) => {
                                return WorkResult {
                                    id,
                                    input: data_clone,
                                    result: Ok(result),
                                };
                            }
                            Err(e) => {
                                last_error = Some(e);
                                if attempt < max_retries {
                                    tokio::time::sleep(
                                        tokio::time::Duration::from_millis(100 * (attempt as u64 + 1))
                                    ).await;
                                }
                            }
                        }
                    }
                    
                    WorkResult {
                        id,
                        input: data_clone,
                        result: Err(last_error.unwrap()),
                    }
                }
            })
            .buffer_unordered(concurrency)
    }
}
```

### Approach 5: Rayon with Async Bridge

This approach bridges Rayon's work-stealing threadpool with async operations.

```rust
use rayon::prelude::*;
use tokio::runtime::Handle;

pub struct RayonAsyncWorkerPool;

impl RayonAsyncWorkerPool {
    pub async fn process_parallel<T, R, F, Fut>(
        items: Vec<(usize, T)>,
        process_fn: F,
    ) -> Vec<WorkResult<T, R>>
    where
        T: Send + Sync + Clone + 'static,
        R: Send + Sync + 'static,
        F: Fn(T) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<R>> + Send + 'static,
    {
        let handle = Handle::current();
        
        tokio::task::spawn_blocking(move || {
            items
                .into_par_iter()
                .map(|(id, data)| {
                    let data_clone = data.clone();
                    let result = handle.block_on(process_fn(data_clone.clone()));
                    
                    WorkResult {
                        id,
                        input: data_clone,
                        result,
                    }
                })
                .collect()
        })
        .await
        .unwrap()
    }
    
    pub async fn process_parallel_chunked<T, R, F, Fut>(
        items: Vec<(usize, T)>,
        chunk_size: usize,
        process_fn: F,
    ) -> Vec<WorkResult<T, R>>
    where
        T: Send + Sync + Clone + 'static,
        R: Send + Sync + 'static,
        F: Fn(T) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<R>> + Send + 'static,
    {
        let chunks: Vec<_> = items.chunks(chunk_size).map(|c| c.to_vec()).collect();
        let process_fn = Arc::new(process_fn);
        
        let tasks: Vec<_> = chunks
            .into_iter()
            .map(|chunk| {
                let process_fn = process_fn.clone();
                tokio::spawn(async move {
                    let mut results = Vec::new();
                    for (id, data) in chunk {
                        let result = process_fn(data.clone()).await;
                        results.push(WorkResult {
                            id,
                            input: data,
                            result,
                        });
                    }
                    results
                })
            })
            .collect();
        
        let mut all_results = Vec::new();
        for task in tasks {
            all_results.extend(task.await.unwrap());
        }
        
        all_results
    }
}
```

## Performance Comparison

```rust
use criterion::{criterion_group, criterion_main, Criterion};

async fn benchmark_worker_pools(c: &mut Criterion) {
    let work_items: Vec<_> = (0..1000).map(|i| (i, i as u64)).collect();
    
    async fn work_fn(n: u64) -> Result<u64> {
        tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
        Ok(n * n)
    }
    
    c.bench_function("channel_based", |b| {
        b.to_async(&tokio::runtime::Runtime::new().unwrap())
            .iter(|| async {
                let pool = ChannelWorkerPool::new(8, work_fn);
                for (id, data) in &work_items {
                    pool.submit(*id, *data).await.unwrap();
                }
                pool.shutdown().await.unwrap()
            });
    });
    
    c.bench_function("semaphore_based", |b| {
        b.to_async(&tokio::runtime::Runtime::new().unwrap())
            .iter(|| async {
                let pool = SemaphoreWorkerPool::new(8);
                pool.process_batch(work_items.clone(), work_fn).await
            });
    });
    
    c.bench_function("stream_based", |b| {
        b.to_async(&tokio::runtime::Runtime::new().unwrap())
            .iter(|| async {
                StreamWorkerPool::process_vec(work_items.clone(), 8, work_fn).await.unwrap()
            });
    });
}
```

## Best Practices

### 1. Choose the Right Approach

- **Channel-based**: Best for long-running services with continuous work submission
- **Semaphore-based**: Ideal for batch processing with controlled concurrency
- **Actor-based**: Great for stateful workers or complex routing logic
- **Stream-based**: Perfect for functional composition and data pipelines
- **Rayon-async**: Optimal for CPU-bound tasks with async I/O

### 2. Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum WorkerPoolError {
    #[error("Worker pool shutdown")]
    Shutdown,
    #[error("Task submission failed")]
    SubmissionFailed,
    #[error("Processing error: {0}")]
    ProcessingError(#[from] anyhow::Error),
}

// Implement graceful error recovery
pub trait ErrorRecovery<T, R> {
    async fn process_with_recovery<F, Fut>(
        &self,
        items: Vec<(usize, T)>,
        process_fn: F,
        recovery_fn: impl Fn(&T, &anyhow::Error) -> Option<R>,
    ) -> Vec<WorkResult<T, R>>
    where
        F: Fn(T) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<R>> + Send + 'static;
}
```

### 3. Monitoring and Observability

```rust
use tracing::{info, warn, error, instrument};

#[instrument(skip(process_fn))]
pub async fn monitored_process<T, R, F, Fut>(
    id: usize,
    data: T,
    process_fn: F,
) -> WorkResult<T, R>
where
    T: Send + Clone + std::fmt::Debug + 'static,
    R: Send + 'static,
    F: Fn(T) -> Fut,
    Fut: std::future::Future<Output = Result<R>>,
{
    info!("Processing task {}", id);
    let start = tokio::time::Instant::now();
    
    let result = process_fn(data.clone()).await;
    let duration = start.elapsed();
    
    match &result {
        Ok(_) => info!("Task {} completed in {:?}", id, duration),
        Err(e) => error!("Task {} failed: {}", id, e),
    }
    
    WorkResult { id, input: data, result }
}
```

### 4. Resource Management

```rust
pub struct ResourceBoundedPool<T, R> {
    memory_limit: usize,
    cpu_limit: f64,
    inner: Box<dyn WorkerPool<T, R>>,
}

impl<T, R> ResourceBoundedPool<T, R> {
    pub async fn submit_if_resources_available(&self, id: usize, data: T) -> Result<()> {
        if self.check_resources().await? {
            self.inner.submit(id, data).await
        } else {
            Err(anyhow::anyhow!("Insufficient resources"))
        }
    }
}
```

## Running the Examples

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rust-async-worker-pool
cd rust-async-worker-pool
```

2. Run tests:
```bash
cargo test
```

3. Run benchmarks:
```bash
cargo bench
```

4. Run specific example:
```bash
cargo run --example channel_worker_pool
cargo run --example stream_processing
```

## Advanced Topics (Future Tutorials)

- Dynamic worker scaling based on load
- Persistent job queues with database backing
- Distributed worker pools across multiple machines
- Integration with message brokers (RabbitMQ, Kafka)
- Priority-based task scheduling
- Graceful shutdown and draining strategies

## Contributing

Feel free to submit issues and enhancement requests!

## License

This tutorial is provided under the MIT License.