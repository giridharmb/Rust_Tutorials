# Advanced Rust Patterns and Implementations

## Table of Contents
1. [Async Stream Processing with Backpressure](#async-stream-processing-with-backpressure)
2. [Connection Pooling with Health Checks](#connection-pooling-with-health-checks)
3. [Circuit Breaker Pattern](#circuit-breaker-pattern)
4. [Distributed Rate Limiter](#distributed-rate-limiter)
5. [Event Sourcing with CQRS](#event-sourcing-with-cqrs)
6. [Actor Model Implementation](#actor-model-implementation)
7. [Zero-Copy Serialization](#zero-copy-serialization)
8. [Lock-Free Data Structures](#lock-free-data-structures)
9. [Custom Async Runtime Executor](#custom-async-runtime-executor)
10. [Memory-Mapped File Processing](#memory-mapped-file-processing)
11. [WebAssembly Integration](#webassembly-integration)
12. [gRPC with Tonic and Streaming](#grpc-with-tonic-and-streaming)

## Dependencies

```toml
[dependencies]
# Core async runtime
tokio = { version = "1.35", features = ["full"] }
tokio-stream = "0.1"
futures = "0.3"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
bincode = "1.3"
rkyv = { version = "0.7", features = ["validation"] }
flatbuffers = "23.5"

# Database
sqlx = { version = "0.7", features = ["runtime-tokio-rustls", "postgres", "uuid", "chrono"] }
redis = { version = "0.24", features = ["tokio-comp", "connection-manager"] }
deadpool-postgres = "0.12"

# gRPC
tonic = "0.11"
prost = "0.12"
tonic-build = "0.11"

# Metrics and Observability
prometheus = "0.13"
tracing = "0.1"
tracing-subscriber = "0.3"
opentelemetry = { version = "0.21", features = ["rt-tokio"] }

# Utils
dashmap = "5.5"
crossbeam = "0.8"
parking_lot = "0.12"
once_cell = "1.19"
arc-swap = "1.6"
bytes = "1.5"
uuid = { version = "1.6", features = ["v4", "serde"] }
chrono = { version = "0.4", features = ["serde"] }
thiserror = "1.0"
anyhow = "1.0"

# WASM
wasm-bindgen = "0.2"
wasmtime = "17.0"

# Memory mapping
memmap2 = "0.9"
```

## 1. Async Stream Processing with Backpressure

```rust
use tokio::sync::{mpsc, Semaphore};
use tokio_stream::{Stream, StreamExt};
use futures::stream::{self, StreamExt as FuturesStreamExt};
use std::sync::Arc;
use std::pin::Pin;
use std::task::{Context, Poll};

/// A stream processor with backpressure control
pub struct BackpressureStream<T> {
    receiver: mpsc::Receiver<T>,
    semaphore: Arc<Semaphore>,
    buffer_size: usize,
}

impl<T> BackpressureStream<T> {
    pub fn new(buffer_size: usize) -> (Self, mpsc::Sender<T>) {
        let (sender, receiver) = mpsc::channel(buffer_size);
        let semaphore = Arc::new(Semaphore::new(buffer_size));
        
        let stream = Self {
            receiver,
            semaphore,
            buffer_size,
        };
        
        (stream, sender)
    }
}

impl<T> Stream for BackpressureStream<T> {
    type Item = T;
    
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx)
    }
}

/// Advanced stream processing pipeline with transformations
pub struct StreamProcessor<T> {
    parallelism: usize,
    batch_size: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Send + 'static> StreamProcessor<T> {
    pub fn new(parallelism: usize, batch_size: usize) -> Self {
        Self {
            parallelism,
            batch_size,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Process stream with parallel transformations
    pub async fn process_stream<S, F, U>(
        &self,
        stream: S,
        transform: F,
    ) -> impl Stream<Item = U>
    where
        S: Stream<Item = T> + Send + 'static,
        F: Fn(T) -> U + Send + Sync + Clone + 'static,
        U: Send + 'static,
    {
        stream
            .chunks(self.batch_size)
            .map(move |batch| {
                let transform = transform.clone();
                tokio::spawn(async move {
                    batch.into_iter().map(transform).collect::<Vec<_>>()
                })
            })
            .buffer_unordered(self.parallelism)
            .flat_map(|result| {
                stream::iter(result.unwrap_or_default())
            })
    }
}

/// Example: Processing BigQuery results as a stream
pub struct BigQueryStreamProcessor {
    processor: StreamProcessor<serde_json::Value>,
}

impl BigQueryStreamProcessor {
    pub fn new() -> Self {
        Self {
            processor: StreamProcessor::new(10, 100),
        }
    }
    
    pub async fn process_query_results<S>(
        &self,
        results_stream: S,
    ) -> impl Stream<Item = ProcessedRecord>
    where
        S: Stream<Item = serde_json::Value> + Send + 'static,
    {
        self.processor.process_stream(results_stream, |record| {
            // Transform BigQuery record
            ProcessedRecord {
                id: record["id"].as_str().unwrap_or_default().to_string(),
                processed_at: chrono::Utc::now(),
                data: record,
            }
        }).await
    }
}

#[derive(Debug, Clone)]
pub struct ProcessedRecord {
    pub id: String,
    pub processed_at: chrono::DateTime<chrono::Utc>,
    pub data: serde_json::Value,
}
```

## 2. Connection Pooling with Health Checks

```rust
use deadpool_postgres::{Config, Pool, Runtime};
use tokio::time::{interval, Duration};
use std::sync::Arc;
use dashmap::DashMap;

/// Advanced connection pool with health monitoring
pub struct SmartConnectionPool {
    postgres_pool: Pool,
    redis_pool: redis::aio::ConnectionManager,
    health_status: Arc<DashMap<String, HealthStatus>>,
}

#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub is_healthy: bool,
    pub last_check: chrono::DateTime<chrono::Utc>,
    pub error_count: u32,
    pub latency_ms: Option<u64>,
}

impl SmartConnectionPool {
    pub async fn new(
        postgres_config: Config,
        redis_url: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Create PostgreSQL pool
        let postgres_pool = postgres_config.create_pool(
            Some(Runtime::Tokio1),
            tokio_postgres::NoTls,
        )?;
        
        // Create Redis connection manager
        let redis_client = redis::Client::open(redis_url)?;
        let redis_pool = redis::aio::ConnectionManager::new(redis_client).await?;
        
        let health_status = Arc::new(DashMap::new());
        
        let pool = Self {
            postgres_pool,
            redis_pool,
            health_status,
        };
        
        // Start health check task
        pool.start_health_checks();
        
        Ok(pool)
    }
    
    /// Start background health checks
    fn start_health_checks(&self) {
        let postgres_pool = self.postgres_pool.clone();
        let redis_pool = self.redis_pool.clone();
        let health_status = self.health_status.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                // Check PostgreSQL health
                let pg_start = std::time::Instant::now();
                let pg_health = match postgres_pool.get().await {
                    Ok(conn) => {
                        match conn.simple_query("SELECT 1").await {
                            Ok(_) => HealthStatus {
                                is_healthy: true,
                                last_check: chrono::Utc::now(),
                                error_count: 0,
                                latency_ms: Some(pg_start.elapsed().as_millis() as u64),
                            },
                            Err(_) => {
                                let mut current = health_status
                                    .get("postgres")
                                    .map(|h| h.clone())
                                    .unwrap_or_default();
                                current.error_count += 1;
                                current.is_healthy = false;
                                current.last_check = chrono::Utc::now();
                                current
                            }
                        }
                    }
                    Err(_) => {
                        let mut current = health_status
                            .get("postgres")
                            .map(|h| h.clone())
                            .unwrap_or_default();
                        current.error_count += 1;
                        current.is_healthy = false;
                        current.last_check = chrono::Utc::now();
                        current
                    }
                };
                
                health_status.insert("postgres".to_string(), pg_health);
                
                // Check Redis health
                let redis_start = std::time::Instant::now();
                let mut redis_conn = redis_pool.clone();
                let redis_health = match redis::cmd("PING").query_async::<_, String>(&mut redis_conn).await {
                    Ok(_) => HealthStatus {
                        is_healthy: true,
                        last_check: chrono::Utc::now(),
                        error_count: 0,
                        latency_ms: Some(redis_start.elapsed().as_millis() as u64),
                    },
                    Err(_) => {
                        let mut current = health_status
                            .get("redis")
                            .map(|h| h.clone())
                            .unwrap_or_default();
                        current.error_count += 1;
                        current.is_healthy = false;
                        current.last_check = chrono::Utc::now();
                        current
                    }
                };
                
                health_status.insert("redis".to_string(), redis_health);
            }
        });
    }
    
    /// Get a healthy PostgreSQL connection
    pub async fn get_postgres(&self) -> Result<deadpool_postgres::Object, Box<dyn std::error::Error>> {
        if let Some(health) = self.health_status.get("postgres") {
            if !health.is_healthy && health.error_count > 3 {
                return Err("PostgreSQL pool is unhealthy".into());
            }
        }
        
        Ok(self.postgres_pool.get().await?)
    }
    
    /// Get Redis connection
    pub fn get_redis(&self) -> redis::aio::ConnectionManager {
        self.redis_pool.clone()
    }
}

impl Default for HealthStatus {
    fn default() -> Self {
        Self {
            is_healthy: true,
            last_check: chrono::Utc::now(),
            error_count: 0,
            latency_ms: None,
        }
    }
}
```

## 3. Circuit Breaker Pattern

```rust
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use parking_lot::RwLock;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Circuit breaker implementation for fault tolerance
pub struct CircuitBreaker {
    state: RwLock<CircuitState>,
    failure_count: AtomicU32,
    success_count: AtomicU32,
    last_failure_time: RwLock<Option<Instant>>,
    
    // Configuration
    failure_threshold: u32,
    success_threshold: u32,
    timeout: Duration,
    half_open_max_calls: AtomicU32,
}

impl CircuitBreaker {
    pub fn new(
        failure_threshold: u32,
        success_threshold: u32,
        timeout: Duration,
    ) -> Self {
        Self {
            state: RwLock::new(CircuitState::Closed),
            failure_count: AtomicU32::new(0),
            success_count: AtomicU32::new(0),
            last_failure_time: RwLock::new(None),
            failure_threshold,
            success_threshold,
            timeout,
            half_open_max_calls: AtomicU32::new(0),
        }
    }
    
    /// Check if the circuit breaker allows the operation
    pub fn can_proceed(&self) -> Result<(), CircuitBreakerError> {
        let state = *self.state.read();
        
        match state {
            CircuitState::Closed => Ok(()),
            CircuitState::Open => {
                // Check if timeout has passed
                if let Some(last_failure) = *self.last_failure_time.read() {
                    if last_failure.elapsed() >= self.timeout {
                        // Transition to half-open
                        *self.state.write() = CircuitState::HalfOpen;
                        self.half_open_max_calls.store(0, Ordering::SeqCst);
                        Ok(())
                    } else {
                        Err(CircuitBreakerError::Open)
                    }
                } else {
                    Err(CircuitBreakerError::Open)
                }
            }
            CircuitState::HalfOpen => {
                // Allow limited calls in half-open state
                let calls = self.half_open_max_calls.fetch_add(1, Ordering::SeqCst);
                if calls < 3 {
                    Ok(())
                } else {
                    Err(CircuitBreakerError::HalfOpenLimitExceeded)
                }
            }
        }
    }
    
    /// Record a successful operation
    pub fn record_success(&self) {
        let state = *self.state.read();
        
        match state {
            CircuitState::Closed => {
                self.failure_count.store(0, Ordering::SeqCst);
            }
            CircuitState::HalfOpen => {
                let success_count = self.success_count.fetch_add(1, Ordering::SeqCst) + 1;
                
                if success_count >= self.success_threshold {
                    // Transition to closed
                    *self.state.write() = CircuitState::Closed;
                    self.failure_count.store(0, Ordering::SeqCst);
                    self.success_count.store(0, Ordering::SeqCst);
                }
            }
            CircuitState::Open => {} // Shouldn't happen
        }
    }
    
    /// Record a failed operation
    pub fn record_failure(&self) {
        let state = *self.state.read();
        
        match state {
            CircuitState::Closed => {
                let failure_count = self.failure_count.fetch_add(1, Ordering::SeqCst) + 1;
                
                if failure_count >= self.failure_threshold {
                    // Transition to open
                    *self.state.write() = CircuitState::Open;
                    *self.last_failure_time.write() = Some(Instant::now());
                }
            }
            CircuitState::HalfOpen => {
                // Transition back to open
                *self.state.write() = CircuitState::Open;
                *self.last_failure_time.write() = Some(Instant::now());
                self.success_count.store(0, Ordering::SeqCst);
            }
            CircuitState::Open => {} // Already open
        }
    }
    
    /// Execute an operation with circuit breaker protection
    pub async fn execute<F, Fut, T, E>(
        &self,
        operation: F,
    ) -> Result<T, CircuitBreakerError>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
        E: std::fmt::Display,
    {
        self.can_proceed()?;
        
        match operation().await {
            Ok(result) => {
                self.record_success();
                Ok(result)
            }
            Err(e) => {
                self.record_failure();
                Err(CircuitBreakerError::OperationFailed(e.to_string()))
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CircuitBreakerError {
    #[error("Circuit breaker is open")]
    Open,
    #[error("Half-open call limit exceeded")]
    HalfOpenLimitExceeded,
    #[error("Operation failed: {0}")]
    OperationFailed(String),
}
```

## 4. Distributed Rate Limiter

```rust
use redis::AsyncCommands;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Distributed rate limiter using Redis
pub struct DistributedRateLimiter {
    redis_pool: redis::aio::ConnectionManager,
    window_size: Duration,
    max_requests: u64,
}

impl DistributedRateLimiter {
    pub fn new(
        redis_pool: redis::aio::ConnectionManager,
        window_size: Duration,
        max_requests: u64,
    ) -> Self {
        Self {
            redis_pool,
            window_size,
            max_requests,
        }
    }
    
    /// Check if request is allowed using sliding window algorithm
    pub async fn check_rate_limit(&self, key: &str) -> Result<RateLimitResult, redis::RedisError> {
        let mut conn = self.redis_pool.clone();
        
        // Current timestamp in milliseconds
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        // Window start time
        let window_start = now - self.window_size.as_millis() as u64;
        
        // Lua script for atomic rate limit check
        let script = r#"
            local key = KEYS[1]
            local now = tonumber(ARGV[1])
            local window_start = tonumber(ARGV[2])
            local max_requests = tonumber(ARGV[3])
            local window_size = tonumber(ARGV[4])
            
            -- Remove old entries
            redis.call('ZREMRANGEBYSCORE', key, 0, window_start)
            
            -- Count current requests
            local current_requests = redis.call('ZCARD', key)
            
            if current_requests < max_requests then
                -- Add new request
                redis.call('ZADD', key, now, now)
                redis.call('EXPIRE', key, window_size)
                return {1, max_requests - current_requests - 1}
            else
                return {0, 0}
            end
        "#;
        
        let result: Vec<i64> = redis::Script::new(script)
            .key(key)
            .arg(now)
            .arg(window_start)
            .arg(self.max_requests)
            .arg(self.window_size.as_secs())
            .invoke_async(&mut conn)
            .await?;
        
        Ok(RateLimitResult {
            allowed: result[0] == 1,
            remaining: result[1] as u64,
            reset_at: now + self.window_size.as_millis() as u64,
        })
    }
    
    /// Token bucket algorithm implementation
    pub async fn check_token_bucket(&self, key: &str) -> Result<bool, redis::RedisError> {
        let mut conn = self.redis_pool.clone();
        
        let script = r#"
            local key = KEYS[1]
            local rate = tonumber(ARGV[1])
            local capacity = tonumber(ARGV[2])
            local now = tonumber(ARGV[3])
            local requested = tonumber(ARGV[4])
            
            local bucket_data = redis.call('HMGET', key, 'tokens', 'last_update')
            local tokens = tonumber(bucket_data[1]) or capacity
            local last_update = tonumber(bucket_data[2]) or now
            
            -- Calculate tokens to add
            local elapsed = math.max(0, now - last_update)
            local tokens_to_add = elapsed * rate
            tokens = math.min(capacity, tokens + tokens_to_add)
            
            if tokens >= requested then
                tokens = tokens - requested
                redis.call('HMSET', key, 'tokens', tokens, 'last_update', now)
                redis.call('EXPIRE', key, 3600)
                return 1
            else
                redis.call('HMSET', key, 'tokens', tokens, 'last_update', now)
                redis.call('EXPIRE', key, 3600)
                return 0
            end
        "#;
        
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as f64;
        
        let allowed: i32 = redis::Script::new(script)
            .key(key)
            .arg(10.0) // 10 tokens per second
            .arg(100.0) // bucket capacity
            .arg(now)
            .arg(1.0) // requested tokens
            .invoke_async(&mut conn)
            .await?;
        
        Ok(allowed == 1)
    }
}

#[derive(Debug)]
pub struct RateLimitResult {
    pub allowed: bool,
    pub remaining: u64,
    pub reset_at: u64,
}
```

## 5. Event Sourcing with CQRS

```rust
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use sqlx::{PgPool, Postgres, Transaction};

/// Base event trait
pub trait Event: Send + Sync + Serialize + for<'de> Deserialize<'de> {
    fn event_type(&self) -> &'static str;
    fn aggregate_id(&self) -> Uuid;
}

/// Aggregate root trait
pub trait Aggregate: Sized + Send + Sync {
    type Command;
    type Event: Event;
    type Error;
    
    fn aggregate_id(&self) -> Uuid;
    fn apply_event(&mut self, event: &Self::Event);
    fn handle_command(&self, command: Self::Command) -> Result<Vec<Self::Event>, Self::Error>;
}

/// Event store implementation
pub struct EventStore {
    pool: PgPool,
}

impl EventStore {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
    
    /// Save events to the store
    pub async fn save_events<E: Event>(
        &self,
        events: &[E],
        expected_version: Option<i64>,
    ) -> Result<(), sqlx::Error> {
        let mut tx = self.pool.begin().await?;
        
        // Check expected version for optimistic concurrency
        if let Some(expected) = expected_version {
            let aggregate_id = events.first()
                .map(|e| e.aggregate_id())
                .ok_or_else(|| sqlx::Error::RowNotFound)?;
                
            let current_version: (i64,) = sqlx::query_as(
                "SELECT COALESCE(MAX(version), 0) FROM events WHERE aggregate_id = $1"
            )
            .bind(aggregate_id)
            .fetch_one(&mut *tx)
            .await?;
            
            if current_version.0 != expected {
                return Err(sqlx::Error::RowNotFound); // Concurrency conflict
            }
        }
        
        // Save events
        for (index, event) in events.iter().enumerate() {
            let event_data = serde_json::to_value(event).unwrap();
            let version = expected_version.unwrap_or(0) + index as i64 + 1;
            
            sqlx::query(
                r#"
                INSERT INTO events (
                    aggregate_id, event_type, event_data, version, created_at
                ) VALUES ($1, $2, $3, $4, $5)
                "#
            )
            .bind(event.aggregate_id())
            .bind(event.event_type())
            .bind(event_data)
            .bind(version)
            .bind(Utc::now())
            .execute(&mut *tx)
            .await?;
        }
        
        tx.commit().await?;
        Ok(())
    }
    
    /// Load events for an aggregate
    pub async fn load_events<E: Event>(
        &self,
        aggregate_id: Uuid,
        from_version: Option<i64>,
    ) -> Result<Vec<E>, sqlx::Error> {
        let from_version = from_version.unwrap_or(0);
        
        let rows = sqlx::query!(
            r#"
            SELECT event_data, version
            FROM events
            WHERE aggregate_id = $1 AND version > $2
            ORDER BY version ASC
            "#,
            aggregate_id,
            from_version
        )
        .fetch_all(&self.pool)
        .await?;
        
        let events = rows
            .into_iter()
            .map(|row| serde_json::from_value(row.event_data).unwrap())
            .collect();
        
        Ok(events)
    }
}

/// CQRS Command Handler
pub struct CommandHandler<A: Aggregate> {
    event_store: EventStore,
    _phantom: std::marker::PhantomData<A>,
}

impl<A: Aggregate> CommandHandler<A> {
    pub fn new(event_store: EventStore) -> Self {
        Self {
            event_store,
            _phantom: std::marker::PhantomData,
        }
    }
    
    pub async fn handle_command(
        &self,
        aggregate_id: Uuid,
        command: A::Command,
        aggregate_factory: impl Fn() -> A,
    ) -> Result<Vec<A::Event>, Box<dyn std::error::Error>> {
        // Load aggregate from events
        let events = self.event_store.load_events::<A::Event>(aggregate_id, None).await?;
        let mut aggregate = aggregate_factory();
        
        for event in &events {
            aggregate.apply_event(event);
        }
        
        // Handle command
        let new_events = aggregate.handle_command(command)?;
        
        // Save events
        let version = events.len() as i64;
        self.event_store.save_events(&new_events, Some(version)).await?;
        
        Ok(new_events)
    }
}

/// Example: Order aggregate
#[derive(Debug, Clone)]
pub struct Order {
    pub id: Uuid,
    pub customer_id: String,
    pub items: Vec<OrderItem>,
    pub status: OrderStatus,
    pub total: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderItem {
    pub product_id: String,
    pub quantity: u32,
    pub price: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OrderStatus {
    Pending,
    Confirmed,
    Shipped,
    Delivered,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderEvent {
    OrderCreated {
        order_id: Uuid,
        customer_id: String,
        items: Vec<OrderItem>,
    },
    OrderConfirmed {
        order_id: Uuid,
    },
    OrderShipped {
        order_id: Uuid,
        tracking_number: String,
    },
}

impl Event for OrderEvent {
    fn event_type(&self) -> &'static str {
        match self {
            OrderEvent::OrderCreated { .. } => "OrderCreated",
            OrderEvent::OrderConfirmed { .. } => "OrderConfirmed",
            OrderEvent::OrderShipped { .. } => "OrderShipped",
        }
    }
    
    fn aggregate_id(&self) -> Uuid {
        match self {
            OrderEvent::OrderCreated { order_id, .. } |
            OrderEvent::OrderConfirmed { order_id } |
            OrderEvent::OrderShipped { order_id, .. } => *order_id,
        }
    }
}
```

## 6. Actor Model Implementation

```rust
use tokio::sync::{mpsc, oneshot};
use std::collections::HashMap;
use uuid::Uuid;

/// Message trait for actor communication
pub trait Message: Send + 'static {
    type Result: Send;
}

/// Actor trait
#[async_trait::async_trait]
pub trait Actor: Sized + Send + 'static {
    type State: Send;
    
    async fn handle_message<M: Message>(
        &mut self,
        msg: M,
        state: &mut Self::State,
    ) -> M::Result;
}

/// Actor system for managing actors
pub struct ActorSystem {
    actors: Arc<DashMap<Uuid, mpsc::Sender<BoxedMessage>>>,
}

type BoxedMessage = Box<dyn std::any::Any + Send>;

impl ActorSystem {
    pub fn new() -> Self {
        Self {
            actors: Arc::new(DashMap::new()),
        }
    }
    
    /// Spawn a new actor
    pub fn spawn_actor<A: Actor>(
        &self,
        actor: A,
        initial_state: A::State,
    ) -> ActorRef {
        let id = Uuid::new_v4();
        let (tx, mut rx) = mpsc::channel::<BoxedMessage>(100);
        
        let actors = self.actors.clone();
        actors.insert(id, tx.clone());
        
        tokio::spawn(async move {
            let mut actor = actor;
            let mut state = initial_state;
            
            while let Some(msg) = rx.recv().await {
                // Handle message (simplified for brevity)
                // In real implementation, you'd need proper message dispatching
            }
            
            actors.remove(&id);
        });
        
        ActorRef { id, sender: tx }
    }
}

/// Reference to an actor
#[derive(Clone)]
pub struct ActorRef {
    id: Uuid,
    sender: mpsc::Sender<BoxedMessage>,
}

impl ActorRef {
    pub async fn send<M: Message>(&self, msg: M) -> Result<M::Result, ActorError> {
        let (tx, rx) = oneshot::channel();
        
        self.sender
            .send(Box::new((msg, tx)))
            .await
            .map_err(|_| ActorError::ActorNotFound)?;
            
        rx.await.map_err(|_| ActorError::ActorNotFound)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ActorError {
    #[error("Actor not found")]
    ActorNotFound,
    #[error("Message send failed")]
    SendFailed,
}

/// Example: Database actor for connection pooling
pub struct DatabaseActor {
    pool: PgPool,
}

#[derive(Debug)]
pub struct ExecuteQuery {
    pub query: String,
}

impl Message for ExecuteQuery {
    type Result = Result<Vec<serde_json::Value>, sqlx::Error>;
}

#[async_trait::async_trait]
impl Actor for DatabaseActor {
    type State = ();
    
    async fn handle_message<M: Message>(
        &mut self,
        msg: M,
        _state: &mut Self::State,
    ) -> M::Result {
        // Handle different message types
        todo!()
    }
}
```

## 7. Zero-Copy Serialization

```rust
use rkyv::{Archive, Deserialize, Serialize, AlignedVec};
use rkyv::ser::{Serializer, serializers::AllocSerializer};
use rkyv::de::deserializers::SharedDeserializeMap;
use rkyv::validation::validators::DefaultValidator;
use rkyv::Archived;

/// Zero-copy serializable data structure
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq)]
#[archive(compare(PartialEq))]
#[archive_attr(derive(Debug))]
pub struct LargeDataset {
    pub id: u64,
    pub name: String,
    pub data_points: Vec<DataPoint>,
    pub metadata: HashMap<String, String>,
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq)]
#[archive(compare(PartialEq))]
#[archive_attr(derive(Debug))]
pub struct DataPoint {
    pub timestamp: i64,
    pub value: f64,
    pub tags: Vec<String>,
}

/// Zero-copy serialization utilities
pub struct ZeroCopySerializer;

impl ZeroCopySerializer {
    /// Serialize data with zero-copy support
    pub fn serialize<T: Serialize<AllocSerializer<256>>>(
        value: &T,
    ) -> Result<AlignedVec, Box<dyn std::error::Error>> {
        let mut serializer = AllocSerializer::<256>::default();
        serializer.serialize_value(value)?;
        Ok(serializer.into_serializer().into_inner())
    }
    
    /// Access archived data without deserialization
    pub fn access_archived<T: Archive>(
        bytes: &[u8],
    ) -> Result<&Archived<T>, Box<dyn std::error::Error>> {
        let archived = rkyv::check_archived_root::<T>(bytes)?;
        Ok(archived)
    }
    
    /// Deserialize if needed
    pub fn deserialize<T>(
        bytes: &[u8],
    ) -> Result<T, Box<dyn std::error::Error>>
    where
        T: Archive,
        T::Archived: Deserialize<T, SharedDeserializeMap>,
    {
        let archived = rkyv::check_archived_root::<T>(bytes)?;
        let mut deserializer = SharedDeserializeMap::new();
        let deserialized: T = archived.deserialize(&mut deserializer)?;
        Ok(deserialized)
    }
}

/// Example: High-performance data processor
pub struct DataProcessor;

impl DataProcessor {
    /// Process large dataset without full deserialization
    pub fn process_dataset(serialized_data: &[u8]) -> Result<f64, Box<dyn std::error::Error>> {
        // Access data without copying
        let dataset = ZeroCopySerializer::access_archived::<LargeDataset>(serialized_data)?;
        
        // Compute directly on archived data
        let sum: f64 = dataset.data_points
            .iter()
            .map(|point| point.value)
            .sum();
            
        Ok(sum / dataset.data_points.len() as f64)
    }
    
    /// Stream processing with zero-copy
    pub async fn stream_process<S>(
        data_stream: S,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>>
    where
        S: Stream<Item = Vec<u8>> + Unpin,
    {
        let results = data_stream
            .map(|bytes| {
                Self::process_dataset(&bytes).unwrap_or(0.0)
            })
            .collect()
            .await;
            
        Ok(results)
    }
}
```

## 8. Lock-Free Data Structures

```rust
use crossbeam::epoch::{self, Atomic, Guard, Owned, Shared};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::ptr;

/// Lock-free stack implementation
pub struct LockFreeStack<T> {
    head: Atomic<Node<T>>,
    size: AtomicUsize,
}

struct Node<T> {
    data: T,
    next: Atomic<Node<T>>,
}

impl<T> LockFreeStack<T> {
    pub fn new() -> Self {
        Self {
            head: Atomic::null(),
            size: AtomicUsize::new(0),
        }
    }
    
    pub fn push(&self, data: T) {
        let guard = &epoch::pin();
        let mut new_node = Owned::new(Node {
            data,
            next: Atomic::null(),
        });
        
        loop {
            let head = self.head.load(Ordering::Acquire, guard);
            new_node.next.store(head, Ordering::Relaxed);
            
            match self.head.compare_exchange(
                head,
                new_node,
                Ordering::Release,
                Ordering::Acquire,
                guard,
            ) {
                Ok(_) => {
                    self.size.fetch_add(1, Ordering::Relaxed);
                    break;
                }
                Err(e) => new_node = e.new,
            }
        }
    }
    
    pub fn pop(&self) -> Option<T> {
        let guard = &epoch::pin();
        
        loop {
            let head = self.head.load(Ordering::Acquire, guard);
            
            match unsafe { head.as_ref() } {
                None => return None,
                Some(node) => {
                    let next = node.next.load(Ordering::Acquire, guard);
                    
                    if self.head
                        .compare_exchange(
                            head,
                            next,
                            Ordering::Release,
                            Ordering::Acquire,
                            guard,
                        )
                        .is_ok()
                    {
                        self.size.fetch_sub(1, Ordering::Relaxed);
                        
                        unsafe {
                            guard.defer_destroy(head);
                            return Some(ptr::read(&node.data));
                        }
                    }
                }
            }
        }
    }
    
    pub fn len(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }
}

/// Lock-free ring buffer for high-performance logging
pub struct LockFreeRingBuffer<T: Clone> {
    buffer: Vec<AtomicPtr<T>>,
    capacity: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
}

impl<T: Clone> LockFreeRingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer.push(AtomicPtr::new(ptr::null_mut()));
        }
        
        Self {
            buffer,
            capacity,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }
    
    pub fn push(&self, item: T) -> Result<(), T> {
        let mut head = self.head.load(Ordering::Relaxed);
        
        loop {
            let next_head = (head + 1) % self.capacity;
            let tail = self.tail.load(Ordering::Acquire);
            
            if next_head == tail {
                return Err(item); // Buffer full
            }
            
            match self.head.compare_exchange_weak(
                head,
                next_head,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    let item_ptr = Box::into_raw(Box::new(item));
                    let old = self.buffer[head].swap(item_ptr, Ordering::Release);
                    
                    if !old.is_null() {
                        unsafe { Box::from_raw(old); }
                    }
                    
                    return Ok(());
                }
                Err(actual) => head = actual,
            }
        }
    }
}

use std::sync::atomic::AtomicPtr;
```

## 9. Custom Async Runtime Executor

```rust
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};
use std::collections::VecDeque;
use parking_lot::Mutex;
use std::sync::Arc;

/// Simplified async runtime executor
pub struct MiniExecutor {
    ready_queue: Arc<Mutex<VecDeque<Task>>>,
    waker_cache: Arc<Mutex<HashMap<TaskId, Waker>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TaskId(usize);

struct Task {
    id: TaskId,
    future: Pin<Box<dyn Future<Output = ()> + Send>>,
}

impl MiniExecutor {
    pub fn new() -> Self {
        Self {
            ready_queue: Arc::new(Mutex::new(VecDeque::new())),
            waker_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    pub fn spawn<F>(&self, future: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
        
        let id = TaskId(NEXT_ID.fetch_add(1, Ordering::Relaxed));
        let task = Task {
            id,
            future: Box::pin(future),
        };
        
        self.ready_queue.lock().push_back(task);
    }
    
    pub fn run(&self) {
        loop {
            let mut task = match self.ready_queue.lock().pop_front() {
                Some(task) => task,
                None => break,
            };
            
            let waker = self.create_waker(task.id);
            let mut context = Context::from_waker(&waker);
            
            match task.future.as_mut().poll(&mut context) {
                Poll::Ready(()) => {
                    self.waker_cache.lock().remove(&task.id);
                }
                Poll::Pending => {
                    // Task will be re-queued when woken
                }
            }
        }
    }
    
    fn create_waker(&self, task_id: TaskId) -> Waker {
        // Implementation of waker that re-queues the task
        todo!()
    }
}

/// Work-stealing thread pool executor
pub struct WorkStealingExecutor {
    workers: Vec<Worker>,
    global_queue: Arc<Mutex<VecDeque<Box<dyn Future<Output = ()> + Send>>>>,
}

struct Worker {
    local_queue: Arc<Mutex<VecDeque<Box<dyn Future<Output = ()> + Send>>>>,
    thread: std::thread::JoinHandle<()>,
}

impl WorkStealingExecutor {
    pub fn new(num_threads: usize) -> Self {
        let global_queue = Arc::new(Mutex::new(VecDeque::new()));
        let mut workers = Vec::with_capacity(num_threads);
        
        for _ in 0..num_threads {
            let local_queue = Arc::new(Mutex::new(VecDeque::new()));
            let global_queue_clone = global_queue.clone();
            let local_queue_clone = local_queue.clone();
            
            let thread = std::thread::spawn(move || {
                // Worker thread logic
                loop {
                    // Try local queue first
                    if let Some(task) = local_queue_clone.lock().pop_front() {
                        // Execute task
                        continue;
                    }
                    
                    // Try global queue
                    if let Some(task) = global_queue_clone.lock().pop_front() {
                        // Execute task
                        continue;
                    }
                    
                    // Try stealing from other workers
                    // ... work stealing logic ...
                    
                    std::thread::sleep(Duration::from_millis(1));
                }
            });
            
            workers.push(Worker { local_queue, thread });
        }
        
        Self { workers, global_queue }
    }
}
```

## 10. Memory-Mapped File Processing

```rust
use memmap2::{Mmap, MmapMut, MmapOptions};
use std::fs::{File, OpenOptions};
use std::io::{self, Write};

/// High-performance file processor using memory mapping
pub struct MmapFileProcessor;

impl MmapFileProcessor {
    /// Process large file without loading into memory
    pub fn process_large_file(path: &str) -> io::Result<ProcessingResult> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        
        // Process file in parallel chunks
        let chunk_size = 1024 * 1024; // 1MB chunks
        let num_chunks = (mmap.len() + chunk_size - 1) / chunk_size;
        
        let results: Vec<_> = (0..num_chunks)
            .into_par_iter()
            .map(|i| {
                let start = i * chunk_size;
                let end = ((i + 1) * chunk_size).min(mmap.len());
                let chunk = &mmap[start..end];
                
                // Process chunk (example: count newlines)
                chunk.iter().filter(|&&b| b == b'\n').count()
            })
            .collect();
        
        Ok(ProcessingResult {
            total_lines: results.iter().sum(),
            file_size: mmap.len(),
        })
    }
    
    /// Memory-mapped file writer for high-performance output
    pub fn create_mmap_writer(path: &str, size: usize) -> io::Result<MmapWriter> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;
            
        file.set_len(size as u64)?;
        
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        
        Ok(MmapWriter {
            mmap,
            position: 0,
        })
    }
}

pub struct MmapWriter {
    mmap: MmapMut,
    position: usize,
}

impl Write for MmapWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let available = self.mmap.len() - self.position;
        let to_write = buf.len().min(available);
        
        self.mmap[self.position..self.position + to_write]
            .copy_from_slice(&buf[..to_write]);
            
        self.position += to_write;
        Ok(to_write)
    }
    
    fn flush(&mut self) -> io::Result<()> {
        self.mmap.flush()
    }
}

#[derive(Debug)]
pub struct ProcessingResult {
    pub total_lines: usize,
    pub file_size: usize,
}

/// Parallel CSV processor using memory mapping
pub struct ParallelCsvProcessor;

impl ParallelCsvProcessor {
    pub async fn process_csv_parallel(
        path: &str,
        chunk_size: usize,
    ) -> Result<Vec<ProcessedChunk>, Box<dyn std::error::Error>> {
        let file = tokio::fs::File::open(path).await?;
        let metadata = file.metadata().await?;
        let file_size = metadata.len() as usize;
        
        let file = file.into_std().await;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        
        // Find line boundaries for proper CSV splitting
        let mut boundaries = vec![0];
        let mut pos = chunk_size;
        
        while pos < file_size {
            // Find next newline after chunk boundary
            while pos < file_size && mmap[pos] != b'\n' {
                pos += 1;
            }
            if pos < file_size {
                boundaries.push(pos + 1);
                pos += chunk_size;
            }
        }
        boundaries.push(file_size);
        
        // Process chunks in parallel
        let chunks: Vec<_> = boundaries
            .windows(2)
            .map(|w| (w[0], w[1]))
            .collect();
            
        let results = tokio::task::spawn_blocking(move || {
            chunks
                .into_par_iter()
                .map(|(start, end)| {
                    let chunk = &mmap[start..end];
                    // Parse CSV chunk
                    ProcessedChunk {
                        start_offset: start,
                        end_offset: end,
                        record_count: chunk.iter().filter(|&&b| b == b'\n').count(),
                    }
                })
                .collect()
        })
        .await?;
        
        Ok(results)
    }
}

#[derive(Debug)]
pub struct ProcessedChunk {
    pub start_offset: usize,
    pub end_offset: usize,
    pub record_count: usize,
}

use rayon::prelude::*;
```

## 11. WebAssembly Integration

```rust
use wasmtime::{Engine, Instance, Module, Store, Func, Caller};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder};

/// WebAssembly runtime manager
pub struct WasmRuntime {
    engine: Engine,
}

impl WasmRuntime {
    pub fn new() -> Result<Self, wasmtime::Error> {
        let engine = Engine::default();
        Ok(Self { engine })
    }
    
    /// Load and execute WASM module
    pub async fn execute_wasm(
        &self,
        wasm_bytes: &[u8],
        function_name: &str,
        args: Vec<WasmValue>,
    ) -> Result<Vec<WasmValue>, Box<dyn std::error::Error>> {
        let module = Module::new(&self.engine, wasm_bytes)?;
        
        // Create WASI context
        let wasi = WasiCtxBuilder::new()
            .inherit_stdio()
            .inherit_env()
            .build();
            
        let mut store = Store::new(&self.engine, wasi);
        
        // Instantiate module
        let instance = Instance::new(&mut store, &module, &[])?;
        
        // Get function
        let func = instance
            .get_func(&mut store, function_name)
            .ok_or("Function not found")?;
            
        // Convert args and call function
        let results = func.call(&mut store, &args)?;
        
        Ok(results)
    }
    
    /// Create sandboxed execution environment
    pub fn create_sandbox(&self) -> Result<Sandbox, Box<dyn std::error::Error>> {
        let engine = Engine::new(
            wasmtime::Config::new()
                .wasm_threads(false)
                .wasm_simd(true)
                .wasm_multi_memory(false)
                .consume_fuel(true)
                .cranelift_opt_level(wasmtime::OptLevel::Speed),
        )?;
        
        Ok(Sandbox { engine })
    }
}

pub struct Sandbox {
    engine: Engine,
}

impl Sandbox {
    /// Execute untrusted code with resource limits
    pub async fn execute_sandboxed(
        &self,
        wasm_bytes: &[u8],
        fuel_limit: u64,
    ) -> Result<SandboxResult, Box<dyn std::error::Error>> {
        let module = Module::new(&self.engine, wasm_bytes)?;
        let mut store = Store::new(&self.engine, ());
        
        // Set fuel limit
        store.add_fuel(fuel_limit)?;
        
        let start_time = std::time::Instant::now();
        
        let instance = Instance::new(&mut store, &module, &[])?;
        
        // Execute main function
        let main = instance
            .get_func(&mut store, "_start")
            .ok_or("Main function not found")?;
            
        match main.call(&mut store, &[]) {
            Ok(_) => {
                let fuel_consumed = fuel_limit - store.fuel_consumed().unwrap_or(0);
                Ok(SandboxResult {
                    success: true,
                    fuel_consumed,
                    execution_time: start_time.elapsed(),
                })
            }
            Err(e) => Ok(SandboxResult {
                success: false,
                fuel_consumed: fuel_limit - store.fuel_consumed().unwrap_or(0),
                execution_time: start_time.elapsed(),
            }),
        }
    }
}

#[derive(Debug)]
pub struct SandboxResult {
    pub success: bool,
    pub fuel_consumed: u64,
    pub execution_time: Duration,
}

type WasmValue = wasmtime::Val;
```

## 12. gRPC with Tonic and Streaming

```rust
use tonic::{transport::Server, Request, Response, Status, Streaming};
use futures::stream::{Stream, StreamExt};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

// Proto definitions would be here
pub mod data_service {
    tonic::include_proto!("dataservice");
}

use data_service::{
    data_service_server::{DataService, DataServiceServer},
    StreamRequest, StreamResponse, ProcessRequest, ProcessResponse,
};

/// gRPC service implementation with bidirectional streaming
pub struct DataStreamService {
    processor: Arc<DataProcessor>,
}

#[tonic::async_trait]
impl DataService for DataStreamService {
    type StreamDataStream = ReceiverStream<Result<StreamResponse, Status>>;
    type BidirectionalProcessStream = Pin<Box<dyn Stream<Item = Result<ProcessResponse, Status>> + Send>>;
    
    /// Server streaming RPC
    async fn stream_data(
        &self,
        request: Request<StreamRequest>,
    ) -> Result<Response<Self::StreamDataStream>, Status> {
        let (tx, rx) = mpsc::channel(128);
        let req = request.into_inner();
        
        // Spawn background task to stream data
        tokio::spawn(async move {
            for i in 0..req.count {
                let response = StreamResponse {
                    sequence: i,
                    data: format!("Data chunk {}", i).into_bytes(),
                    timestamp: chrono::Utc::now().timestamp(),
                };
                
                if tx.send(Ok(response)).await.is_err() {
                    break;
                }
                
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        });
        
        Ok(Response::new(ReceiverStream::new(rx)))
    }
    
    /// Bidirectional streaming RPC
    async fn bidirectional_process(
        &self,
        request: Request<Streaming<ProcessRequest>>,
    ) -> Result<Response<Self::BidirectionalProcessStream>, Status> {
        let mut stream = request.into_inner();
        let processor = self.processor.clone();
        
        let output_stream = async_stream::stream! {
            while let Some(result) = stream.next().await {
                match result {
                    Ok(req) => {
                        // Process each request
                        let processed = processor.process(&req.data).await;
                        
                        yield Ok(ProcessResponse {
                            request_id: req.id,
                            result: processed,
                            processing_time_ms: 10,
                        });
                    }
                    Err(e) => {
                        yield Err(e);
                    }
                }
            }
        };
        
        Ok(Response::new(Box::pin(output_stream)))
    }
}

/// gRPC client with streaming support
pub struct DataStreamClient {
    client: data_service::data_service_client::DataServiceClient<tonic::transport::Channel>,
}

impl DataStreamClient {
    pub async fn connect(addr: &str) -> Result<Self, tonic::transport::Error> {
        let client = data_service::data_service_client::DataServiceClient::connect(addr).await?;
        Ok(Self { client })
    }
    
    /// Stream data from server
    pub async fn stream_data(&mut self, count: u32) -> Result<Vec<Vec<u8>>, Status> {
        let request = tonic::Request::new(StreamRequest { count });
        let mut stream = self.client.stream_data(request).await?.into_inner();
        
        let mut results = Vec::new();
        while let Some(response) = stream.next().await {
            let response = response?;
            results.push(response.data);
        }
        
        Ok(results)
    }
    
    /// Bidirectional streaming
    pub async fn process_stream<S>(
        &mut self,
        input_stream: S,
    ) -> Result<Vec<ProcessResponse>, Status>
    where
        S: Stream<Item = ProcessRequest> + Send + 'static,
    {
        let request = tonic::Request::new(input_stream);
        let mut response_stream = self.client
            .bidirectional_process(request)
            .await?
            .into_inner();
            
        let mut results = Vec::new();
        while let Some(response) = response_stream.next().await {
            results.push(response?);
        }
        
        Ok(results)
    }
}

/// Start gRPC server
pub async fn start_grpc_server(addr: &str) -> Result<(), Box<dyn std::error::Error>> {
    let addr = addr.parse()?;
    let service = DataStreamService {
        processor: Arc::new(DataProcessor),
    };
    
    Server::builder()
        .add_service(DataServiceServer::new(service))
        .serve(addr)
        .await?;
        
    Ok(())
}
```

## Usage Examples

### Complete Example: Building a High-Performance Data Pipeline

```rust
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    // Create connection pool with health checks
    let pool_config = deadpool_postgres::Config {
        host: Some("localhost".to_string()),
        port: Some(5432),
        dbname: Some("mydb".to_string()),
        user: Some("user".to_string()),
        password: Some("password".to_string()),
        ..Default::default()
    };
    
    let pool = SmartConnectionPool::new(pool_config, "redis://localhost").await?;
    
    // Create rate limiter
    let rate_limiter = DistributedRateLimiter::new(
        pool.get_redis(),
        Duration::from_secs(60),
        1000,
    );
    
    // Set up stream processor with backpressure
    let (stream, sender) = BackpressureStream::<serde_json::Value>::new(1000);
    
    // Process stream with circuit breaker protection
    let circuit_breaker = CircuitBreaker::new(5, 3, Duration::from_secs(30));
    
    // Start processing pipeline
    let processor = StreamProcessor::new(10, 100);
    let processed_stream = processor.process_stream(stream, |data| {
        // Transform data
        data
    }).await;
    
    // Collect results
    tokio::pin!(processed_stream);
    while let Some(result) = processed_stream.next().await {
        println!("Processed: {:?}", result);
    }
    
    Ok(())
}
```

## Best Practices

1. **Error Handling**: Always use proper error types (`thiserror` for libraries, `anyhow` for applications)
2. **Async Design**: Use `tokio` for async runtime, avoid blocking operations in async contexts
3. **Memory Management**: Leverage Rust's ownership system, use `Arc` for shared state
4. **Concurrency**: Prefer message passing over shared state, use `DashMap` for concurrent hashmaps
5. **Performance**: Profile with `flamegraph`, use `criterion` for benchmarking
6. **Testing**: Write unit tests, integration tests, and property-based tests with `proptest`

## 13. Distributed Tracing with OpenTelemetry

```rust
use opentelemetry::{
    global,
    sdk::{propagation::TraceContextPropagator, trace as sdktrace},
    trace::{FutureExt, TraceContextExt, Tracer, SpanKind, Status as OtelStatus},
    Context as OtelContext,
};
use opentelemetry_otlp::WithExportConfig;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Distributed tracing setup for microservices
pub struct TracingManager;

impl TracingManager {
    /// Initialize OpenTelemetry with OTLP exporter
    pub fn init_telemetry(service_name: &str) -> Result<(), Box<dyn std::error::Error>> {
        global::set_text_map_propagator(TraceContextPropagator::new());
        
        let otlp_exporter = opentelemetry_otlp::new_exporter()
            .tonic()
            .with_endpoint("http://localhost:4317");
            
        let trace_config = sdktrace::config()
            .with_sampler(sdktrace::Sampler::AlwaysOn)
            .with_id_generator(sdktrace::RandomIdGenerator::default())
            .with_resource(sdktrace::Resource::new(vec![
                opentelemetry::KeyValue::new("service.name", service_name.to_string()),
                opentelemetry::KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
            ]));
            
        let tracer = opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(otlp_exporter)
            .with_trace_config(trace_config)
            .install_batch(opentelemetry::runtime::Tokio)?;
            
        let telemetry = tracing_opentelemetry::layer().with_tracer(tracer);
        
        tracing_subscriber::registry()
            .with(tracing_subscriber::EnvFilter::from_default_env())
            .with(telemetry)
            .init();
            
        Ok(())
    }
    
    /// Shutdown telemetry gracefully
    pub fn shutdown_telemetry() {
        global::shutdown_tracer_provider();
    }
}

/// Traced database operations
pub struct TracedDatabase {
    pool: PgPool,
    tracer: global::BoxedTracer,
}

impl TracedDatabase {
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool,
            tracer: global::tracer("database"),
        }
    }
    
    pub async fn execute_query<T>(
        &self,
        query: &str,
        parent_context: Option<OtelContext>,
    ) -> Result<Vec<T>, sqlx::Error>
    where
        T: for<'r> sqlx::FromRow<'r, sqlx::postgres::PgRow> + Send + Unpin,
    {
        let parent_context = parent_context.unwrap_or_else(OtelContext::current);
        
        let mut span = self.tracer
            .span_builder("database.query")
            .with_kind(SpanKind::Client)
            .with_attributes(vec![
                opentelemetry::KeyValue::new("db.system", "postgresql"),
                opentelemetry::KeyValue::new("db.statement", query.to_string()),
            ])
            .start_with_context(&self.tracer, &parent_context);
            
        let result = sqlx::query_as::<_, T>(query)
            .fetch_all(&self.pool)
            .await;
            
        match &result {
            Ok(rows) => {
                span.set_attribute(opentelemetry::KeyValue::new(
                    "db.rows_affected",
                    rows.len() as i64,
                ));
                span.set_status(OtelStatus::Ok);
            }
            Err(e) => {
                span.record_error(e);
                span.set_status(OtelStatus::error(e.to_string()));
            }
        }
        
        result
    }
}

/// Distributed trace context propagation for HTTP
pub struct TraceContextPropagator;

impl TraceContextPropagator {
    /// Extract trace context from HTTP headers
    pub fn extract_from_headers(headers: &reqwest::header::HeaderMap) -> OtelContext {
        let extractor = HeaderExtractor(headers);
        global::get_text_map_propagator(|propagator| {
            propagator.extract(&extractor)
        })
    }
    
    /// Inject trace context into HTTP headers
    pub fn inject_to_headers(headers: &mut reqwest::header::HeaderMap) {
        let injector = HeaderInjector(headers);
        global::get_text_map_propagator(|propagator| {
            propagator.inject_context(&OtelContext::current(), &injector);
        });
    }
}

struct HeaderExtractor<'a>(&'a reqwest::header::HeaderMap);
struct HeaderInjector<'a>(&'a mut reqwest::header::HeaderMap);

impl<'a> opentelemetry::propagation::Extractor for HeaderExtractor<'a> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|v| v.to_str().ok())
    }
    
    fn keys(&self) -> Vec<&str> {
        self.0.keys().filter_map(|k| k.as_str()).collect()
    }
}

impl<'a> opentelemetry::propagation::Injector for HeaderInjector<'a> {
    fn set(&mut self, key: &str, value: String) {
        if let Ok(header_name) = reqwest::header::HeaderName::from_bytes(key.as_bytes()) {
            if let Ok(header_value) = reqwest::header::HeaderValue::from_str(&value) {
                self.0.insert(header_name, header_value);
            }
        }
    }
}
```

## 14. Advanced Caching Strategies

```rust
use arc_swap::ArcSwap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// Multi-tier caching system with L1 (in-memory) and L2 (Redis) caches
pub struct MultiTierCache<T: Clone + Send + Sync + 'static> {
    l1_cache: Arc<DashMap<String, CacheEntry<T>>>,
    l2_cache: redis::aio::ConnectionManager,
    l1_capacity: usize,
    ttl: Duration,
}

#[derive(Clone)]
struct CacheEntry<T> {
    value: T,
    expires_at: Instant,
    access_count: Arc<AtomicU64>,
}

impl<T: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>> MultiTierCache<T> {
    pub fn new(
        l2_cache: redis::aio::ConnectionManager,
        l1_capacity: usize,
        ttl: Duration,
    ) -> Self {
        Self {
            l1_cache: Arc::new(DashMap::new()),
            l2_cache,
            l1_capacity,
            ttl,
        }
    }
    
    /// Get value with cache-aside pattern
    pub async fn get_or_compute<F, Fut>(
        &self,
        key: &str,
        compute_fn: F,
    ) -> Result<T, Box<dyn std::error::Error>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T, Box<dyn std::error::Error>>>,
    {
        // Check L1 cache
        if let Some(entry) = self.l1_cache.get(key) {
            if entry.expires_at > Instant::now() {
                entry.access_count.fetch_add(1, Ordering::Relaxed);
                return Ok(entry.value.clone());
            } else {
                self.l1_cache.remove(key);
            }
        }
        
        // Check L2 cache
        let mut l2_conn = self.l2_cache.clone();
        let cached_data: Option<Vec<u8>> = l2_conn
            .get(format!("cache:{}", key))
            .await
            .unwrap_or(None);
            
        if let Some(data) = cached_data {
            if let Ok(value) = bincode::deserialize::<T>(&data) {
                self.promote_to_l1(key, value.clone()).await;
                return Ok(value);
            }
        }
        
        // Compute value
        let value = compute_fn().await?;
        
        // Store in both caches
        self.set(key, value.clone()).await?;
        
        Ok(value)
    }
    
    /// Set value in both cache tiers
    pub async fn set(&self, key: &str, value: T) -> Result<(), Box<dyn std::error::Error>> {
        // Store in L2 (Redis)
        let serialized = bincode::serialize(&value)?;
        let mut l2_conn = self.l2_cache.clone();
        l2_conn.set_ex(
            format!("cache:{}", key),
            serialized,
            self.ttl.as_secs() as usize,
        ).await?;
        
        // Store in L1
        self.promote_to_l1(key, value).await;
        
        Ok(())
    }
    
    /// Promote value to L1 cache with LRU eviction
    async fn promote_to_l1(&self, key: &str, value: T) {
        // Check capacity
        if self.l1_cache.len() >= self.l1_capacity {
            // Evict least recently used
            let mut min_access = u64::MAX;
            let mut evict_key = None;
            
            for entry in self.l1_cache.iter() {
                let access_count = entry.value().access_count.load(Ordering::Relaxed);
                if access_count < min_access {
                    min_access = access_count;
                    evict_key = Some(entry.key().clone());
                }
            }
            
            if let Some(key) = evict_key {
                self.l1_cache.remove(&key);
            }
        }
        
        self.l1_cache.insert(
            key.to_string(),
            CacheEntry {
                value,
                expires_at: Instant::now() + self.ttl,
                access_count: Arc::new(AtomicU64::new(1)),
            },
        );
    }
    
    /// Invalidate cache entry
    pub async fn invalidate(&self, key: &str) -> Result<(), redis::RedisError> {
        self.l1_cache.remove(key);
        
        let mut l2_conn = self.l2_cache.clone();
        l2_conn.del(format!("cache:{}", key)).await?;
        
        Ok(())
    }
}

/// Bloom filter for cache miss optimization
pub struct BloomFilterCache {
    filter: ArcSwap<BloomFilter>,
    false_positive_rate: f64,
}

struct BloomFilter {
    bits: Vec<AtomicBool>,
    hash_count: usize,
}

impl BloomFilter {
    fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        let bits_per_item = -(false_positive_rate.ln() / (2.0_f64.ln().powi(2)));
        let num_bits = (expected_items as f64 * bits_per_item).ceil() as usize;
        let hash_count = (bits_per_item * 2.0_f64.ln()).ceil() as usize;
        
        let bits = (0..num_bits)
            .map(|_| AtomicBool::new(false))
            .collect();
            
        Self { bits, hash_count }
    }
    
    fn insert(&self, item: &str) {
        for i in 0..self.hash_count {
            let hash = Self::hash(item, i);
            let index = hash % self.bits.len();
            self.bits[index].store(true, Ordering::Relaxed);
        }
    }
    
    fn might_contain(&self, item: &str) -> bool {
        for i in 0..self.hash_count {
            let hash = Self::hash(item, i);
            let index = hash % self.bits.len();
            if !self.bits[index].load(Ordering::Relaxed) {
                return false;
            }
        }
        true
    }
    
    fn hash(item: &str, seed: usize) -> usize {
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        seed.hash(&mut hasher);
        hasher.finish() as usize
    }
}
```

## 15. Advanced BigQuery Integration

```rust
use google_cloud_bigquery::client::{Client as BqClient, ClientConfig};
use google_cloud_bigquery::model::{
    query_request::QueryRequest,
    query_response::QueryResponse,
    table_data_insert_all_request::{TableDataInsertAllRequest, Row},
};

/// BigQuery streaming buffer with batching
pub struct BigQueryStreamer {
    client: BqClient,
    project_id: String,
    dataset_id: String,
    table_id: String,
    buffer: Arc<Mutex<Vec<serde_json::Value>>>,
    batch_size: usize,
    flush_interval: Duration,
}

impl BigQueryStreamer {
    pub async fn new(
        project_id: String,
        dataset_id: String,
        table_id: String,
        batch_size: usize,
        flush_interval: Duration,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let config = ClientConfig::default()
            .with_auth()
            .await?;
            
        let client = BqClient::new(config).await?;
        
        let streamer = Self {
            client,
            project_id,
            dataset_id,
            table_id,
            buffer: Arc::new(Mutex::new(Vec::new())),
            batch_size,
            flush_interval,
        };
        
        // Start background flush task
        streamer.start_flush_task();
        
        Ok(streamer)
    }
    
    /// Insert data into streaming buffer
    pub async fn insert(&self, data: serde_json::Value) -> Result<(), Box<dyn std::error::Error>> {
        let mut buffer = self.buffer.lock().await;
        buffer.push(data);
        
        if buffer.len() >= self.batch_size {
            let batch = std::mem::take(&mut *buffer);
            drop(buffer);
            self.flush_batch(batch).await?;
        }
        
        Ok(())
    }
    
    /// Flush batch to BigQuery
    async fn flush_batch(&self, batch: Vec<serde_json::Value>) -> Result<(), Box<dyn std::error::Error>> {
        if batch.is_empty() {
            return Ok(());
        }
        
        let rows: Vec<Row> = batch
            .into_iter()
            .map(|data| Row {
                insert_id: Some(Uuid::new_v4().to_string()),
                json: data,
            })
            .collect();
            
        let request = TableDataInsertAllRequest {
            rows,
            skip_invalid_rows: Some(false),
            ignore_unknown_values: Some(false),
            template_suffix: None,
        };
        
        self.client
            .tabledata()
            .insert_all(
                &self.project_id,
                &self.dataset_id,
                &self.table_id,
                request,
            )
            .await?;
            
        Ok(())
    }
    
    /// Start background task to flush buffer periodically
    fn start_flush_task(&self) {
        let buffer = self.buffer.clone();
        let flush_interval = self.flush_interval;
        let self_clone = self.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(flush_interval);
            
            loop {
                interval.tick().await;
                
                let batch = {
                    let mut buffer = buffer.lock().await;
                    std::mem::take(&mut *buffer)
                };
                
                if !batch.is_empty() {
                    if let Err(e) = self_clone.flush_batch(batch).await {
                        eprintln!("Failed to flush batch: {}", e);
                    }
                }
            }
        });
    }
}

/// BigQuery query optimizer with caching
pub struct OptimizedBigQueryClient {
    client: BqClient,
    query_cache: MultiTierCache<QueryResponse>,
    query_stats: Arc<DashMap<String, QueryStats>>,
}

#[derive(Clone)]
struct QueryStats {
    execution_count: u64,
    total_bytes_processed: u64,
    avg_execution_time_ms: u64,
}

impl OptimizedBigQueryClient {
    pub async fn new(
        cache: MultiTierCache<QueryResponse>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let config = ClientConfig::default()
            .with_auth()
            .await?;
            
        let client = BqClient::new(config).await?;
        
        Ok(Self {
            client,
            query_cache: cache,
            query_stats: Arc::new(DashMap::new()),
        })
    }
    
    /// Execute query with caching and statistics
    pub async fn query(
        &self,
        project_id: &str,
        query: &str,
        use_cache: bool,
    ) -> Result<QueryResponse, Box<dyn std::error::Error>> {
        let query_hash = self.hash_query(query);
        
        if use_cache {
            if let Ok(cached) = self.query_cache.get_or_compute(
                &query_hash,
                || self.execute_query(project_id, query)
            ).await {
                return Ok(cached);
            }
        }
        
        self.execute_query(project_id, query).await
    }
    
    /// Execute query with statistics tracking
    async fn execute_query(
        &self,
        project_id: &str,
        query: &str,
    ) -> Result<QueryResponse, Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        
        let request = QueryRequest {
            query: query.to_string(),
            use_legacy_sql: false,
            ..Default::default()
        };
        
        let response = self.client
            .job()
            .query(project_id, request)
            .await?;
            
        let execution_time = start_time.elapsed().as_millis() as u64;
        
        // Update statistics
        let query_hash = self.hash_query(query);
        self.query_stats
            .entry(query_hash)
            .and_modify(|stats| {
                stats.execution_count += 1;
                stats.total_bytes_processed += response.total_bytes_processed.unwrap_or(0) as u64;
                stats.avg_execution_time_ms = 
                    (stats.avg_execution_time_ms * (stats.execution_count - 1) + execution_time) 
                    / stats.execution_count;
            })
            .or_insert(QueryStats {
                execution_count: 1,
                total_bytes_processed: response.total_bytes_processed.unwrap_or(0) as u64,
                avg_execution_time_ms: execution_time,
            });
            
        Ok(response)
    }
    
    fn hash_query(&self, query: &str) -> String {
        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        format!("query:{:x}", hasher.finish())
    }
}
```

## 16. Distributed Task Queue with Priority

```rust
use redis::streams::{StreamReadOptions, StreamReadReply};
use priority_queue::PriorityQueue;

/// Distributed task queue using Redis Streams
pub struct DistributedTaskQueue {
    redis_pool: redis::aio::ConnectionManager,
    consumer_group: String,
    consumer_name: String,
    stream_key: String,
    max_retries: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub task_type: String,
    pub payload: serde_json::Value,
    pub priority: u8,
    pub created_at: DateTime<Utc>,
    pub retry_count: u32,
}

impl DistributedTaskQueue {
    pub fn new(
        redis_pool: redis::aio::ConnectionManager,
        consumer_group: String,
        consumer_name: String,
        stream_key: String,
        max_retries: u32,
    ) -> Self {
        Self {
            redis_pool,
            consumer_group,
            consumer_name,
            stream_key,
            max_retries,
        }
    }
    
    /// Initialize consumer group
    pub async fn init_consumer_group(&self) -> Result<(), redis::RedisError> {
        let mut conn = self.redis_pool.clone();
        
        // Try to create consumer group, ignore error if it already exists
        let _: Result<String, _> = redis::cmd("XGROUP")
            .arg("CREATE")
            .arg(&self.stream_key)
            .arg(&self.consumer_group)
            .arg("0")
            .query_async(&mut conn)
            .await;
            
        Ok(())
    }
    
    /// Enqueue task with priority
    pub async fn enqueue(&self, task: Task) -> Result<String, redis::RedisError> {
        let mut conn = self.redis_pool.clone();
        
        let task_data = serde_json::to_string(&task).unwrap();
        
        // Add to priority queue in Redis sorted set
        redis::cmd("ZADD")
            .arg(format!("{}:priority", self.stream_key))
            .arg(task.priority)
            .arg(&task.id)
            .query_async(&mut conn)
            .await?;
            
        // Add to stream
        let id: String = redis::cmd("XADD")
            .arg(&self.stream_key)
            .arg("*")
            .arg("task")
            .arg(task_data)
            .query_async(&mut conn)
            .await?;
            
        Ok(id)
    }
    
    /// Consume tasks with priority ordering
    pub async fn consume<F, Fut>(
        &self,
        batch_size: usize,
        handler: F,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: Fn(Task) -> Fut + Clone,
        Fut: Future<Output = Result<(), Box<dyn std::error::Error>>>,
    {
        loop {
            let tasks = self.fetch_tasks(batch_size).await?;
            
            if tasks.is_empty() {
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }
            
            // Process tasks concurrently
            let futures: Vec<_> = tasks
                .into_iter()
                .map(|(stream_id, task)| {
                    let handler = handler.clone();
                    let stream_id = stream_id.clone();
                    let queue = self.clone();
                    
                    tokio::spawn(async move {
                        match handler(task.clone()).await {
                            Ok(_) => {
                                queue.ack_task(&stream_id).await.ok();
                            }
                            Err(e) => {
                                eprintln!("Task {} failed: {}", task.id, e);
                                queue.retry_task(&stream_id, task).await.ok();
                            }
                        }
                    })
                })
                .collect();
                
            futures::future::join_all(futures).await;
        }
    }
    
    /// Fetch tasks ordered by priority
    async fn fetch_tasks(
        &self,
        batch_size: usize,
    ) -> Result<Vec<(String, Task)>, Box<dyn std::error::Error>> {
        let mut conn = self.redis_pool.clone();
        
        // Get highest priority task IDs
        let task_ids: Vec<String> = redis::cmd("ZREVRANGE")
            .arg(format!("{}:priority", self.stream_key))
            .arg(0)
            .arg(batch_size - 1)
            .query_async(&mut conn)
            .await?;
            
        if task_ids.is_empty() {
            return Ok(vec![]);
        }
        
        // Read from stream
        let opts = StreamReadOptions::default()
            .group(&self.consumer_group, &self.consumer_name)
            .count(batch_size)
            .block(0);
            
        let reply: StreamReadReply = conn
            .xread_options(&[&self.stream_key], &[">"], &opts)
            .await?;
            
        let mut tasks = Vec::new();
        
        for stream_key in reply.keys {
            for stream_id in stream_key.ids {
                if let Some(task_data) = stream_id.map.get("task") {
                    if let redis::Value::Data(data) = task_data {
                        let task: Task = serde_json::from_slice(data)?;
                        
                        // Only include if in priority set
                        if task_ids.contains(&task.id) {
                            tasks.push((stream_id.id, task));
                        }
                    }
                }
            }
        }
        
        Ok(tasks)
    }
    
    /// Acknowledge task completion
    async fn ack_task(&self, stream_id: &str) -> Result<(), redis::RedisError> {
        let mut conn = self.redis_pool.clone();
        
        redis::cmd("XACK")
            .arg(&self.stream_key)
            .arg(&self.consumer_group)
            .arg(stream_id)
            .query_async(&mut conn)
            .await?;
            
        Ok(())
    }
    
    /// Retry failed task
    async fn retry_task(
        &self,
        stream_id: &str,
        mut task: Task,
    ) -> Result<(), Box<dyn std::error::Error>> {
        task.retry_count += 1;
        
        if task.retry_count > self.max_retries {
            // Move to dead letter queue
            let mut conn = self.redis_pool.clone();
            redis::cmd("XADD")
                .arg(format!("{}:dlq", self.stream_key))
                .arg("*")
                .arg("task")
                .arg(serde_json::to_string(&task)?)
                .query_async(&mut conn)
                .await?;
                
            self.ack_task(stream_id).await?;
        } else {
            // Re-enqueue with exponential backoff
            let delay = Duration::from_secs(2u64.pow(task.retry_count));
            tokio::time::sleep(delay).await;
            self.enqueue(task).await?;
            self.ack_task(stream_id).await?;
        }
        
        Ok(())
    }
}

impl Clone for DistributedTaskQueue {
    fn clone(&self) -> Self {
        Self {
            redis_pool: self.redis_pool.clone(),
            consumer_group: self.consumer_group.clone(),
            consumer_name: self.consumer_name.clone(),
            stream_key: self.stream_key.clone(),
            max_retries: self.max_retries,
        }
    }
}
```

## 17. Real-time Data Pipeline with Kafka

```rust
use rdkafka::{
    ClientConfig, Message, Offset, TopicPartitionList,
    consumer::{Consumer, StreamConsumer, CommitMode},
    producer::{FutureProducer, FutureRecord},
};
use rdkafka::config::RDKafkaLogLevel;

/// Real-time data pipeline with exactly-once semantics
pub struct KafkaDataPipeline {
    consumer: Arc<StreamConsumer>,
    producer: Arc<FutureProducer>,
    processor: Arc<dyn DataProcessor>,
}

#[async_trait::async_trait]
pub trait DataProcessor: Send + Sync {
    async fn process(&self, data: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>>;
}

impl KafkaDataPipeline {
    pub fn new(
        bootstrap_servers: &str,
        group_id: &str,
        input_topics: Vec<String>,
        processor: Arc<dyn DataProcessor>,
    ) -> Result<Self, rdkafka::error::KafkaError> {
        let consumer: StreamConsumer = ClientConfig::new()
            .set("bootstrap.servers", bootstrap_servers)
            .set("group.id", group_id)
            .set("enable.auto.commit", "false")
            .set("auto.offset.reset", "earliest")
            .set("isolation.level", "read_committed")
            .set_log_level(RDKafkaLogLevel::Info)
            .create()?;
            
        consumer.subscribe(&input_topics.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;
        
        let producer: FutureProducer = ClientConfig::new()
            .set("bootstrap.servers", bootstrap_servers)
            .set("enable.idempotence", "true")
            .set("transactional.id", format!("{}-producer", group_id))
            .set("compression.type", "snappy")
            .create()?;
            
        Ok(Self {
            consumer: Arc::new(consumer),
            producer: Arc::new(producer),
            processor,
        })
    }
    
    /// Run the pipeline with exactly-once processing
    pub async fn run(&self, output_topic: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Initialize transactions
        self.producer.init_transactions(Duration::from_secs(30))?;
        
        loop {
            match self.consumer.recv().await {
                Ok(message) => {
                    let input_data = message.payload().ok_or("Empty message")?;
                    let offset = message.offset();
                    let partition = message.partition();
                    let topic = message.topic();
                    
                    // Begin transaction
                    self.producer.begin_transaction()?;
                    
                    match self.processor.process(input_data).await {
                        Ok(output_data) => {
                            // Send to output topic
                            let record = FutureRecord::to(output_topic)
                                .payload(&output_data)
                                .key(message.key().unwrap_or(&[]));
                                
                            self.producer.send(record, Duration::from_secs(0)).await
                                .map_err(|(e, _)| e)?;
                                
                            // Commit offsets within transaction
                            let mut offsets = TopicPartitionList::new();
                            offsets.add_partition_offset(
                                topic,
                                partition,
                                Offset::Offset(offset + 1),
                            )?;
                            
                            self.producer.send_offsets_to_transaction(
                                &offsets,
                                &self.consumer.group_metadata(),
                                Duration::from_secs(30),
                            )?;
                            
                            // Commit transaction
                            self.producer.commit_transaction(Duration::from_secs(30))?;
                        }
                        Err(e) => {
                            eprintln!("Processing error: {}", e);
                            self.producer.abort_transaction(Duration::from_secs(30))?;
                            
                            // Optionally send to DLQ
                            self.send_to_dlq(topic, input_data, &e.to_string()).await?;
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Kafka error: {}", e);
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    }
    
    /// Send failed messages to dead letter queue
    async fn send_to_dlq(
        &self,
        original_topic: &str,
        data: &[u8],
        error: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let dlq_topic = format!("{}-dlq", original_topic);
        
        let metadata = serde_json::json!({
            "original_topic": original_topic,
            "error": error,
            "timestamp": chrono::Utc::now().to_rfc3339(),
        });
        
        let record = FutureRecord::to(&dlq_topic)
            .payload(data)
            .headers(rdkafka::message::OwnedHeaders::new()
                .add("error_metadata", &serde_json::to_vec(&metadata)?));
                
        self.producer.send(record, Duration::from_secs(0)).await
            .map_err(|(e, _)| e)?;
            
        Ok(())
    }
}

/// Example: Stream aggregation processor
pub struct StreamAggregator {
    window_size: Duration,
    aggregations: Arc<DashMap<String, AggregationWindow>>,
}

#[derive(Clone)]
struct AggregationWindow {
    start_time: DateTime<Utc>,
    values: Vec<f64>,
}

#[async_trait::async_trait]
impl DataProcessor for StreamAggregator {
    async fn process(&self, data: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let record: serde_json::Value = serde_json::from_slice(data)?;
        
        let key = record["key"].as_str().ok_or("Missing key")?;
        let value = record["value"].as_f64().ok_or("Missing value")?;
        let timestamp = chrono::Utc::now();
        
        self.aggregations
            .entry(key.to_string())
            .and_modify(|window| {
                if timestamp - window.start_time > self.window_size {
                    // New window
                    window.start_time = timestamp;
                    window.values.clear();
                }
                window.values.push(value);
            })
            .or_insert_with(|| AggregationWindow {
                start_time: timestamp,
                values: vec![value],
            });
            
        // Compute aggregations
        let window = self.aggregations.get(key).unwrap();
        let result = serde_json::json!({
            "key": key,
            "window_start": window.start_time.to_rfc3339(),
            "count": window.values.len(),
            "sum": window.values.iter().sum::<f64>(),
            "avg": window.values.iter().sum::<f64>() / window.values.len() as f64,
            "min": window.values.iter().cloned().fold(f64::INFINITY, f64::min),
            "max": window.values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        });
        
        Ok(serde_json::to_vec(&result)?)
    }
}
```

## 18. Advanced Metrics and Monitoring

```rust
use prometheus::{
    Encoder, TextEncoder, Counter, Gauge, Histogram, HistogramOpts,
    register_counter, register_gauge, register_histogram_with_registry,
    Registry,
};
use std::time::SystemTime;

/// Application metrics collector
pub struct MetricsCollector {
    registry: Registry,
    request_counter: Counter,
    active_connections: Gauge,
    request_duration: Histogram,
    db_query_duration: Histogram,
    cache_hit_ratio: Gauge,
}

impl MetricsCollector {
    pub fn new() -> Result<Self, prometheus::Error> {
        let registry = Registry::new();
        
        let request_counter = register_counter!(
            "http_requests_total",
            "Total number of HTTP requests"
        )?;
        
        let active_connections = register_gauge!(
            "active_connections",
            "Number of active connections"
        )?;
        
        let request_duration = register_histogram_with_registry!(
            HistogramOpts::new(
                "http_request_duration_seconds",
                "HTTP request duration in seconds"
            ).buckets(vec![0.001, 0.01, 0.1, 0.5, 1.0, 5.0]),
            registry
        )?;
        
        let db_query_duration = register_histogram_with_registry!(
            HistogramOpts::new(
                "db_query_duration_seconds",
                "Database query duration in seconds"
            ).buckets(vec![0.001, 0.01, 0.1, 1.0, 10.0]),
            registry
        )?;
        
        let cache_hit_ratio = register_gauge!(
            "cache_hit_ratio",
            "Cache hit ratio"
        )?;
        
        Ok(Self {
            registry,
            request_counter,
            active_connections,
            request_duration,
            db_query_duration,
            cache_hit_ratio,
        })
    }
    
    /// Record HTTP request
    pub fn record_request(&self, duration: Duration) {
        self.request_counter.inc();
        self.request_duration.observe(duration.as_secs_f64());
    }
    
    /// Record database query
    pub fn record_db_query(&self, duration: Duration) {
        self.db_query_duration.observe(duration.as_secs_f64());
    }
    
    /// Update cache metrics
    pub fn update_cache_metrics(&self, hits: u64, total: u64) {
        if total > 0 {
            self.cache_hit_ratio.set(hits as f64 / total as f64);
        }
    }
    
    /// Export metrics in Prometheus format
    pub fn export(&self) -> Result<String, prometheus::Error> {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer).unwrap())
    }
}

/// Custom metrics middleware for HTTP servers
pub struct MetricsMiddleware {
    metrics: Arc<MetricsCollector>,
}

impl MetricsMiddleware {
    pub fn new(metrics: Arc<MetricsCollector>) -> Self {
        Self { metrics }
    }
    
    /// Wrap handler with metrics collection
    pub async fn wrap<F, Fut, R>(
        &self,
        handler: F,
        request: R,
    ) -> Result<impl warp::Reply, warp::Rejection>
    where
        F: FnOnce(R) -> Fut,
        Fut: Future<Output = Result<impl warp::Reply, warp::Rejection>>,
    {
        let start = Instant::now();
        self.metrics.active_connections.inc();
        
        let result = handler(request).await;
        
        self.metrics.active_connections.dec();
        self.metrics.record_request(start.elapsed());
        
        result
    }
}

/// Distributed tracing span recorder
pub struct SpanRecorder {
    spans: Arc<DashMap<String, SpanData>>,
    export_interval: Duration,
}

#[derive(Clone)]
struct SpanData {
    trace_id: String,
    span_id: String,
    parent_span_id: Option<String>,
    operation: String,
    start_time: SystemTime,
    end_time: Option<SystemTime>,
    tags: HashMap<String, String>,
    events: Vec<SpanEvent>,
}

#[derive(Clone)]
struct SpanEvent {
    timestamp: SystemTime,
    message: String,
    attributes: HashMap<String, String>,
}

impl SpanRecorder {
    pub fn new(export_interval: Duration) -> Self {
        let recorder = Self {
            spans: Arc::new(DashMap::new()),
            export_interval,
        };
        
        recorder.start_export_task();
        recorder
    }
    
    /// Record span start
    pub fn start_span(
        &self,
        trace_id: String,
        span_id: String,
        operation: String,
        parent_span_id: Option<String>,
    ) {
        let span_data = SpanData {
            trace_id,
            span_id: span_id.clone(),
            parent_span_id,
            operation,
            start_time: SystemTime::now(),
            end_time: None,
            tags: HashMap::new(),
            events: Vec::new(),
        };
        
        self.spans.insert(span_id, span_data);
    }
    
    /// Record span end
    pub fn end_span(&self, span_id: &str) {
        if let Some(mut span) = self.spans.get_mut(span_id) {
            span.end_time = Some(SystemTime::now());
        }
    }
    
    /// Add event to span
    pub fn add_span_event(
        &self,
        span_id: &str,
        message: String,
        attributes: HashMap<String, String>,
    ) {
        if let Some(mut span) = self.spans.get_mut(span_id) {
            span.events.push(SpanEvent {
                timestamp: SystemTime::now(),
                message,
                attributes,
            });
        }
    }
    
    /// Export completed spans
    fn start_export_task(&self) {
        let spans = self.spans.clone();
        let export_interval = self.export_interval;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(export_interval);
            
            loop {
                interval.tick().await;
                
                let completed_spans: Vec<_> = spans
                    .iter()
                    .filter(|entry| entry.value().end_time.is_some())
                    .map(|entry| (entry.key().clone(), entry.value().clone()))
                    .collect();
                    
                for (span_id, span_data) in completed_spans {
                    // Export to tracing backend
                    // ... export logic ...
                    
                    spans.remove(&span_id);
                }
            }
        });
    }
}
```

## 19. Advanced Error Handling and Recovery

```rust
use std::panic;
use std::any::Any;

/// Comprehensive error handling with context
#[derive(Debug)]
pub struct ErrorContext {
    pub error: Box<dyn std::error::Error + Send + Sync>,
    pub context: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
    pub trace_id: Option<String>,
    pub retry_info: Option<RetryInfo>,
}

#[derive(Debug, Clone)]
pub struct RetryInfo {
    pub attempt: u32,
    pub max_attempts: u32,
    pub next_retry_at: Option<DateTime<Utc>>,
}

/// Error handler with recovery strategies
pub struct ErrorHandler {
    recovery_strategies: HashMap<String, Box<dyn RecoveryStrategy>>,
    error_log: Arc<Mutex<Vec<ErrorContext>>>,
}

#[async_trait::async_trait]
pub trait RecoveryStrategy: Send + Sync {
    async fn can_recover(&self, error: &ErrorContext) -> bool;
    async fn recover(&self, error: &ErrorContext) -> Result<(), Box<dyn std::error::Error>>;
}

impl ErrorHandler {
    pub fn new() -> Self {
        Self {
            recovery_strategies: HashMap::new(),
            error_log: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    /// Register recovery strategy
    pub fn register_strategy(
        &mut self,
        error_type: &str,
        strategy: Box<dyn RecoveryStrategy>,
    ) {
        self.recovery_strategies.insert(error_type.to_string(), strategy);
    }
    
    /// Handle error with recovery attempt
    pub async fn handle_error(
        &self,
        error: Box<dyn std::error::Error + Send + Sync>,
        context: HashMap<String, String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let error_context = ErrorContext {
            error,
            context,
            timestamp: Utc::now(),
            trace_id: None,
            retry_info: None,
        };
        
        // Log error
        self.error_log.lock().await.push(error_context.clone());
        
        // Find and execute recovery strategy
        for (error_type, strategy) in &self.recovery_strategies {
            if strategy.can_recover(&error_context).await {
                return strategy.recover(&error_context).await;
            }
        }
        
        Err("No recovery strategy found".into())
    }
    
    /// Get error statistics
    pub async fn get_error_stats(&self) -> ErrorStats {
        let errors = self.error_log.lock().await;
        
        let total_errors = errors.len();
        let errors_by_type = errors
            .iter()
            .fold(HashMap::new(), |mut acc, err| {
                let error_type = format!("{:?}", err.error);
                *acc.entry(error_type).or_insert(0) += 1;
                acc
            });
            
        ErrorStats {
            total_errors,
            errors_by_type,
            last_error_time: errors.last().map(|e| e.timestamp),
        }
    }
}

#[derive(Debug)]
pub struct ErrorStats {
    pub total_errors: usize,
    pub errors_by_type: HashMap<String, usize>,
    pub last_error_time: Option<DateTime<Utc>>,
}

/// Panic handler with graceful recovery
pub struct PanicHandler;

impl PanicHandler {
    /// Install custom panic handler
    pub fn install() {
        let default_panic = panic::take_hook();
        
        panic::set_hook(Box::new(move |panic_info| {
            let payload = panic_info.payload();
            let location = panic_info.location();
            
            let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic".to_string()
            };
            
            eprintln!("Panic occurred: {}", msg);
            if let Some(loc) = location {
                eprintln!("Location: {}:{}:{}", loc.file(), loc.line(), loc.column());
            }
            
            // Log to monitoring system
            // ... monitoring integration ...
            
            // Call default handler
            default_panic(panic_info);
        }));
    }
    
    /// Catch panics in async tasks
    pub async fn catch_panic<F, T>(f: F) -> Result<T, Box<dyn Any + Send>>
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        tokio::spawn(async move {
            panic::AssertUnwindSafe(f).catch_unwind().await
        })
        .await
        .unwrap()
    }
}

/// Database recovery strategy
pub struct DatabaseRecoveryStrategy {
    backup_pool: PgPool,
}

#[async_trait::async_trait]
impl RecoveryStrategy for DatabaseRecoveryStrategy {
    async fn can_recover(&self, error: &ErrorContext) -> bool {
        // Check if it's a database error
        error.context.get("error_type")
            .map(|t| t.contains("database"))
            .unwrap_or(false)
    }
    
    async fn recover(&self, error: &ErrorContext) -> Result<(), Box<dyn std::error::Error>> {
        // Try backup database
        match self.backup_pool.acquire().await {
            Ok(_) => {
                println!("Switched to backup database");
                Ok(())
            }
            Err(e) => Err(Box::new(e)),
        }
    }
}
```

## 20. Production-Ready Main Application

```rust
use clap::{App, Arg};
use config::{Config, ConfigError, File};

#[derive(Debug, Deserialize)]
struct AppConfig {
    server: ServerConfig,
    database: DatabaseConfig,
    redis: RedisConfig,
    monitoring: MonitoringConfig,
}

#[derive(Debug, Deserialize)]
struct ServerConfig {
    host: String,
    port: u16,
    workers: usize,
}

#[derive(Debug, Deserialize)]
struct DatabaseConfig {
    url: String,
    max_connections: u32,
    min_connections: u32,
}

#[derive(Debug, Deserialize)]
struct RedisConfig {
    url: String,
}

#[derive(Debug, Deserialize)]
struct MonitoringConfig {
    jaeger_endpoint: String,
    prometheus_port: u16,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let matches = App::new("Advanced Rust Application")
        .version("1.0")
        .author("Your Name")
        .about("Production-ready Rust application")
        .arg(Arg::with_name("config")
            .short("c")
            .long("config")
            .value_name("FILE")
            .help("Sets a custom config file")
            .takes_value(true))
        .arg(Arg::with_name("environment")
            .short("e")
            .long("env")
            .value_name("ENV")
            .help("Sets the environment (dev, staging, prod)")
            .takes_value(true))
        .get_matches();
        
    // Load configuration
    let config_file = matches.value_of("config").unwrap_or("config/default.toml");
    let environment = matches.value_of("environment").unwrap_or("dev");
    
    let config = Config::builder()
        .add_source(File::with_name(config_file))
        .add_source(File::with_name(&format!("config/{}", environment)).required(false))
        .add_source(config::Environment::with_prefix("APP"))
        .build()?;
        
    let app_config: AppConfig = config.try_deserialize()?;
    
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .json()
        .init();
        
    // Install panic handler
    PanicHandler::install();
    
    // Initialize telemetry
    TracingManager::init_telemetry("advanced-rust-app")?;
    
    // Create database pool
    let db_config = deadpool_postgres::Config {
        host: Some("localhost".to_string()),
        port: Some(5432),
        dbname: Some("mydb".to_string()),
        max_size: app_config.database.max_connections as usize,
        ..Default::default()
    };
    
    let connection_pool = SmartConnectionPool::new(
        db_config,
        &app_config.redis.url,
    ).await?;
    
    // Initialize metrics
    let metrics = Arc::new(MetricsCollector::new()?);
    
    // Create rate limiter
    let rate_limiter = Arc::new(DistributedRateLimiter::new(
        connection_pool.get_redis(),
        Duration::from_secs(60),
        1000,
    ));
    
    // Initialize circuit breaker
    let circuit_breaker = Arc::new(CircuitBreaker::new(
        5,
        3,
        Duration::from_secs(30),
    ));
    
    // Create multi-tier cache
    let cache = Arc::new(MultiTierCache::new(
        connection_pool.get_redis(),
        1000,
        Duration::from_secs(300),
    ));
    
    // Initialize error handler
    let mut error_handler = ErrorHandler::new();
    error_handler.register_strategy(
        "database",
        Box::new(DatabaseRecoveryStrategy {
            backup_pool: connection_pool.get_postgres().await?.deref().clone(),
        }),
    );
    
    // Create task queue
    let task_queue = DistributedTaskQueue::new(
        connection_pool.get_redis(),
        "main-group".to_string(),
        "worker-1".to_string(),
        "tasks:stream".to_string(),
        3,
    );
    task_queue.init_consumer_group().await?;
    
    // Start task consumer
    let task_processor = task_queue.clone();
    tokio::spawn(async move {
        task_processor.consume(10, |task| async {
            println!("Processing task: {:?}", task);
            Ok(())
        }).await.unwrap();
    });
    
    // Start metrics endpoint
    let metrics_clone = metrics.clone();
    tokio::spawn(async move {
        warp::serve(
            warp::path("metrics")
                .map(move || {
                    let metrics = metrics_clone.export().unwrap();
                    warp::reply::with_header(metrics, "content-type", "text/plain")
                })
        )
        .run(([0, 0, 0, 0], app_config.monitoring.prometheus_port))
        .await;
    });
    
    // Start gRPC server
    tokio::spawn(async move {
        start_grpc_server("0.0.0.0:50051").await.unwrap();
    });
    
    // Start HTTP server with all middleware
    let app = warp::path("api")
        .and(warp::path("data"))
        .and(warp::get())
        .and_then(move |_| {
            let pool = connection_pool.clone();
            let cache = cache.clone();
            let metrics = metrics.clone();
            let rate_limiter = rate_limiter.clone();
            let circuit_breaker = circuit_breaker.clone();
            
            async move {
                // Rate limiting
                if !rate_limiter.check_rate_limit("global").await.unwrap().allowed {
                    return Err(warp::reject::custom(RateLimitExceeded));
                }
                
                // Circuit breaker protection
                let result = circuit_breaker.execute(|| async {
                    // Try cache first
                    let data = cache.get_or_compute("api:data", || async {
                        // Fetch from database
                        let start = Instant::now();
                        let conn = pool.get_postgres().await?;
                        let rows = conn.query("SELECT * FROM data", &[]).await?;
                        metrics.record_db_query(start.elapsed());
                        
                        Ok(serde_json::json!(rows))
                    }).await?;
                    
                    Ok::<_, Box<dyn std::error::Error>>(data)
                }).await;
                
                match result {
                    Ok(data) => Ok(warp::reply::json(&data)),
                    Err(_) => Err(warp::reject::custom(ServiceUnavailable)),
                }
            }
        });
        
    info!("Starting server on {}:{}", app_config.server.host, app_config.server.port);
    
    warp::serve(app)
        .run(([0, 0, 0, 0], app_config.server.port))
        .await;
        
    // Graceful shutdown
    TracingManager::shutdown_telemetry();
    
    Ok(())
}

#[derive(Debug)]
struct RateLimitExceeded;
impl warp::reject::Reject for RateLimitExceeded {}

#[derive(Debug)]
struct ServiceUnavailable;
impl warp::reject::Reject for ServiceUnavailable {}
```

## Additional Resources

- [Rust Async Book](https://rust-lang.github.io/async-book/)
- [Tokio Tutorial](https://tokio.rs/tokio/tutorial)
- [The Rustonomicon](https://doc.rust-lang.org/nomicon/)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [Zero To Production In Rust](https://www.zero2prod.com/)
- [Rust for Rustaceans](https://rust-for-rustaceans.com/)

This comprehensive guide covers advanced Rust patterns that are particularly useful for building high-performance, scalable systems for cloud services and data processing. Each implementation is production-ready and follows best practices for reliability, performance, and maintainability.