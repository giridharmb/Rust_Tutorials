# RUST Retry Logic Guide with Exponential Backoff

## Table of Contents
- [Dependencies](#dependencies)
- [Basic Retry Logic for Functions](#basic-retry-logic-for-functions)
- [Advanced Retry with Configurable Options](#advanced-retry-with-configurable-options)
- [HTTP Client with Retry Logic](#http-client-with-retry-logic)
  - [Secure HTTPS Client](#secure-https-client)
  - [Insecure HTTP Client](#insecure-http-client)
- [Complete Working Examples](#complete-working-examples)

## Dependencies

Add these dependencies to your `Cargo.toml`:

```toml
[dependencies]
tokio = { version = "1.35", features = ["full"] }
reqwest = { version = "0.11", features = ["json", "rustls-tls"] }
log = "0.4"
env_logger = "0.11"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
rand = "0.8"
thiserror = "1.0"
async-trait = "0.1"
```

## Basic Retry Logic for Functions

### Simple Retry Implementation

```rust
use log::{info, warn, error};
use std::time::Duration;
use tokio::time::sleep;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RetryError {
    #[error("Maximum retries exceeded: {0}")]
    MaxRetriesExceeded(String),
    #[error("Operation failed: {0}")]
    OperationFailed(String),
}

/// Basic retry configuration
#[derive(Clone, Debug)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub exponential_base: f64,
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay_ms: 100,
            max_delay_ms: 10_000,
            exponential_base: 2.0,
            jitter: true,
        }
    }
}

/// Calculate the next delay with exponential backoff
fn calculate_backoff(config: &RetryConfig, attempt: u32) -> Duration {
    let exponential_delay = (config.initial_delay_ms as f64) 
        * config.exponential_base.powi(attempt as i32);
    
    let mut delay_ms = exponential_delay.min(config.max_delay_ms as f64) as u64;
    
    // Add jitter to prevent thundering herd problem
    if config.jitter {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let jitter_range = (delay_ms as f64 * 0.3) as u64;
        delay_ms = delay_ms + rng.gen_range(0..=jitter_range);
    }
    
    Duration::from_millis(delay_ms)
}

/// Generic retry function for any async operation
pub async fn retry_with_backoff<F, Fut, T, E>(
    config: RetryConfig,
    operation_name: &str,
    mut operation: F,
) -> Result<T, RetryError>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
    E: std::fmt::Display,
{
    let mut attempt = 0;
    
    loop {
        match operation().await {
            Ok(result) => {
                if attempt > 0 {
                    info!(
                        "Operation '{}' succeeded after {} retries",
                        operation_name,
                        attempt
                    );
                }
                return Ok(result);
            }
            Err(e) => {
                if attempt >= config.max_retries {
                    error!(
                        "Operation '{}' failed after {} attempts: {}",
                        operation_name,
                        attempt + 1,
                        e
                    );
                    return Err(RetryError::MaxRetriesExceeded(e.to_string()));
                }
                
                let delay = calculate_backoff(&config, attempt);
                warn!(
                    "Operation '{}' failed (attempt {}/{}): {}. Retrying in {:?}",
                    operation_name,
                    attempt + 1,
                    config.max_retries + 1,
                    e,
                    delay
                );
                
                sleep(delay).await;
                attempt += 1;
            }
        }
    }
}

/// Example usage with a simple function
async fn example_function_with_retry() -> Result<String, RetryError> {
    let config = RetryConfig::default();
    
    // Simulate a function that might fail
    let mut counter = 0;
    let result = retry_with_backoff(
        config,
        "fetch_data",
        || async {
            counter += 1;
            if counter < 3 {
                Err("Simulated failure")
            } else {
                Ok("Success!".to_string())
            }
        },
    )
    .await?;
    
    Ok(result)
}
```

## Advanced Retry with Configurable Options

### Retry Policy with Different Strategies

```rust
use async_trait::async_trait;
use std::fmt::Debug;

/// Trait for determining if an error is retryable
#[async_trait]
pub trait RetryableError: Debug + std::fmt::Display {
    fn is_retryable(&self) -> bool;
}

/// Advanced retry policy
#[derive(Clone)]
pub struct RetryPolicy {
    pub config: RetryConfig,
    pub retry_on: Vec<Box<dyn Fn(&dyn std::any::Any) -> bool + Send + Sync>>,
}

impl RetryPolicy {
    pub fn new(config: RetryConfig) -> Self {
        Self {
            config,
            retry_on: vec![],
        }
    }
    
    /// Add a condition for retrying
    pub fn with_retry_condition<F>(mut self, condition: F) -> Self
    where
        F: Fn(&dyn std::any::Any) -> bool + Send + Sync + 'static,
    {
        self.retry_on.push(Box::new(condition));
        self
    }
}

/// Advanced retry executor
pub struct RetryExecutor {
    policy: RetryPolicy,
}

impl RetryExecutor {
    pub fn new(policy: RetryPolicy) -> Self {
        Self { policy }
    }
    
    pub async fn execute<F, Fut, T, E>(
        &self,
        operation_name: &str,
        mut operation: F,
    ) -> Result<T, RetryError>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
        E: std::error::Error + 'static,
    {
        let mut attempt = 0;
        let config = &self.policy.config;
        
        loop {
            match operation().await {
                Ok(result) => {
                    if attempt > 0 {
                        info!(
                            "Operation '{}' succeeded after {} retries",
                            operation_name, attempt
                        );
                    }
                    return Ok(result);
                }
                Err(e) => {
                    // Check if error is retryable using custom conditions
                    let should_retry = self.policy.retry_on.is_empty() || 
                        self.policy.retry_on.iter().any(|f| f(&e as &dyn std::any::Any));
                    
                    if !should_retry || attempt >= config.max_retries {
                        error!(
                            "Operation '{}' failed permanently after {} attempts: {}",
                            operation_name,
                            attempt + 1,
                            e
                        );
                        return Err(RetryError::OperationFailed(e.to_string()));
                    }
                    
                    let delay = calculate_backoff(config, attempt);
                    warn!(
                        "Operation '{}' failed (attempt {}/{}): {}. Retrying in {:?}",
                        operation_name,
                        attempt + 1,
                        config.max_retries + 1,
                        e,
                        delay
                    );
                    
                    sleep(delay).await;
                    attempt += 1;
                }
            }
        }
    }
}
```

## HTTP Client with Retry Logic

### Secure HTTPS Client

```rust
use reqwest::{Client, Response, Error as ReqwestError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct HttpRetryClient {
    client: Client,
    retry_config: RetryConfig,
}

impl HttpRetryClient {
    /// Create a new secure HTTPS client with retry capabilities
    pub fn new_secure(retry_config: RetryConfig) -> Result<Self, ReqwestError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .connect_timeout(Duration::from_secs(10))
            .build()?;
            
        Ok(Self {
            client,
            retry_config,
        })
    }
    
    /// Create an insecure client (accepts invalid certificates)
    pub fn new_insecure(retry_config: RetryConfig) -> Result<Self, ReqwestError> {
        let client = Client::builder()
            .danger_accept_invalid_certs(true)
            .timeout(Duration::from_secs(30))
            .connect_timeout(Duration::from_secs(10))
            .build()?;
            
        Ok(Self {
            client,
            retry_config,
        })
    }
    
    /// Determine if an HTTP error is retryable
    fn is_retryable_error(error: &ReqwestError) -> bool {
        if error.is_timeout() || error.is_connect() {
            return true;
        }
        
        if let Some(status) = error.status() {
            // Retry on 5xx errors and specific 4xx errors
            return status.is_server_error() || 
                   status.as_u16() == 429 || // Too Many Requests
                   status.as_u16() == 408;   // Request Timeout
        }
        
        true // Default to retry for unknown errors
    }
    
    /// Execute GET request with retry
    pub async fn get(&self, url: &str) -> Result<Response, RetryError> {
        retry_with_backoff(
            self.retry_config.clone(),
            &format!("GET {}", url),
            || async {
                let response = self.client
                    .get(url)
                    .send()
                    .await?;
                    
                if response.status().is_success() {
                    Ok(response)
                } else {
                    Err(ReqwestError::from(response))
                }
            },
        )
        .await
    }
    
    /// Execute POST request with JSON body and retry
    pub async fn post_json<T: Serialize>(
        &self,
        url: &str,
        body: &T,
    ) -> Result<Response, RetryError> {
        retry_with_backoff(
            self.retry_config.clone(),
            &format!("POST {}", url),
            || async {
                let response = self.client
                    .post(url)
                    .json(body)
                    .send()
                    .await?;
                    
                if response.status().is_success() {
                    Ok(response)
                } else {
                    Err(ReqwestError::from(response))
                }
            },
        )
        .await
    }
    
    /// Execute request with custom retry logic
    pub async fn execute_with_custom_retry<F, Fut>(
        &self,
        operation_name: &str,
        request_fn: F,
    ) -> Result<Response, RetryError>
    where
        F: Fn(&Client) -> Fut,
        Fut: std::future::Future<Output = Result<Response, ReqwestError>>,
    {
        let mut attempt = 0;
        let config = &self.retry_config;
        
        loop {
            match request_fn(&self.client).await {
                Ok(response) => {
                    if response.status().is_success() {
                        if attempt > 0 {
                            info!("HTTP request '{}' succeeded after {} retries", 
                                  operation_name, attempt);
                        }
                        return Ok(response);
                    } else {
                        let status = response.status();
                        let error_body = response.text().await.unwrap_or_default();
                        
                        if !status.is_server_error() && status.as_u16() != 429 {
                            error!("HTTP request '{}' failed with status {}: {}",
                                   operation_name, status, error_body);
                            return Err(RetryError::OperationFailed(
                                format!("HTTP {} - {}", status, error_body)
                            ));
                        }
                        
                        // Continue to retry logic for retryable status codes
                        if attempt >= config.max_retries {
                            error!("HTTP request '{}' failed after {} attempts",
                                   operation_name, attempt + 1);
                            return Err(RetryError::MaxRetriesExceeded(
                                format!("HTTP {} - {}", status, error_body)
                            ));
                        }
                    }
                }
                Err(e) => {
                    if !Self::is_retryable_error(&e) || attempt >= config.max_retries {
                        error!("HTTP request '{}' failed: {}", operation_name, e);
                        return Err(RetryError::OperationFailed(e.to_string()));
                    }
                    
                    warn!("HTTP request '{}' failed (attempt {}/{}): {}",
                          operation_name, attempt + 1, config.max_retries + 1, e);
                }
            }
            
            let delay = calculate_backoff(config, attempt);
            warn!("Retrying HTTP request '{}' in {:?}", operation_name, delay);
            sleep(delay).await;
            attempt += 1;
        }
    }
}
```

## Complete Working Examples

### Example 1: Database Operation with Retry

```rust
use tokio_postgres::{Client as PgClient, Error as PgError};

/// Example of retrying database operations
pub struct DatabaseService {
    retry_config: RetryConfig,
}

impl DatabaseService {
    pub fn new() -> Self {
        Self {
            retry_config: RetryConfig {
                max_retries: 5,
                initial_delay_ms: 50,
                max_delay_ms: 5000,
                exponential_base: 2.0,
                jitter: true,
            },
        }
    }
    
    pub async fn execute_query(
        &self,
        client: &PgClient,
        query: &str,
    ) -> Result<Vec<tokio_postgres::Row>, RetryError> {
        retry_with_backoff(
            self.retry_config.clone(),
            "database_query",
            || async {
                client
                    .query(query, &[])
                    .await
                    .map_err(|e| format!("Database error: {}", e))
            },
        )
        .await
    }
}
```

### Example 2: API Client with Comprehensive Retry

```rust
#[derive(Debug, Serialize, Deserialize)]
pub struct ApiResponse {
    pub data: serde_json::Value,
    pub status: String,
}

pub struct ApiClient {
    http_client: HttpRetryClient,
    base_url: String,
}

impl ApiClient {
    pub fn new(base_url: String, insecure: bool) -> Result<Self, ReqwestError> {
        let retry_config = RetryConfig {
            max_retries: 3,
            initial_delay_ms: 200,
            max_delay_ms: 15_000,
            exponential_base: 2.5,
            jitter: true,
        };
        
        let http_client = if insecure {
            HttpRetryClient::new_insecure(retry_config)?
        } else {
            HttpRetryClient::new_secure(retry_config)?
        };
        
        Ok(Self {
            http_client,
            base_url,
        })
    }
    
    pub async fn fetch_data(&self, endpoint: &str) -> Result<ApiResponse, RetryError> {
        let url = format!("{}/{}", self.base_url, endpoint);
        
        info!("Fetching data from: {}", url);
        
        let response = self.http_client.get(&url).await?;
        let api_response = response
            .json::<ApiResponse>()
            .await
            .map_err(|e| RetryError::OperationFailed(e.to_string()))?;
            
        Ok(api_response)
    }
    
    pub async fn submit_data<T: Serialize>(
        &self,
        endpoint: &str,
        data: &T,
    ) -> Result<ApiResponse, RetryError> {
        let url = format!("{}/{}", self.base_url, endpoint);
        
        info!("Submitting data to: {}", url);
        
        let response = self.http_client.post_json(&url, data).await?;
        let api_response = response
            .json::<ApiResponse>()
            .await
            .map_err(|e| RetryError::OperationFailed(e.to_string()))?;
            
        Ok(api_response)
    }
}
```

### Example 3: Main Application with Full Logging Setup

```rust
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();
    
    info!("Starting application with retry examples");
    
    // Example 1: Simple function retry
    info!("Testing simple function retry...");
    match example_function_with_retry().await {
        Ok(result) => info!("Function succeeded: {}", result),
        Err(e) => error!("Function failed: {}", e),
    }
    
    // Example 2: HTTP client with retry (secure)
    info!("Testing secure HTTP client...");
    let secure_client = ApiClient::new("https://api.example.com".to_string(), false)?;
    match secure_client.fetch_data("users").await {
        Ok(data) => info!("Secure API call succeeded: {:?}", data),
        Err(e) => error!("Secure API call failed: {}", e),
    }
    
    // Example 3: HTTP client with retry (insecure)
    info!("Testing insecure HTTP client...");
    let insecure_client = ApiClient::new("http://localhost:8080".to_string(), true)?;
    match insecure_client.fetch_data("health").await {
        Ok(data) => info!("Insecure API call succeeded: {:?}", data),
        Err(e) => error!("Insecure API call failed: {}", e),
    }
    
    // Example 4: Custom retry executor with conditions
    info!("Testing custom retry executor...");
    let policy = RetryPolicy::new(RetryConfig::default())
        .with_retry_condition(|_| true); // Always retry
        
    let executor = RetryExecutor::new(policy);
    let result = executor.execute(
        "custom_operation",
        || async {
            // Your custom operation here
            Ok::<_, std::io::Error>("Custom operation succeeded".to_string())
        },
    ).await;
    
    match result {
        Ok(msg) => info!("{}", msg),
        Err(e) => error!("Custom operation failed: {}", e),
    }
    
    Ok(())
}
```

### Example 4: Integration with BigQuery Client

```rust
/// Example integration with Google BigQuery using retry logic
pub struct BigQueryRetryClient {
    retry_config: RetryConfig,
}

impl BigQueryRetryClient {
    pub fn new() -> Self {
        Self {
            retry_config: RetryConfig {
                max_retries: 3,
                initial_delay_ms: 500,
                max_delay_ms: 30_000,
                exponential_base: 2.0,
                jitter: true,
            },
        }
    }
    
    pub async fn execute_query(&self, query: &str) -> Result<Vec<serde_json::Value>, RetryError> {
        retry_with_backoff(
            self.retry_config.clone(),
            "bigquery_execute",
            || async {
                // Simulate BigQuery client call
                // In real implementation, use actual BigQuery client
                info!("Executing BigQuery: {}", query);
                
                // Mock implementation
                Ok(vec![serde_json::json!({
                    "result": "mock_data"
                })])
            },
        )
        .await
    }
}
```

## Best Practices

1. **Configure Appropriate Delays**: Start with small delays (50-200ms) and cap maximum delays based on your use case.

2. **Add Jitter**: Always use jitter to prevent thundering herd problems when multiple clients retry simultaneously.

3. **Log Everything**: Use structured logging to track retry attempts, delays, and final outcomes.

4. **Be Selective About Retries**: Not all errors should be retried. Network timeouts and 5xx errors are good candidates, but 4xx errors (except 429) usually aren't.

5. **Set Reasonable Limits**: Don't retry indefinitely. Set maximum retry counts based on your SLA requirements.

6. **Monitor Retry Metrics**: Track retry rates, success rates after retries, and average retry counts to optimize your configuration.

7. **Circuit Breaker Pattern**: For production systems, consider implementing circuit breakers to prevent cascading failures.

## Testing Retry Logic

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_exponential_backoff() {
        let config = RetryConfig {
            max_retries: 3,
            initial_delay_ms: 100,
            max_delay_ms: 1000,
            exponential_base: 2.0,
            jitter: false,
        };
        
        assert_eq!(calculate_backoff(&config, 0).as_millis(), 100);
        assert_eq!(calculate_backoff(&config, 1).as_millis(), 200);
        assert_eq!(calculate_backoff(&config, 2).as_millis(), 400);
        assert_eq!(calculate_backoff(&config, 3).as_millis(), 800);
        assert_eq!(calculate_backoff(&config, 4).as_millis(), 1000); // Capped at max
    }
    
    #[tokio::test]
    async fn test_retry_success_after_failures() {
        let config = RetryConfig {
            max_retries: 3,
            initial_delay_ms: 10,
            max_delay_ms: 100,
            exponential_base: 2.0,
            jitter: false,
        };
        
        let counter = std::sync::Arc::new(std::sync::Mutex::new(0));
        let counter_clone = counter.clone();
        
        let result = retry_with_backoff(
            config,
            "test_operation",
            || async {
                let mut count = counter_clone.lock().unwrap();
                *count += 1;
                if *count < 3 {
                    Err("Simulated failure")
                } else {
                    Ok("Success")
                }
            },
        )
        .await;
        
        assert!(result.is_ok());
        assert_eq!(*counter.lock().unwrap(), 3);
    }
}
```

This guide provides production-ready retry implementations with exponential backoff for both general functions and HTTP clients in Rust. The code is designed to be extensible and follows Rust best practices with comprehensive error handling and logging.