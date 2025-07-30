# Rust High-Performance REST API Tutorial

This tutorial demonstrates how to build a high-performance REST API in Rust using async programming, comprehensive logging, and a corresponding HTTP client.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Server Implementation](#server-implementation)
- [Client Implementation](#client-implementation)
- [Running the Examples](#running-the-examples)
- [Performance Considerations](#performance-considerations)

## Prerequisites

- Rust 1.70+ installed
- Basic understanding of async programming in Rust
- Familiarity with REST API concepts

## Project Structure

```
rust-rest-api/
├── Cargo.toml
├── src/
│   ├── main.rs          # Server implementation
│   └── bin/
│       └── client.rs    # Client implementation
└── README.md
```

## Server Implementation

### Dependencies (Cargo.toml)

```toml
[package]
name = "rust-rest-api"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "server"
path = "src/main.rs"

[[bin]]
name = "client"
path = "src/bin/client.rs"

[dependencies]
# Async runtime
tokio = { version = "1.35", features = ["full"] }

# Web framework
axum = { version = "0.7", features = ["json"] }
tower = { version = "0.4", features = ["util"] }
tower-http = { version = "0.5", features = ["cors", "trace"] }

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Logging
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }

# Error handling
anyhow = "1.0"
thiserror = "1.0"

# HTTP client
reqwest = { version = "0.11", features = ["json", "rustls-tls"] }

# UUID for resource IDs
uuid = { version = "1.6", features = ["v4", "serde"] }

# Time handling
chrono = { version = "0.4", features = ["serde"] }

# Environment variables
dotenv = "0.15"
```

### Server Code (src/main.rs)

```rust
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, patch, post, put},
    Json, Router,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{Arc, RwLock},
};
use tower_http::{
    cors::CorsLayer,
    trace::{DefaultMakeSpan, DefaultOnRequest, DefaultOnResponse, TraceLayer},
};
use tracing::{info, Level};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use uuid::Uuid;

// Error handling
#[derive(Debug, thiserror::Error)]
enum ApiError {
    #[error("Resource not found")]
    NotFound,
    #[error("Bad request: {0}")]
    BadRequest(String),
    #[error("Internal server error")]
    InternalError,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, error_message) = match self {
            ApiError::NotFound => (StatusCode::NOT_FOUND, self.to_string()),
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            ApiError::InternalError => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
        };

        let body = Json(serde_json::json!({
            "error": error_message,
            "timestamp": Utc::now()
        }));

        (status, body).into_response()
    }
}

// Data models
#[derive(Debug, Clone, Serialize, Deserialize)]
struct User {
    id: Uuid,
    username: String,
    email: String,
    full_name: String,
    is_active: bool,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

#[derive(Debug, Deserialize)]
struct CreateUserRequest {
    username: String,
    email: String,
    full_name: String,
}

#[derive(Debug, Deserialize)]
struct UpdateUserRequest {
    username: Option<String>,
    email: Option<String>,
    full_name: Option<String>,
    is_active: Option<bool>,
}

#[derive(Debug, Serialize)]
struct UserResponse {
    id: Uuid,
    username: String,
    email: String,
    full_name: String,
    is_active: bool,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

impl From<User> for UserResponse {
    fn from(user: User) -> Self {
        UserResponse {
            id: user.id,
            username: user.username,
            email: user.email,
            full_name: user.full_name,
            is_active: user.is_active,
            created_at: user.created_at,
            updated_at: user.updated_at,
        }
    }
}

#[derive(Debug, Serialize)]
struct UsersListResponse {
    users: Vec<UserResponse>,
    total: usize,
    page: usize,
    page_size: usize,
}

// Application state
type UsersDb = Arc<RwLock<HashMap<Uuid, User>>>;

#[derive(Clone)]
struct AppState {
    users_db: UsersDb,
}

// Handler functions
async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "timestamp": Utc::now()
    }))
}

async fn get_users(State(state): State<AppState>) -> Result<impl IntoResponse, ApiError> {
    let users = state.users_db
        .read()
        .map_err(|_| ApiError::InternalError)?;
    
    let users_list: Vec<UserResponse> = users
        .values()
        .cloned()
        .map(UserResponse::from)
        .collect();
    
    let response = UsersListResponse {
        total: users_list.len(),
        users: users_list,
        page: 1,
        page_size: 100,
    };
    
    Ok(Json(response))
}

async fn get_user(
    State(state): State<AppState>,
    Path(user_id): Path<Uuid>,
) -> Result<impl IntoResponse, ApiError> {
    let users = state.users_db
        .read()
        .map_err(|_| ApiError::InternalError)?;
    
    let user = users.get(&user_id)
        .ok_or(ApiError::NotFound)?
        .clone();
    
    Ok(Json(UserResponse::from(user)))
}

async fn create_user(
    State(state): State<AppState>,
    Json(payload): Json<CreateUserRequest>,
) -> Result<impl IntoResponse, ApiError> {
    // Validation
    if payload.username.is_empty() || payload.email.is_empty() {
        return Err(ApiError::BadRequest("Username and email are required".to_string()));
    }
    
    let now = Utc::now();
    let user = User {
        id: Uuid::new_v4(),
        username: payload.username,
        email: payload.email,
        full_name: payload.full_name,
        is_active: true,
        created_at: now,
        updated_at: now,
    };
    
    let mut users = state.users_db
        .write()
        .map_err(|_| ApiError::InternalError)?;
    
    let user_id = user.id;
    users.insert(user_id, user.clone());
    
    info!("Created new user: {}", user_id);
    
    Ok((StatusCode::CREATED, Json(UserResponse::from(user))))
}

async fn update_user(
    State(state): State<AppState>,
    Path(user_id): Path<Uuid>,
    Json(payload): Json<UpdateUserRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let mut users = state.users_db
        .write()
        .map_err(|_| ApiError::InternalError)?;
    
    let user = users.get_mut(&user_id)
        .ok_or(ApiError::NotFound)?;
    
    // Apply updates
    if let Some(username) = payload.username {
        user.username = username;
    }
    if let Some(email) = payload.email {
        user.email = email;
    }
    if let Some(full_name) = payload.full_name {
        user.full_name = full_name;
    }
    if let Some(is_active) = payload.is_active {
        user.is_active = is_active;
    }
    
    user.updated_at = Utc::now();
    
    info!("Updated user: {}", user_id);
    
    Ok(Json(UserResponse::from(user.clone())))
}

async fn patch_user(
    State(state): State<AppState>,
    Path(user_id): Path<Uuid>,
    Json(payload): Json<UpdateUserRequest>,
) -> Result<impl IntoResponse, ApiError> {
    // PATCH is identical to PUT in this example, but typically would have different validation
    update_user(State(state), Path(user_id), Json(payload)).await
}

async fn delete_user(
    State(state): State<AppState>,
    Path(user_id): Path<Uuid>,
) -> Result<impl IntoResponse, ApiError> {
    let mut users = state.users_db
        .write()
        .map_err(|_| ApiError::InternalError)?;
    
    users.remove(&user_id)
        .ok_or(ApiError::NotFound)?;
    
    info!("Deleted user: {}", user_id);
    
    Ok(StatusCode::NO_CONTENT)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "rust_rest_api=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer().json())
        .init();

    // Initialize shared state
    let users_db = Arc::new(RwLock::new(HashMap::new()));
    
    // Add some sample data
    {
        let mut users = users_db.write().unwrap();
        let sample_user = User {
            id: Uuid::new_v4(),
            username: "john_doe".to_string(),
            email: "john@example.com".to_string(),
            full_name: "John Doe".to_string(),
            is_active: true,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        users.insert(sample_user.id, sample_user);
    }
    
    let app_state = AppState { users_db };

    // Build our application with routes
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/api/v1/users", get(get_users).post(create_user))
        .route(
            "/api/v1/users/:id",
            get(get_user)
                .put(update_user)
                .patch(patch_user)
                .delete(delete_user),
        )
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(DefaultMakeSpan::new().level(Level::INFO))
                .on_request(DefaultOnRequest::new().level(Level::INFO))
                .on_response(DefaultOnResponse::new().level(Level::INFO)),
        )
        .layer(CorsLayer::permissive())
        .with_state(app_state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    info!("Server listening on {}", addr);

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}
```

## Client Implementation

### Client Code (src/bin/client.rs)

```rust
use anyhow::Result;
use chrono::{DateTime, Utc};
use reqwest::{Certificate, Client, ClientBuilder};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use uuid::Uuid;

// Reuse the data models from server
#[derive(Debug, Serialize, Deserialize)]
struct User {
    id: Uuid,
    username: String,
    email: String,
    full_name: String,
    is_active: bool,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

#[derive(Debug, Serialize)]
struct CreateUserRequest {
    username: String,
    email: String,
    full_name: String,
}

#[derive(Debug, Serialize)]
struct UpdateUserRequest {
    username: Option<String>,
    email: Option<String>,
    full_name: Option<String>,
    is_active: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct UsersListResponse {
    users: Vec<User>,
    total: usize,
    page: usize,
    page_size: usize,
}

#[derive(Debug, Deserialize)]
struct ErrorResponse {
    error: String,
    timestamp: DateTime<Utc>,
}

struct ApiClient {
    client: Client,
    base_url: String,
}

impl ApiClient {
    /// Create a new API client with default settings (secure mode)
    pub fn new(base_url: String) -> Result<Self> {
        let client = ClientBuilder::new()
            .timeout(Duration::from_secs(30))
            .connect_timeout(Duration::from_secs(10))
            .build()?;

        Ok(Self { client, base_url })
    }

    /// Create an API client that accepts self-signed certificates (insecure mode)
    pub fn new_insecure(base_url: String) -> Result<Self> {
        let client = ClientBuilder::new()
            .danger_accept_invalid_certs(true)
            .timeout(Duration::from_secs(30))
            .connect_timeout(Duration::from_secs(10))
            .build()?;

        Ok(Self { client, base_url })
    }

    /// Create an API client with a custom CA certificate
    pub fn new_with_ca_cert(base_url: String, cert_pem: &[u8]) -> Result<Self> {
        let cert = Certificate::from_pem(cert_pem)?;
        let client = ClientBuilder::new()
            .add_root_certificate(cert)
            .timeout(Duration::from_secs(30))
            .connect_timeout(Duration::from_secs(10))
            .build()?;

        Ok(Self { client, base_url })
    }

    /// Health check endpoint
    pub async fn health_check(&self) -> Result<()> {
        let url = format!("{}/health", self.base_url);
        let response = self.client.get(&url).send().await?;

        if response.status().is_success() {
            let health: serde_json::Value = response.json().await?;
            info!("Health check response: {:?}", health);
            Ok(())
        } else {
            anyhow::bail!("Health check failed with status: {}", response.status());
        }
    }

    /// GET all users
    pub async fn get_users(&self) -> Result<UsersListResponse> {
        let url = format!("{}/api/v1/users", self.base_url);
        debug!("GET {}", url);

        let response = self.client.get(&url).send().await?;

        if response.status().is_success() {
            let users: UsersListResponse = response.json().await?;
            Ok(users)
        } else {
            let error: ErrorResponse = response.json().await?;
            anyhow::bail!("Failed to get users: {}", error.error);
        }
    }

    /// GET a specific user
    pub async fn get_user(&self, user_id: Uuid) -> Result<User> {
        let url = format!("{}/api/v1/users/{}", self.base_url, user_id);
        debug!("GET {}", url);

        let response = self.client.get(&url).send().await?;

        if response.status().is_success() {
            let user: User = response.json().await?;
            Ok(user)
        } else if response.status() == 404 {
            anyhow::bail!("User not found");
        } else {
            let error: ErrorResponse = response.json().await?;
            anyhow::bail!("Failed to get user: {}", error.error);
        }
    }

    /// POST create a new user
    pub async fn create_user(&self, request: CreateUserRequest) -> Result<User> {
        let url = format!("{}/api/v1/users", self.base_url);
        debug!("POST {} with payload: {:?}", url, request);

        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            let user: User = response.json().await?;
            Ok(user)
        } else {
            let error: ErrorResponse = response.json().await?;
            anyhow::bail!("Failed to create user: {}", error.error);
        }
    }

    /// PUT update a user (full update)
    pub async fn update_user(&self, user_id: Uuid, request: UpdateUserRequest) -> Result<User> {
        let url = format!("{}/api/v1/users/{}", self.base_url, user_id);
        debug!("PUT {} with payload: {:?}", url, request);

        let response = self.client
            .put(&url)
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            let user: User = response.json().await?;
            Ok(user)
        } else {
            let error: ErrorResponse = response.json().await?;
            anyhow::bail!("Failed to update user: {}", error.error);
        }
    }

    /// PATCH update a user (partial update)
    pub async fn patch_user(&self, user_id: Uuid, request: UpdateUserRequest) -> Result<User> {
        let url = format!("{}/api/v1/users/{}", self.base_url, user_id);
        debug!("PATCH {} with payload: {:?}", url, request);

        let response = self.client
            .patch(&url)
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            let user: User = response.json().await?;
            Ok(user)
        } else {
            let error: ErrorResponse = response.json().await?;
            anyhow::bail!("Failed to patch user: {}", error.error);
        }
    }

    /// DELETE a user
    pub async fn delete_user(&self, user_id: Uuid) -> Result<()> {
        let url = format!("{}/api/v1/users/{}", self.base_url, user_id);
        debug!("DELETE {}", url);

        let response = self.client.delete(&url).send().await?;

        if response.status().is_success() || response.status() == 204 {
            Ok(())
        } else {
            let error: ErrorResponse = response.json().await?;
            anyhow::bail!("Failed to delete user: {}", error.error);
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "client=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Create API client
    let base_url = "http://localhost:3000".to_string();
    let client = ApiClient::new(base_url)?;

    // For insecure mode (e.g., self-signed certificates):
    // let client = ApiClient::new_insecure("https://localhost:3000".to_string())?;

    info!("Starting API client demo");

    // 1. Health check
    info!("Performing health check...");
    client.health_check().await?;

    // 2. Get all users
    info!("Getting all users...");
    let users_response = client.get_users().await?;
    info!("Found {} users", users_response.total);
    for user in &users_response.users {
        info!("  - {} ({})", user.username, user.email);
    }

    // 3. Create a new user
    info!("Creating a new user...");
    let new_user_request = CreateUserRequest {
        username: "jane_smith".to_string(),
        email: "jane@example.com".to_string(),
        full_name: "Jane Smith".to_string(),
    };
    let created_user = client.create_user(new_user_request).await?;
    info!("Created user: {:?}", created_user);

    // 4. Get the specific user
    info!("Getting user by ID...");
    let fetched_user = client.get_user(created_user.id).await?;
    info!("Fetched user: {:?}", fetched_user);

    // 5. Update the user (PUT)
    info!("Updating user with PUT...");
    let update_request = UpdateUserRequest {
        username: Some("jane_doe".to_string()),
        email: Some("jane.doe@example.com".to_string()),
        full_name: Some("Jane Doe".to_string()),
        is_active: Some(true),
    };
    let updated_user = client.update_user(created_user.id, update_request).await?;
    info!("Updated user: {:?}", updated_user);

    // 6. Partially update the user (PATCH)
    info!("Updating user with PATCH...");
    let patch_request = UpdateUserRequest {
        username: None,
        email: None,
        full_name: None,
        is_active: Some(false),
    };
    let patched_user = client.patch_user(created_user.id, patch_request).await?;
    info!("Patched user: {:?}", patched_user);

    // 7. Get all users again
    info!("Getting all users after updates...");
    let users_response = client.get_users().await?;
    info!("Total users: {}", users_response.total);

    // 8. Delete the user
    info!("Deleting user...");
    client.delete_user(created_user.id).await?;
    info!("User deleted successfully");

    // 9. Try to get the deleted user (should fail)
    info!("Trying to get deleted user...");
    match client.get_user(created_user.id).await {
        Ok(_) => info!("Unexpected: User still exists"),
        Err(e) => info!("Expected error: {}", e),
    }

    info!("Client demo completed successfully!");

    Ok(())
}
```

## Running the Examples

### 1. Start the Server

```bash
cargo run --bin server
```

The server will start on `http://localhost:3000` with JSON logging enabled.

### 2. Run the Client

In another terminal:

```bash
cargo run --bin client
```

### 3. Test with curl

You can also test the API endpoints directly with curl:

```bash
# Health check
curl http://localhost:3000/health

# Get all users
curl http://localhost:3000/api/v1/users

# Create a user
curl -X POST http://localhost:3000/api/v1/users \
  -H "Content-Type: application/json" \
  -d '{
    "username": "test_user",
    "email": "test@example.com",
    "full_name": "Test User"
  }'

# Get a specific user (replace with actual UUID)
curl http://localhost:3000/api/v1/users/{user-id}

# Update a user (PUT)
curl -X PUT http://localhost:3000/api/v1/users/{user-id} \
  -H "Content-Type: application/json" \
  -d '{
    "username": "updated_user",
    "email": "updated@example.com",
    "full_name": "Updated User",
    "is_active": true
  }'

# Partially update a user (PATCH)
curl -X PATCH http://localhost:3000/api/v1/users/{user-id} \
  -H "Content-Type: application/json" \
  -d '{
    "is_active": false
  }'

# Delete a user
curl -X DELETE http://localhost:3000/api/v1/users/{user-id}
```

## Performance Considerations

### 1. Async Runtime
- Uses Tokio for high-performance async I/O
- Handles thousands of concurrent connections efficiently
- Non-blocking operations throughout the stack

### 2. Connection Pooling
- The client reuses connections via reqwest's built-in connection pooling
- Reduces latency for subsequent requests

### 3. Logging and Tracing
- Structured JSON logging for production environments
- Configurable log levels via environment variables
- Request/response tracing for debugging

### 4. Error Handling
- Type-safe error handling with custom error types
- Proper HTTP status codes for different error scenarios
- Detailed error messages in JSON format

### 5. Serialization
- Efficient JSON serialization with Serde
- Zero-copy deserialization where possible

### 6. Database Considerations
For production use, replace the in-memory HashMap with:
- PostgreSQL with `sqlx` or `diesel` for async database operations
- Connection pooling with `deadpool` or `bb8`
- Prepared statements for better performance

### 7. Additional Optimizations
- Add caching layer (Redis) for frequently accessed data
- Implement pagination for large datasets
- Use compression (gzip) for response bodies
- Add rate limiting to prevent abuse
- Implement request validation middleware

## Security Considerations

### TLS/HTTPS Support
The client implementation supports three modes:
1. **Secure (default)**: Validates SSL certificates
2. **Insecure**: Accepts self-signed certificates (development only)
3. **Custom CA**: Uses provided CA certificate for validation

### Authentication & Authorization
For production, add:
- JWT token authentication
- API key management
- Role-based access control (RBAC)
- Request signing for additional security

## Next Steps

1. Add database integration with PostgreSQL
2. Implement authentication middleware
3. Add OpenAPI/Swagger documentation
4. Create integration tests
5. Set up CI/CD pipeline
6. Add metrics collection (Prometheus)
7. Implement graceful shutdown
8. Add request/response compression