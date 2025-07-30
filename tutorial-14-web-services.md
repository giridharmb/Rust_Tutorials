# Tutorial 14: Web Services - REST APIs

## Setting Up Web Frameworks

This tutorial covers building REST APIs with Actix-web and Axum, two popular Rust web frameworks.

```toml
# Cargo.toml for Actix-web
[dependencies]
actix-web = "4"
actix-rt = "2"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
env_logger = "0.10"
log = "0.4"
chrono = { version = "0.4", features = ["serde"] }
uuid = { version = "1.0", features = ["v4", "serde"] }
sqlx = { version = "0.7", features = ["runtime-actix-native-tls", "postgres", "uuid", "chrono"] }
jsonwebtoken = "9"
bcrypt = "0.15"
validator = { version = "0.16", features = ["derive"] }
dotenv = "0.15"

# Cargo.toml for Axum
[dependencies]
axum = "0.7"
tokio = { version = "1", features = ["full"] }
tower = "0.4"
tower-http = { version = "0.5", features = ["cors", "trace"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
sqlx = { version = "0.7", features = ["runtime-tokio-native-tls", "postgres", "uuid", "chrono"] }
jsonwebtoken = "9"
bcrypt = "0.15"
validator = { version = "0.16", features = ["derive"] }
```

## Actix-web REST API

```rust
// src/main.rs - Actix-web implementation
use actix_web::{web, App, HttpResponse, HttpServer, Result, middleware};
use serde::{Deserialize, Serialize};
use sqlx::postgres::PgPool;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use validator::Validate;

// Request/Response models
#[derive(Debug, Deserialize, Validate)]
struct CreateUserRequest {
    #[validate(length(min = 3, max = 50))]
    username: String,
    #[validate(email)]
    email: String,
    #[validate(length(min = 8))]
    password: String,
}

#[derive(Debug, Serialize)]
struct UserResponse {
    id: Uuid,
    username: String,
    email: String,
    created_at: DateTime<Utc>,
}

#[derive(Debug, Deserialize, Validate)]
struct LoginRequest {
    #[validate(email)]
    email: String,
    password: String,
}

#[derive(Debug, Serialize)]
struct LoginResponse {
    token: String,
    user: UserResponse,
}

#[derive(Debug, Deserialize)]
struct PaginationParams {
    page: Option<u32>,
    per_page: Option<u32>,
}

#[derive(Debug, Serialize)]
struct PaginatedResponse<T> {
    data: Vec<T>,
    total: i64,
    page: u32,
    per_page: u32,
    total_pages: u32,
}

// Database models
#[derive(Debug, sqlx::FromRow)]
struct User {
    id: Uuid,
    username: String,
    email: String,
    password_hash: String,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

// Error handling
#[derive(Debug)]
enum ApiError {
    BadRequest(String),
    Unauthorized,
    NotFound,
    Conflict(String),
    Internal(String),
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiError::BadRequest(msg) => write!(f, "Bad request: {}", msg),
            ApiError::Unauthorized => write!(f, "Unauthorized"),
            ApiError::NotFound => write!(f, "Not found"),
            ApiError::Conflict(msg) => write!(f, "Conflict: {}", msg),
            ApiError::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl actix_web::error::ResponseError for ApiError {
    fn error_response(&self) -> HttpResponse {
        match self {
            ApiError::BadRequest(msg) => HttpResponse::BadRequest().json(serde_json::json!({
                "error": "Bad Request",
                "message": msg
            })),
            ApiError::Unauthorized => HttpResponse::Unauthorized().json(serde_json::json!({
                "error": "Unauthorized",
                "message": "Invalid credentials"
            })),
            ApiError::NotFound => HttpResponse::NotFound().json(serde_json::json!({
                "error": "Not Found",
                "message": "Resource not found"
            })),
            ApiError::Conflict(msg) => HttpResponse::Conflict().json(serde_json::json!({
                "error": "Conflict",
                "message": msg
            })),
            ApiError::Internal(msg) => {
                log::error!("Internal error: {}", msg);
                HttpResponse::InternalServerError().json(serde_json::json!({
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred"
                }))
            }
        }
    }
}

// JWT Authentication
use jsonwebtoken::{encode, decode, Header, Validation, EncodingKey, DecodingKey};

#[derive(Debug, Serialize, Deserialize)]
struct Claims {
    sub: Uuid,  // user id
    exp: usize, // expiration time
    iat: usize, // issued at
}

fn create_jwt(user_id: Uuid) -> Result<String, ApiError> {
    let now = chrono::Utc::now();
    let exp = (now + chrono::Duration::hours(24)).timestamp() as usize;
    let iat = now.timestamp() as usize;
    
    let claims = Claims {
        sub: user_id,
        exp,
        iat,
    };
    
    let secret = std::env::var("JWT_SECRET")
        .map_err(|_| ApiError::Internal("JWT_SECRET not set".to_string()))?;
    
    encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(secret.as_ref()),
    )
    .map_err(|e| ApiError::Internal(format!("Failed to create token: {}", e)))
}

// Middleware for authentication
use actix_web::{dev::ServiceRequest, Error, FromRequest, HttpMessage};
use actix_web_httpauth::extractors::bearer::BearerAuth;
use actix_web_httpauth::extractors::AuthenticationError;
use actix_web_httpauth::middleware::HttpAuthentication;

async fn validator(
    req: ServiceRequest,
    credentials: BearerAuth,
) -> Result<ServiceRequest, (Error, ServiceRequest)> {
    let secret = std::env::var("JWT_SECRET")
        .map_err(|_| (ApiError::Internal("JWT_SECRET not set".to_string()).into(), req))?;
    
    let token = credentials.token();
    
    let token_data = decode::<Claims>(
        token,
        &DecodingKey::from_secret(secret.as_ref()),
        &Validation::default(),
    )
    .map_err(|_| {
        let config = req
            .app_data::<actix_web_httpauth::extractors::bearer::Config>()
            .cloned()
            .unwrap_or_default();
        (AuthenticationError::from(config).into(), req)
    })?;
    
    req.extensions_mut().insert(token_data.claims.sub);
    Ok(req)
}

// Repository layer
struct UserRepository;

impl UserRepository {
    async fn create(
        pool: &PgPool,
        req: CreateUserRequest,
    ) -> Result<User, sqlx::Error> {
        let password_hash = bcrypt::hash(&req.password, 12)
            .map_err(|e| sqlx::Error::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
        
        sqlx::query_as!(
            User,
            r#"
            INSERT INTO users (username, email, password_hash)
            VALUES ($1, $2, $3)
            RETURNING id, username, email, password_hash, created_at, updated_at
            "#,
            req.username,
            req.email,
            password_hash
        )
        .fetch_one(pool)
        .await
    }
    
    async fn find_by_email(pool: &PgPool, email: &str) -> Result<Option<User>, sqlx::Error> {
        sqlx::query_as!(
            User,
            r#"
            SELECT id, username, email, password_hash, created_at, updated_at
            FROM users
            WHERE email = $1
            "#,
            email
        )
        .fetch_optional(pool)
        .await
    }
    
    async fn find_by_id(pool: &PgPool, id: Uuid) -> Result<Option<User>, sqlx::Error> {
        sqlx::query_as!(
            User,
            r#"
            SELECT id, username, email, password_hash, created_at, updated_at
            FROM users
            WHERE id = $1
            "#,
            id
        )
        .fetch_optional(pool)
        .await
    }
    
    async fn list(
        pool: &PgPool,
        page: u32,
        per_page: u32,
    ) -> Result<(Vec<User>, i64), sqlx::Error> {
        let offset = ((page - 1) * per_page) as i64;
        let limit = per_page as i64;
        
        let total = sqlx::query_scalar!(
            r#"SELECT COUNT(*) as "count!" FROM users"#
        )
        .fetch_one(pool)
        .await?;
        
        let users = sqlx::query_as!(
            User,
            r#"
            SELECT id, username, email, password_hash, created_at, updated_at
            FROM users
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
            "#,
            limit,
            offset
        )
        .fetch_all(pool)
        .await?;
        
        Ok((users, total))
    }
}

// Handlers
async fn register(
    pool: web::Data<PgPool>,
    req: web::Json<CreateUserRequest>,
) -> Result<HttpResponse, ApiError> {
    req.validate()
        .map_err(|e| ApiError::BadRequest(e.to_string()))?;
    
    // Check if user already exists
    if let Some(_) = UserRepository::find_by_email(&pool, &req.email).await
        .map_err(|e| ApiError::Internal(e.to_string()))?
    {
        return Err(ApiError::Conflict("Email already registered".to_string()));
    }
    
    let user = UserRepository::create(&pool, req.into_inner()).await
        .map_err(|e| ApiError::Internal(e.to_string()))?;
    
    let token = create_jwt(user.id)?;
    
    Ok(HttpResponse::Created().json(LoginResponse {
        token,
        user: UserResponse {
            id: user.id,
            username: user.username,
            email: user.email,
            created_at: user.created_at,
        },
    }))
}

async fn login(
    pool: web::Data<PgPool>,
    req: web::Json<LoginRequest>,
) -> Result<HttpResponse, ApiError> {
    req.validate()
        .map_err(|e| ApiError::BadRequest(e.to_string()))?;
    
    let user = UserRepository::find_by_email(&pool, &req.email).await
        .map_err(|e| ApiError::Internal(e.to_string()))?
        .ok_or(ApiError::Unauthorized)?;
    
    let valid = bcrypt::verify(&req.password, &user.password_hash)
        .map_err(|e| ApiError::Internal(e.to_string()))?;
    
    if !valid {
        return Err(ApiError::Unauthorized);
    }
    
    let token = create_jwt(user.id)?;
    
    Ok(HttpResponse::Ok().json(LoginResponse {
        token,
        user: UserResponse {
            id: user.id,
            username: user.username,
            email: user.email,
            created_at: user.created_at,
        },
    }))
}

async fn get_current_user(
    pool: web::Data<PgPool>,
    req: actix_web::HttpRequest,
) -> Result<HttpResponse, ApiError> {
    let user_id = req.extensions()
        .get::<Uuid>()
        .copied()
        .ok_or(ApiError::Unauthorized)?;
    
    let user = UserRepository::find_by_id(&pool, user_id).await
        .map_err(|e| ApiError::Internal(e.to_string()))?
        .ok_or(ApiError::NotFound)?;
    
    Ok(HttpResponse::Ok().json(UserResponse {
        id: user.id,
        username: user.username,
        email: user.email,
        created_at: user.created_at,
    }))
}

async fn list_users(
    pool: web::Data<PgPool>,
    query: web::Query<PaginationParams>,
) -> Result<HttpResponse, ApiError> {
    let page = query.page.unwrap_or(1).max(1);
    let per_page = query.per_page.unwrap_or(20).min(100);
    
    let (users, total) = UserRepository::list(&pool, page, per_page).await
        .map_err(|e| ApiError::Internal(e.to_string()))?;
    
    let total_pages = ((total as f64) / (per_page as f64)).ceil() as u32;
    
    let user_responses: Vec<UserResponse> = users
        .into_iter()
        .map(|u| UserResponse {
            id: u.id,
            username: u.username,
            email: u.email,
            created_at: u.created_at,
        })
        .collect();
    
    Ok(HttpResponse::Ok().json(PaginatedResponse {
        data: user_responses,
        total,
        page,
        per_page,
        total_pages,
    }))
}

// Application configuration
fn configure_app(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api/v1")
            .service(
                web::scope("/auth")
                    .route("/register", web::post().to(register))
                    .route("/login", web::post().to(login))
            )
            .service(
                web::scope("/users")
                    .wrap(HttpAuthentication::bearer(validator))
                    .route("/me", web::get().to(get_current_user))
                    .route("", web::get().to(list_users))
            )
    );
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    dotenv::dotenv().ok();
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));
    
    let database_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL must be set");
    
    let pool = PgPool::connect(&database_url)
        .await
        .expect("Failed to create pool");
    
    log::info!("Starting server on http://localhost:8080");
    
    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(pool.clone()))
            .wrap(middleware::Logger::default())
            .wrap(middleware::NormalizePath::trim())
            .configure(configure_app)
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
```

## Axum REST API

```rust
// src/main.rs - Axum implementation
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use sqlx::postgres::PgPool;
use std::sync::Arc;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use validator::Validate;

// Shared state
#[derive(Clone)]
struct AppState {
    pool: PgPool,
    jwt_secret: String,
}

// Error handling
#[derive(Debug)]
enum AppError {
    BadRequest(String),
    Unauthorized,
    NotFound,
    Conflict(String),
    Internal(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            AppError::Unauthorized => (StatusCode::UNAUTHORIZED, "Unauthorized".to_string()),
            AppError::NotFound => (StatusCode::NOT_FOUND, "Not found".to_string()),
            AppError::Conflict(msg) => (StatusCode::CONFLICT, msg),
            AppError::Internal(msg) => {
                tracing::error!("Internal error: {}", msg);
                (StatusCode::INTERNAL_SERVER_ERROR, "Internal error".to_string())
            }
        };
        
        let body = Json(serde_json::json!({
            "error": error_message,
        }));
        
        (status, body).into_response()
    }
}

// Convert common errors to AppError
impl From<sqlx::Error> for AppError {
    fn from(err: sqlx::Error) -> Self {
        match err {
            sqlx::Error::RowNotFound => AppError::NotFound,
            _ => AppError::Internal(err.to_string()),
        }
    }
}

impl From<jsonwebtoken::errors::Error> for AppError {
    fn from(err: jsonwebtoken::errors::Error) -> Self {
        AppError::Internal(err.to_string())
    }
}

// JWT middleware
use axum::{
    async_trait,
    extract::{FromRequestParts, TypedHeader},
    headers::{authorization::Bearer, Authorization},
    http::request::Parts,
    RequestPartsExt,
};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};

#[derive(Debug, Serialize, Deserialize)]
struct Claims {
    sub: Uuid,
    exp: usize,
    iat: usize,
}

struct AuthUser {
    user_id: Uuid,
}

#[async_trait]
impl<S> FromRequestParts<S> for AuthUser
where
    S: Send + Sync,
    Arc<AppState>: FromRequestParts<S>,
{
    type Rejection = AppError;
    
    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let TypedHeader(Authorization(bearer)) = parts
            .extract::<TypedHeader<Authorization<Bearer>>>()
            .await
            .map_err(|_| AppError::Unauthorized)?;
        
        let State(app_state) = parts
            .extract::<State<Arc<AppState>>>()
            .await
            .map_err(|_| AppError::Internal("Failed to extract state".to_string()))?;
        
        let token_data = decode::<Claims>(
            bearer.token(),
            &DecodingKey::from_secret(app_state.jwt_secret.as_ref()),
            &Validation::default(),
        )
        .map_err(|_| AppError::Unauthorized)?;
        
        Ok(AuthUser {
            user_id: token_data.claims.sub,
        })
    }
}

// Handlers
async fn register(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateUserRequest>,
) -> Result<impl IntoResponse, AppError> {
    req.validate()
        .map_err(|e| AppError::BadRequest(e.to_string()))?;
    
    // Check existing user
    let existing = sqlx::query!(
        "SELECT id FROM users WHERE email = $1",
        req.email
    )
    .fetch_optional(&state.pool)
    .await?;
    
    if existing.is_some() {
        return Err(AppError::Conflict("Email already registered".to_string()));
    }
    
    let password_hash = bcrypt::hash(&req.password, 12)
        .map_err(|e| AppError::Internal(e.to_string()))?;
    
    let user = sqlx::query_as!(
        User,
        r#"
        INSERT INTO users (username, email, password_hash)
        VALUES ($1, $2, $3)
        RETURNING id, username, email, password_hash, created_at, updated_at
        "#,
        req.username,
        req.email,
        password_hash
    )
    .fetch_one(&state.pool)
    .await?;
    
    let token = create_jwt(user.id, &state.jwt_secret)?;
    
    Ok((
        StatusCode::CREATED,
        Json(LoginResponse {
            token,
            user: UserResponse {
                id: user.id,
                username: user.username,
                email: user.email,
                created_at: user.created_at,
            },
        }),
    ))
}

async fn login(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LoginRequest>,
) -> Result<Json<LoginResponse>, AppError> {
    req.validate()
        .map_err(|e| AppError::BadRequest(e.to_string()))?;
    
    let user = sqlx::query_as!(
        User,
        r#"
        SELECT id, username, email, password_hash, created_at, updated_at
        FROM users
        WHERE email = $1
        "#,
        req.email
    )
    .fetch_optional(&state.pool)
    .await?
    .ok_or(AppError::Unauthorized)?;
    
    let valid = bcrypt::verify(&req.password, &user.password_hash)
        .map_err(|e| AppError::Internal(e.to_string()))?;
    
    if !valid {
        return Err(AppError::Unauthorized);
    }
    
    let token = create_jwt(user.id, &state.jwt_secret)?;
    
    Ok(Json(LoginResponse {
        token,
        user: UserResponse {
            id: user.id,
            username: user.username,
            email: user.email,
            created_at: user.created_at,
        },
    }))
}

async fn get_current_user(
    auth_user: AuthUser,
    State(state): State<Arc<AppState>>,
) -> Result<Json<UserResponse>, AppError> {
    let user = sqlx::query_as!(
        User,
        r#"
        SELECT id, username, email, password_hash, created_at, updated_at
        FROM users
        WHERE id = $1
        "#,
        auth_user.user_id
    )
    .fetch_optional(&state.pool)
    .await?
    .ok_or(AppError::NotFound)?;
    
    Ok(Json(UserResponse {
        id: user.id,
        username: user.username,
        email: user.email,
        created_at: user.created_at,
    }))
}

async fn list_users(
    _auth_user: AuthUser,
    State(state): State<Arc<AppState>>,
    Query(params): Query<PaginationParams>,
) -> Result<Json<PaginatedResponse<UserResponse>>, AppError> {
    let page = params.page.unwrap_or(1).max(1);
    let per_page = params.per_page.unwrap_or(20).min(100);
    let offset = ((page - 1) * per_page) as i64;
    
    let total = sqlx::query_scalar!(
        r#"SELECT COUNT(*) as "count!" FROM users"#
    )
    .fetch_one(&state.pool)
    .await?;
    
    let users = sqlx::query_as!(
        User,
        r#"
        SELECT id, username, email, password_hash, created_at, updated_at
        FROM users
        ORDER BY created_at DESC
        LIMIT $1 OFFSET $2
        "#,
        per_page as i64,
        offset
    )
    .fetch_all(&state.pool)
    .await?;
    
    let total_pages = ((total as f64) / (per_page as f64)).ceil() as u32;
    
    let user_responses: Vec<UserResponse> = users
        .into_iter()
        .map(|u| UserResponse {
            id: u.id,
            username: u.username,
            email: u.email,
            created_at: u.created_at,
        })
        .collect();
    
    Ok(Json(PaginatedResponse {
        data: user_responses,
        total,
        page,
        per_page,
        total_pages,
    }))
}

fn create_jwt(user_id: Uuid, secret: &str) -> Result<String, AppError> {
    let now = chrono::Utc::now();
    let exp = (now + chrono::Duration::hours(24)).timestamp() as usize;
    let iat = now.timestamp() as usize;
    
    let claims = Claims {
        sub: user_id,
        exp,
        iat,
    };
    
    encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(secret.as_ref()),
    )
    .map_err(|e| AppError::Internal(format!("Failed to create token: {}", e)))
}

// Application setup
async fn create_app(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/api/v1/auth/register", post(register))
        .route("/api/v1/auth/login", post(login))
        .route("/api/v1/users/me", get(get_current_user))
        .route("/api/v1/users", get(list_users))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

#[tokio::main]
async fn main() {
    dotenv::dotenv().ok();
    
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    
    let database_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL must be set");
    let jwt_secret = std::env::var("JWT_SECRET")
        .expect("JWT_SECRET must be set");
    
    let pool = PgPool::connect(&database_url)
        .await
        .expect("Failed to create pool");
    
    let state = Arc::new(AppState {
        pool,
        jwt_secret,
    });
    
    let app = create_app(state).await;
    
    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080")
        .await
        .unwrap();
    
    tracing::info!("Starting server on http://localhost:8080");
    
    axum::serve(listener, app)
        .await
        .unwrap();
}
```

## Advanced API Features

```rust
// src/middleware.rs - Rate limiting and request logging
use axum::{
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::Response,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tower::ServiceBuilder;
use tower_http::timeout::TimeoutLayer;

// Rate limiter
#[derive(Clone)]
struct RateLimiter {
    requests: Arc<Mutex<HashMap<String, Vec<Instant>>>>,
    max_requests: usize,
    window: Duration,
}

impl RateLimiter {
    fn new(max_requests: usize, window: Duration) -> Self {
        Self {
            requests: Arc::new(Mutex::new(HashMap::new())),
            max_requests,
            window,
        }
    }
    
    async fn check_rate_limit(&self, key: String) -> bool {
        let mut requests = self.requests.lock().await;
        let now = Instant::now();
        
        let timestamps = requests.entry(key).or_insert_with(Vec::new);
        timestamps.retain(|&t| now.duration_since(t) < self.window);
        
        if timestamps.len() < self.max_requests {
            timestamps.push(now);
            true
        } else {
            false
        }
    }
}

pub async fn rate_limit_middleware<B>(
    State(limiter): State<RateLimiter>,
    request: Request<B>,
    next: Next<B>,
) -> Result<Response, StatusCode> {
    let key = request
        .headers()
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("127.0.0.1")
        .to_string();
    
    if !limiter.check_rate_limit(key).await {
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }
    
    Ok(next.run(request).await)
}

// Request ID middleware
pub async fn request_id_middleware<B>(
    mut request: Request<B>,
    next: Next<B>,
) -> Response {
    let request_id = Uuid::new_v4().to_string();
    request.headers_mut().insert(
        "x-request-id",
        request_id.parse().unwrap(),
    );
    
    let mut response = next.run(request).await;
    response.headers_mut().insert(
        "x-request-id",
        request_id.parse().unwrap(),
    );
    
    response
}

// API versioning
pub fn api_v1_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/auth/register", post(register))
        .route("/auth/login", post(login))
        .route("/users/me", get(get_current_user))
        .route("/users", get(list_users))
        .route("/users/:id", get(get_user_by_id))
        .route("/posts", post(create_post).get(list_posts))
        .route("/posts/:id", get(get_post).put(update_post).delete(delete_post))
}

pub fn api_v2_routes() -> Router<Arc<AppState>> {
    Router::new()
        // V2 routes with breaking changes
        .route("/auth/signup", post(register_v2))
        .route("/auth/signin", post(login_v2))
}

// File upload handling
use axum::extract::Multipart;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

async fn upload_file(
    auth_user: AuthUser,
    mut multipart: Multipart,
) -> Result<Json<FileUploadResponse>, AppError> {
    let upload_dir = "uploads";
    tokio::fs::create_dir_all(upload_dir).await
        .map_err(|e| AppError::Internal(e.to_string()))?;
    
    let mut files = Vec::new();
    
    while let Some(field) = multipart.next_field().await
        .map_err(|e| AppError::BadRequest(e.to_string()))?
    {
        let name = field.name().unwrap_or("unknown").to_string();
        let file_name = field.file_name()
            .ok_or_else(|| AppError::BadRequest("No filename".to_string()))?
            .to_string();
        
        let content_type = field.content_type()
            .unwrap_or("application/octet-stream")
            .to_string();
        
        let data = field.bytes().await
            .map_err(|e| AppError::BadRequest(e.to_string()))?;
        
        let file_id = Uuid::new_v4();
        let file_path = format!("{}/{}", upload_dir, file_id);
        
        let mut file = File::create(&file_path).await
            .map_err(|e| AppError::Internal(e.to_string()))?;
        
        file.write_all(&data).await
            .map_err(|e| AppError::Internal(e.to_string()))?;
        
        files.push(FileInfo {
            id: file_id,
            name: file_name,
            content_type,
            size: data.len() as i64,
        });
    }
    
    Ok(Json(FileUploadResponse { files }))
}

// WebSocket support
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use futures::{sink::SinkExt, stream::StreamExt};

async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| websocket_connection(socket, state))
}

async fn websocket_connection(socket: WebSocket, state: Arc<AppState>) {
    let (mut sender, mut receiver) = socket.split();
    
    // Send welcome message
    let _ = sender
        .send(Message::Text("Welcome to the WebSocket!".to_string()))
        .await;
    
    // Echo messages back
    while let Some(msg) = receiver.next().await {
        if let Ok(msg) = msg {
            match msg {
                Message::Text(text) => {
                    let response = format!("Echo: {}", text);
                    let _ = sender.send(Message::Text(response)).await;
                }
                Message::Close(_) => break,
                _ => {}
            }
        }
    }
}

// OpenAPI documentation
use utoipa::{OpenApi, ToSchema};
use utoipa_swagger_ui::SwaggerUi;

#[derive(OpenApi)]
#[openapi(
    paths(
        register,
        login,
        get_current_user,
        list_users,
    ),
    components(
        schemas(CreateUserRequest, LoginRequest, UserResponse, LoginResponse)
    ),
    tags(
        (name = "auth", description = "Authentication endpoints"),
        (name = "users", description = "User management endpoints")
    )
)]
struct ApiDoc;

// Health check and metrics
#[derive(Serialize)]
struct HealthResponse {
    status: String,
    database: String,
    version: String,
}

async fn health_check(
    State(state): State<Arc<AppState>>,
) -> Result<Json<HealthResponse>, AppError> {
    // Check database connection
    let db_status = sqlx::query!("SELECT 1 as check")
        .fetch_one(&state.pool)
        .await
        .map(|_| "healthy")
        .unwrap_or("unhealthy");
    
    Ok(Json(HealthResponse {
        status: "ok".to_string(),
        database: db_status.to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    }))
}

// Complete application setup
pub async fn create_complete_app(state: Arc<AppState>) -> Router {
    let rate_limiter = RateLimiter::new(100, Duration::from_secs(60));
    
    Router::new()
        .nest("/api/v1", api_v1_routes())
        .nest("/api/v2", api_v2_routes())
        .route("/health", get(health_check))
        .route("/ws", get(websocket_handler))
        .route("/upload", post(upload_file))
        .merge(SwaggerUi::new("/swagger-ui").url("/api-doc/openapi.json", ApiDoc::openapi()))
        .layer(
            ServiceBuilder::new()
                .layer(TimeoutLayer::new(Duration::from_secs(30)))
                .layer(axum::middleware::from_fn_with_state(
                    rate_limiter,
                    rate_limit_middleware,
                ))
                .layer(axum::middleware::from_fn(request_id_middleware))
                .layer(CorsLayer::permissive())
                .layer(TraceLayer::new_for_http()),
        )
        .with_state(state)
}
```

## Testing REST APIs

```rust
// tests/api_tests.rs
use axum::http::StatusCode;
use sqlx::PgPool;
use tower::ServiceExt;

#[tokio::test]
async fn test_user_registration() {
    let app = test_app().await;
    
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/auth/register")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_string(&CreateUserRequest {
                        username: "testuser".to_string(),
                        email: "test@example.com".to_string(),
                        password: "password123".to_string(),
                    })
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    
    assert_eq!(response.status(), StatusCode::CREATED);
    
    let body = hyper::body::to_bytes(response.into_body()).await.unwrap();
    let login_response: LoginResponse = serde_json::from_slice(&body).unwrap();
    
    assert!(!login_response.token.is_empty());
    assert_eq!(login_response.user.username, "testuser");
}

#[tokio::test]
async fn test_authentication_required() {
    let app = test_app().await;
    
    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/v1/users/me")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

async fn test_app() -> Router {
    let database_url = "postgres://test:test@localhost/test_db";
    let pool = PgPool::connect(database_url).await.unwrap();
    
    // Run migrations
    sqlx::migrate!("./migrations")
        .run(&pool)
        .await
        .unwrap();
    
    let state = Arc::new(AppState {
        pool,
        jwt_secret: "test_secret".to_string(),
    });
    
    create_app(state).await
}
```

## Exercises

1. **GraphQL API**: Implement a GraphQL endpoint using async-graphql that integrates with the existing database models.

2. **API Gateway**: Build an API gateway that routes requests to different microservices based on the path.

3. **Real-time Updates**: Implement Server-Sent Events (SSE) or WebSocket broadcasting for real-time notifications.

4. **OAuth Integration**: Add OAuth2 authentication with providers like Google or GitHub.

5. **API Client SDK**: Generate a Rust client SDK from OpenAPI specifications for consuming the API.

## Key Takeaways

- Actix-web provides a powerful actor-based framework
- Axum offers a more functional approach with excellent type safety
- Use extractors for parsing request data
- Implement proper error handling with custom error types
- JWT tokens for stateless authentication
- Middleware for cross-cutting concerns
- Rate limiting protects against abuse
- API versioning maintains backward compatibility
- WebSocket support enables real-time features
- Always validate input data
- Use connection pooling for database efficiency

## Next Steps

In Tutorial 15, we'll explore **Cloud Integration**, learning how to deploy Rust applications to Google Cloud and Azure, integrate with cloud services, and build cloud-native applications.