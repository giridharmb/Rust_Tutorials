# Tutorial 15: Cloud Integration - Google Cloud & Azure

## Setting Up Cloud SDKs

This tutorial covers integrating Rust applications with Google Cloud Platform (GCP) and Microsoft Azure.

```toml
# Cargo.toml for Google Cloud
[dependencies]
# Google Cloud
google-cloud-storage = "0.15"
google-cloud-pubsub = "0.15"
google-cloud-bigquery = "0.15"
google-cloud-googleapis = "0.10"
google-cloud-auth = "0.13"

# Azure
azure_core = "0.17"
azure_storage = "0.17"
azure_storage_blobs = "0.17"
azure_identity = "0.17"
azure_messaging_servicebus = "0.17"

# Common dependencies
tokio = { version = "1", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
chrono = { version = "0.4", features = ["serde"] }
uuid = { version = "1.0", features = ["v4", "serde"] }
anyhow = "1.0"
tracing = "0.1"
tracing-subscriber = "0.3"
bytes = "1.5"
futures = "0.3"
```

## Google Cloud Storage

```rust
// src/gcs.rs - Google Cloud Storage operations
use google_cloud_storage::{
    client::{Client, ClientConfig},
    http::objects::{
        download::Range,
        get::GetObjectRequest,
        upload::{Media, UploadObjectRequest, UploadType},
        Object,
    },
};
use google_cloud_auth::credentials::CredentialsFile;
use bytes::Bytes;
use std::path::Path;
use tokio::fs::File;
use tokio::io::AsyncReadExt;
use anyhow::{Result, Context};

pub struct CloudStorageService {
    client: Client,
    bucket: String,
}

impl CloudStorageService {
    pub async fn new(bucket: String, credentials_path: Option<&str>) -> Result<Self> {
        let config = if let Some(path) = credentials_path {
            let credentials = CredentialsFile::new_from_file(path)
                .await
                .context("Failed to load credentials")?;
            
            ClientConfig::default()
                .with_credentials(credentials)
                .await?
        } else {
            // Use default credentials (e.g., from environment)
            ClientConfig::default().with_auth().await?
        };
        
        let client = Client::new(config);
        
        Ok(Self { client, bucket })
    }
    
    pub async fn upload_file(
        &self,
        local_path: &Path,
        gcs_path: &str,
        content_type: Option<&str>,
    ) -> Result<Object> {
        let mut file = File::open(local_path)
            .await
            .context("Failed to open local file")?;
        
        let mut contents = Vec::new();
        file.read_to_end(&mut contents)
            .await
            .context("Failed to read file")?;
        
        self.upload_bytes(
            Bytes::from(contents),
            gcs_path,
            content_type.unwrap_or("application/octet-stream"),
        )
        .await
    }
    
    pub async fn upload_bytes(
        &self,
        data: Bytes,
        gcs_path: &str,
        content_type: &str,
    ) -> Result<Object> {
        let upload_type = UploadType::Simple(Media::new(gcs_path));
        
        let mut request = UploadObjectRequest {
            bucket: self.bucket.clone(),
            ..Default::default()
        };
        
        let uploaded = self.client
            .upload_object(&request, data, &upload_type)
            .await
            .context("Failed to upload to GCS")?;
        
        Ok(uploaded)
    }
    
    pub async fn download_file(
        &self,
        gcs_path: &str,
        local_path: &Path,
    ) -> Result<()> {
        let data = self.download_bytes(gcs_path).await?;
        
        tokio::fs::write(local_path, &data)
            .await
            .context("Failed to write downloaded file")?;
        
        Ok(())
    }
    
    pub async fn download_bytes(&self, gcs_path: &str) -> Result<Bytes> {
        let request = GetObjectRequest {
            bucket: self.bucket.clone(),
            object: gcs_path.to_string(),
            ..Default::default()
        };
        
        let result = self.client
            .download_object(&request, &Range::default())
            .await
            .context("Failed to download from GCS")?;
        
        Ok(result)
    }
    
    pub async fn list_objects(&self, prefix: Option<&str>) -> Result<Vec<Object>> {
        let mut request = google_cloud_storage::http::objects::list::ListObjectsRequest {
            bucket: self.bucket.clone(),
            ..Default::default()
        };
        
        if let Some(prefix) = prefix {
            request.prefix = Some(prefix.to_string());
        }
        
        let mut objects = Vec::new();
        let mut page_token = None;
        
        loop {
            request.page_token = page_token.clone();
            
            let response = self.client
                .list_objects(&request)
                .await
                .context("Failed to list objects")?;
            
            if let Some(items) = response.items {
                objects.extend(items);
            }
            
            page_token = response.next_page_token;
            if page_token.is_none() {
                break;
            }
        }
        
        Ok(objects)
    }
    
    pub async fn delete_object(&self, gcs_path: &str) -> Result<()> {
        let request = google_cloud_storage::http::objects::delete::DeleteObjectRequest {
            bucket: self.bucket.clone(),
            object: gcs_path.to_string(),
            ..Default::default()
        };
        
        self.client
            .delete_object(&request)
            .await
            .context("Failed to delete object")?;
        
        Ok(())
    }
    
    pub async fn generate_signed_url(
        &self,
        gcs_path: &str,
        expiration_minutes: i64,
    ) -> Result<String> {
        use google_cloud_storage::sign::SignedURLOptions;
        use google_cloud_storage::sign::SignedURLMethod;
        use chrono::{Duration, Utc};
        
        let options = SignedURLOptions {
            method: SignedURLMethod::GET,
            expires: Utc::now() + Duration::minutes(expiration_minutes),
            ..Default::default()
        };
        
        let url = self.client
            .signed_url(&self.bucket, gcs_path, None, None, options)
            .await
            .context("Failed to generate signed URL")?;
        
        Ok(url)
    }
}

// Example usage
#[tokio::main]
async fn gcs_example() -> Result<()> {
    let storage = CloudStorageService::new(
        "my-bucket".to_string(),
        Some("path/to/credentials.json"),
    )
    .await?;
    
    // Upload a file
    let local_file = Path::new("test.txt");
    let uploaded = storage
        .upload_file(local_file, "uploads/test.txt", Some("text/plain"))
        .await?;
    
    println!("Uploaded: {:?}", uploaded);
    
    // List objects
    let objects = storage.list_objects(Some("uploads/")).await?;
    for obj in objects {
        println!("Object: {}", obj.name);
    }
    
    Ok(())
}
```

## Google Cloud Pub/Sub

```rust
// src/pubsub.rs - Google Cloud Pub/Sub messaging
use google_cloud_pubsub::{
    client::{Client, ClientConfig},
    publisher::Publisher,
    subscription::{Subscription, SubscriptionConfig},
};
use google_cloud_googleapis::pubsub::v1::PubsubMessage;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use anyhow::{Result, Context};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventMessage<T> {
    pub id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub event_type: String,
    pub data: T,
}

pub struct PubSubService {
    client: Client,
    project_id: String,
}

impl PubSubService {
    pub async fn new(project_id: String, credentials_path: Option<&str>) -> Result<Self> {
        let config = if let Some(path) = credentials_path {
            ClientConfig::default()
                .with_credentials_file(path)
                .await?
        } else {
            ClientConfig::default().with_auth().await?
        };
        
        let client = Client::new(config).await?;
        
        Ok(Self { client, project_id })
    }
    
    pub async fn create_topic(&self, topic_name: &str) -> Result<()> {
        let topic = self.client.topic(topic_name);
        
        if !topic.exists(None).await? {
            topic.create(None, None).await?;
            println!("Created topic: {}", topic_name);
        }
        
        Ok(())
    }
    
    pub async fn create_subscription(
        &self,
        subscription_name: &str,
        topic_name: &str,
    ) -> Result<()> {
        let topic = self.client.topic(topic_name);
        let subscription = self.client.subscription(subscription_name);
        
        if !subscription.exists(None).await? {
            let config = SubscriptionConfig {
                topic: topic.fully_qualified_name(),
                ack_deadline_seconds: 30,
                ..Default::default()
            };
            
            subscription.create(config, None, None).await?;
            println!("Created subscription: {}", subscription_name);
        }
        
        Ok(())
    }
    
    pub async fn publish<T: Serialize>(
        &self,
        topic_name: &str,
        event: EventMessage<T>,
    ) -> Result<String> {
        let topic = self.client.topic(topic_name);
        let publisher = topic.new_publisher(None);
        
        let json = serde_json::to_string(&event)?;
        
        let msg = PubsubMessage {
            data: json.into_bytes(),
            attributes: std::collections::HashMap::from([
                ("event_type".to_string(), event.event_type.clone()),
                ("timestamp".to_string(), event.timestamp.to_rfc3339()),
            ]),
            ..Default::default()
        };
        
        let awaiter = publisher.publish(msg).await;
        let message_id = awaiter.get().await?;
        
        Ok(message_id)
    }
    
    pub async fn subscribe<T, F>(
        &self,
        subscription_name: &str,
        handler: F,
    ) -> Result<()>
    where
        T: for<'de> Deserialize<'de> + Send + 'static,
        F: Fn(EventMessage<T>) -> futures::future::BoxFuture<'static, Result<()>> + Send + Sync + 'static,
    {
        let subscription = self.client.subscription(subscription_name);
        let handler = Arc::new(handler);
        
        let mut stream = subscription
            .subscribe(None)
            .await
            .context("Failed to create subscription stream")?;
        
        while let Some(message) = stream.recv().await {
            let handler = Arc::clone(&handler);
            
            tokio::spawn(async move {
                let data = String::from_utf8_lossy(&message.message.data);
                
                match serde_json::from_str::<EventMessage<T>>(&data) {
                    Ok(event) => {
                        match handler(event).await {
                            Ok(_) => {
                                let _ = message.ack().await;
                            }
                            Err(e) => {
                                eprintln!("Handler error: {}", e);
                                let _ = message.nack().await;
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to deserialize message: {}", e);
                        let _ = message.ack().await; // Ack to avoid redelivery
                    }
                }
            });
        }
        
        Ok(())
    }
}

// Example usage
#[derive(Debug, Serialize, Deserialize)]
struct UserEvent {
    user_id: String,
    action: String,
}

async fn pubsub_example() -> Result<()> {
    let pubsub = PubSubService::new(
        "my-project-id".to_string(),
        Some("path/to/credentials.json"),
    )
    .await?;
    
    // Create topic and subscription
    pubsub.create_topic("user-events").await?;
    pubsub.create_subscription("user-events-sub", "user-events").await?;
    
    // Publish an event
    let event = EventMessage {
        id: uuid::Uuid::new_v4().to_string(),
        timestamp: chrono::Utc::now(),
        event_type: "user.created".to_string(),
        data: UserEvent {
            user_id: "123".to_string(),
            action: "signup".to_string(),
        },
    };
    
    let message_id = pubsub.publish("user-events", event).await?;
    println!("Published message: {}", message_id);
    
    // Subscribe to events
    pubsub.subscribe::<UserEvent, _>("user-events-sub", |event| {
        Box::pin(async move {
            println!("Received event: {:?}", event);
            Ok(())
        })
    }).await?;
    
    Ok(())
}
```

## Azure Blob Storage

```rust
// src/azure_storage.rs - Azure Blob Storage operations
use azure_storage::prelude::*;
use azure_storage_blobs::prelude::*;
use azure_core::prelude::*;
use azure_identity::DefaultAzureCredential;
use bytes::Bytes;
use futures::stream::StreamExt;
use std::sync::Arc;
use anyhow::{Result, Context};

pub struct AzureStorageService {
    container_client: ContainerClient,
}

impl AzureStorageService {
    pub async fn new(
        account_name: &str,
        container_name: &str,
        credential: Option<Arc<DefaultAzureCredential>>,
    ) -> Result<Self> {
        let storage_credentials = if let Some(cred) = credential {
            StorageCredentials::TokenCredential(cred)
        } else {
            // Use connection string from environment
            let connection_string = std::env::var("AZURE_STORAGE_CONNECTION_STRING")
                .context("AZURE_STORAGE_CONNECTION_STRING not set")?;
            
            StorageCredentials::ConnectionString(connection_string)
        };
        
        let blob_service = BlobServiceClient::new(account_name, storage_credentials);
        let container_client = blob_service.container_client(container_name);
        
        Ok(Self { container_client })
    }
    
    pub async fn create_container_if_not_exists(&self) -> Result<()> {
        match self.container_client
            .create()
            .public_access(PublicAccess::None)
            .await
        {
            Ok(_) => println!("Container created"),
            Err(e) => {
                if !e.to_string().contains("ContainerAlreadyExists") {
                    return Err(e.into());
                }
            }
        }
        
        Ok(())
    }
    
    pub async fn upload_blob(
        &self,
        blob_name: &str,
        data: Bytes,
        content_type: Option<&str>,
    ) -> Result<()> {
        let blob_client = self.container_client.blob_client(blob_name);
        
        let mut builder = blob_client
            .put_block_blob(data)
            .content_type(content_type.unwrap_or("application/octet-stream"));
        
        builder.await
            .context("Failed to upload blob")?;
        
        Ok(())
    }
    
    pub async fn download_blob(&self, blob_name: &str) -> Result<Bytes> {
        let blob_client = self.container_client.blob_client(blob_name);
        
        let mut stream = blob_client
            .get()
            .into_stream();
        
        let mut data = Vec::new();
        
        while let Some(result) = stream.next().await {
            let response = result.context("Failed to download blob chunk")?;
            data.extend_from_slice(&response.data);
        }
        
        Ok(Bytes::from(data))
    }
    
    pub async fn list_blobs(&self, prefix: Option<&str>) -> Result<Vec<String>> {
        let mut builder = self.container_client.list_blobs();
        
        if let Some(prefix) = prefix {
            builder = builder.prefix(prefix);
        }
        
        let mut blob_names = Vec::new();
        let mut stream = builder.into_stream();
        
        while let Some(result) = stream.next().await {
            let response = result.context("Failed to list blobs")?;
            
            for blob in response.blobs.items {
                blob_names.push(blob.name);
            }
        }
        
        Ok(blob_names)
    }
    
    pub async fn delete_blob(&self, blob_name: &str) -> Result<()> {
        let blob_client = self.container_client.blob_client(blob_name);
        
        blob_client
            .delete()
            .await
            .context("Failed to delete blob")?;
        
        Ok(())
    }
    
    pub async fn generate_sas_url(
        &self,
        blob_name: &str,
        expiry_hours: i64,
    ) -> Result<String> {
        use azure_storage::shared_access_signature::{
            BlobSasPermissions, BlobSharedAccessSignature,
        };
        use chrono::{Duration, Utc};
        
        let blob_client = self.container_client.blob_client(blob_name);
        
        let sas = BlobSharedAccessSignature::new(
            blob_client.account_name().clone(),
            blob_client.container_name().clone(),
            blob_name.to_string(),
            BlobSasPermissions {
                read: true,
                ..Default::default()
            },
            Utc::now() + Duration::hours(expiry_hours),
        );
        
        let url = blob_client.url()?.join(&format!("?{}", sas.token()))?;
        
        Ok(url.to_string())
    }
}

// Example usage
async fn azure_storage_example() -> Result<()> {
    let storage = AzureStorageService::new(
        "mystorageaccount",
        "mycontainer",
        None, // Use connection string from env
    )
    .await?;
    
    // Create container
    storage.create_container_if_not_exists().await?;
    
    // Upload data
    let data = Bytes::from("Hello from Azure!");
    storage.upload_blob("test.txt", data, Some("text/plain")).await?;
    
    // List blobs
    let blobs = storage.list_blobs(None).await?;
    for blob in blobs {
        println!("Blob: {}", blob);
    }
    
    Ok(())
}
```

## Azure Service Bus

```rust
// src/azure_servicebus.rs - Azure Service Bus messaging
use azure_messaging_servicebus::{
    prelude::*,
    ServiceBusClient, ServiceBusMessage,
};
use azure_identity::DefaultAzureCredential;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use anyhow::{Result, Context};

#[derive(Debug, Serialize, Deserialize)]
pub struct QueueMessage<T> {
    pub id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub data: T,
    pub correlation_id: Option<String>,
}

pub struct ServiceBusService {
    client: ServiceBusClient,
    namespace: String,
}

impl ServiceBusService {
    pub async fn new(
        namespace: &str,
        credential: Option<Arc<DefaultAzureCredential>>,
    ) -> Result<Self> {
        let client = if let Some(cred) = credential {
            ServiceBusClient::new(namespace, cred)
        } else {
            // Use connection string
            let connection_string = std::env::var("AZURE_SERVICEBUS_CONNECTION_STRING")
                .context("AZURE_SERVICEBUS_CONNECTION_STRING not set")?;
            
            ServiceBusClient::new_from_connection_string(connection_string)?
        };
        
        Ok(Self {
            client,
            namespace: namespace.to_string(),
        })
    }
    
    pub async fn send_to_queue<T: Serialize>(
        &self,
        queue_name: &str,
        message: QueueMessage<T>,
    ) -> Result<()> {
        let sender = self.client
            .create_sender(queue_name)
            .await
            .context("Failed to create sender")?;
        
        let json = serde_json::to_string(&message)?;
        
        let mut sb_message = ServiceBusMessage::new(json);
        sb_message.message_id = Some(message.id.clone());
        
        if let Some(correlation_id) = message.correlation_id {
            sb_message.correlation_id = Some(correlation_id);
        }
        
        sender.send_message(sb_message)
            .await
            .context("Failed to send message")?;
        
        Ok(())
    }
    
    pub async fn receive_from_queue<T, F>(
        &self,
        queue_name: &str,
        handler: F,
    ) -> Result<()>
    where
        T: for<'de> Deserialize<'de>,
        F: Fn(QueueMessage<T>) -> futures::future::BoxFuture<'static, Result<()>> + Send + Sync,
    {
        let mut receiver = self.client
            .create_receiver(
                queue_name,
                ServiceBusReceiveMode::PeekLock,
            )
            .await
            .context("Failed to create receiver")?;
        
        loop {
            let messages = receiver
                .receive_messages(10) // Batch size
                .max_wait_time(std::time::Duration::from_secs(30))
                .await
                .context("Failed to receive messages")?;
            
            for message in messages {
                let body = message.body()
                    .context("Failed to get message body")?;
                
                let body_str = std::str::from_utf8(&body)
                    .context("Invalid UTF-8 in message body")?;
                
                match serde_json::from_str::<QueueMessage<T>>(body_str) {
                    Ok(queue_message) => {
                        match handler(queue_message).await {
                            Ok(_) => {
                                receiver.complete_message(&message)
                                    .await
                                    .context("Failed to complete message")?;
                            }
                            Err(e) => {
                                eprintln!("Handler error: {}", e);
                                receiver.abandon_message(&message)
                                    .await
                                    .context("Failed to abandon message")?;
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to deserialize message: {}", e);
                        // Dead letter the message
                        receiver.dead_letter_message(&message, None, None)
                            .await
                            .context("Failed to dead letter message")?;
                    }
                }
            }
        }
    }
    
    pub async fn send_to_topic<T: Serialize>(
        &self,
        topic_name: &str,
        message: QueueMessage<T>,
        properties: Option<std::collections::HashMap<String, String>>,
    ) -> Result<()> {
        let sender = self.client
            .create_sender(topic_name)
            .await
            .context("Failed to create topic sender")?;
        
        let json = serde_json::to_string(&message)?;
        
        let mut sb_message = ServiceBusMessage::new(json);
        sb_message.message_id = Some(message.id.clone());
        
        if let Some(props) = properties {
            sb_message.application_properties = props;
        }
        
        sender.send_message(sb_message)
            .await
            .context("Failed to send to topic")?;
        
        Ok(())
    }
}

// Example usage
#[derive(Debug, Serialize, Deserialize)]
struct OrderMessage {
    order_id: String,
    customer_id: String,
    amount: f64,
}

async fn servicebus_example() -> Result<()> {
    let service_bus = ServiceBusService::new(
        "mynamespace.servicebus.windows.net",
        None, // Use connection string
    )
    .await?;
    
    // Send message
    let message = QueueMessage {
        id: uuid::Uuid::new_v4().to_string(),
        timestamp: chrono::Utc::now(),
        data: OrderMessage {
            order_id: "ORDER-123".to_string(),
            customer_id: "CUST-456".to_string(),
            amount: 99.99,
        },
        correlation_id: Some("session-789".to_string()),
    };
    
    service_bus.send_to_queue("orders", message).await?;
    
    // Receive messages
    service_bus.receive_from_queue::<OrderMessage, _>("orders", |msg| {
        Box::pin(async move {
            println!("Received order: {:?}", msg.data);
            Ok(())
        })
    }).await?;
    
    Ok(())
}
```

## Cloud-Native Application Example

```rust
// src/cloud_app.rs - Complete cloud-native application
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

// Application configuration
#[derive(Clone)]
pub struct CloudConfig {
    pub gcp_project_id: String,
    pub gcp_bucket: String,
    pub azure_account: String,
    pub azure_container: String,
    pub environment: String,
}

impl CloudConfig {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            gcp_project_id: std::env::var("GCP_PROJECT_ID")?,
            gcp_bucket: std::env::var("GCP_BUCKET")?,
            azure_account: std::env::var("AZURE_STORAGE_ACCOUNT")?,
            azure_container: std::env::var("AZURE_CONTAINER")?,
            environment: std::env::var("ENVIRONMENT").unwrap_or_else(|_| "development".to_string()),
        })
    }
}

// Application state
#[derive(Clone)]
pub struct AppState {
    config: CloudConfig,
    gcs: Arc<CloudStorageService>,
    azure_storage: Arc<AzureStorageService>,
    pubsub: Arc<PubSubService>,
    cache: Arc<RwLock<std::collections::HashMap<String, CachedFile>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedFile {
    pub id: String,
    pub filename: String,
    pub cloud_provider: String,
    pub cloud_path: String,
    pub content_type: String,
    pub size: i64,
    pub uploaded_at: chrono::DateTime<chrono::Utc>,
}

// API handlers
#[derive(Debug, Deserialize)]
pub struct UploadRequest {
    pub filename: String,
    pub content_type: String,
    pub cloud_provider: String, // "gcp" or "azure"
}

#[derive(Debug, Serialize)]
pub struct UploadResponse {
    pub id: String,
    pub upload_url: String,
}

async fn create_upload(
    State(state): State<Arc<AppState>>,
    Json(req): Json<UploadRequest>,
) -> Result<Json<UploadResponse>, StatusCode> {
    let file_id = Uuid::new_v4().to_string();
    let cloud_path = format!("uploads/{}/{}", file_id, req.filename);
    
    let upload_url = match req.cloud_provider.as_str() {
        "gcp" => {
            state.gcs
                .generate_signed_url(&cloud_path, 60)
                .await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        }
        "azure" => {
            state.azure_storage
                .generate_sas_url(&cloud_path, 1)
                .await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        }
        _ => return Err(StatusCode::BAD_REQUEST),
    };
    
    // Cache file metadata
    let cached_file = CachedFile {
        id: file_id.clone(),
        filename: req.filename,
        cloud_provider: req.cloud_provider,
        cloud_path,
        content_type: req.content_type,
        size: 0, // Will be updated after upload
        uploaded_at: chrono::Utc::now(),
    };
    
    state.cache.write().await.insert(file_id.clone(), cached_file.clone());
    
    // Publish event
    let event = EventMessage {
        id: Uuid::new_v4().to_string(),
        timestamp: chrono::Utc::now(),
        event_type: "file.upload_initiated".to_string(),
        data: cached_file,
    };
    
    let _ = state.pubsub
        .publish("file-events", event)
        .await;
    
    Ok(Json(UploadResponse {
        id: file_id,
        upload_url,
    }))
}

async fn get_file(
    State(state): State<Arc<AppState>>,
    Path(file_id): Path<String>,
) -> Result<Json<CachedFile>, StatusCode> {
    let cache = state.cache.read().await;
    
    cache.get(&file_id)
        .cloned()
        .ok_or(StatusCode::NOT_FOUND)
        .map(Json)
}

async fn list_files(
    State(state): State<Arc<AppState>>,
) -> Json<Vec<CachedFile>> {
    let cache = state.cache.read().await;
    let files: Vec<CachedFile> = cache.values().cloned().collect();
    Json(files)
}

// Health check with cloud service connectivity
#[derive(Serialize)]
pub struct HealthStatus {
    pub status: String,
    pub environment: String,
    pub gcp_storage: String,
    pub azure_storage: String,
}

async fn health_check(
    State(state): State<Arc<AppState>>,
) -> Json<HealthStatus> {
    // Check GCP connectivity
    let gcp_status = match state.gcs.list_objects(Some("health/")).await {
        Ok(_) => "healthy",
        Err(_) => "unhealthy",
    };
    
    // Check Azure connectivity
    let azure_status = match state.azure_storage.list_blobs(Some("health/")).await {
        Ok(_) => "healthy",
        Err(_) => "unhealthy",
    };
    
    Json(HealthStatus {
        status: "ok".to_string(),
        environment: state.config.environment.clone(),
        gcp_storage: gcp_status.to_string(),
        azure_storage: azure_status.to_string(),
    })
}

// Application setup
pub async fn create_cloud_app() -> Result<Router> {
    let config = CloudConfig::from_env()?;
    
    // Initialize cloud services
    let gcs = Arc::new(
        CloudStorageService::new(config.gcp_bucket.clone(), None).await?
    );
    
    let azure_storage = Arc::new(
        AzureStorageService::new(
            &config.azure_account,
            &config.azure_container,
            None,
        ).await?
    );
    
    let pubsub = Arc::new(
        PubSubService::new(config.gcp_project_id.clone(), None).await?
    );
    
    // Create topics
    pubsub.create_topic("file-events").await?;
    
    let state = Arc::new(AppState {
        config,
        gcs,
        azure_storage,
        pubsub,
        cache: Arc::new(RwLock::new(std::collections::HashMap::new())),
    });
    
    // Start background event processor
    let event_state = Arc::clone(&state);
    tokio::spawn(async move {
        let _ = event_state.pubsub
            .subscribe::<CachedFile, _>("file-events-sub", |event| {
                Box::pin(async move {
                    println!("Processing file event: {:?}", event);
                    Ok(())
                })
            })
            .await;
    });
    
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/api/v1/files", post(create_upload).get(list_files))
        .route("/api/v1/files/:id", get(get_file))
        .with_state(state);
    
    Ok(app)
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::dotenv().ok();
    tracing_subscriber::fmt::init();
    
    let app = create_cloud_app().await?;
    
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080")
        .await?;
    
    tracing::info!("Cloud application starting on http://0.0.0.0:8080");
    
    axum::serve(listener, app).await?;
    
    Ok(())
}
```

## Deployment Configuration

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rust-cloud-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rust-cloud-app
  template:
    metadata:
      labels:
        app: rust-cloud-app
    spec:
      serviceAccountName: cloud-app-sa
      containers:
      - name: app
        image: gcr.io/my-project/rust-cloud-app:latest
        ports:
        - containerPort: 8080
        env:
        - name: GCP_PROJECT_ID
          value: "my-project"
        - name: GCP_BUCKET
          value: "my-bucket"
        - name: AZURE_STORAGE_ACCOUNT
          value: "mystorageaccount"
        - name: AZURE_CONTAINER
          value: "mycontainer"
        - name: ENVIRONMENT
          value: "production"
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "200m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

```dockerfile
# Dockerfile - Multi-stage build for cloud deployment
FROM rust:1.75 as builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src

# Build release binary
RUN cargo build --release

# Runtime image
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/cloud-app /app/

EXPOSE 8080

CMD ["./cloud-app"]
```

## Exercises

1. **Multi-Cloud Abstraction**: Create a trait-based abstraction layer that provides a unified interface for storage operations across GCP and Azure.

2. **Event Sourcing**: Implement an event sourcing system using cloud messaging services (Pub/Sub or Service Bus) with event replay capabilities.

3. **Serverless Functions**: Deploy Rust functions to Google Cloud Functions or Azure Functions with automatic scaling.

4. **Cloud Data Pipeline**: Build a data pipeline that ingests from cloud storage, processes data, and writes to BigQuery or Azure Data Lake.

5. **Service Mesh Integration**: Implement distributed tracing and service mesh integration for the cloud-native application.

## Key Takeaways

- Use official SDK crates for cloud provider integration
- Implement proper authentication with service accounts or managed identities
- Design for cloud-native patterns: stateless, horizontally scalable
- Use cloud messaging for asynchronous communication
- Implement health checks for container orchestration
- Handle transient failures with retry logic
- Use structured logging for cloud observability
- Leverage cloud-specific features like signed URLs
- Consider multi-cloud strategies for vendor independence
- Container images should be minimal and secure

## Conclusion

This completes our Rust tutorial series! You've learned everything from basic syntax to building cloud-native applications. Continue exploring Rust's ecosystem and building amazing systems!