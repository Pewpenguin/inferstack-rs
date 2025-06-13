use anyhow::{Context, Result};
use redis::{AsyncCommands, Client};
use serde::{de::DeserializeOwned, Serialize};
use sha2::{Digest, Sha256};
use tracing::{debug, info};

pub struct CacheService {
    client: Client,
}

impl CacheService {
    pub fn new(redis_url: &str) -> Result<Self> {
        let client = redis::Client::open(redis_url)
            .context("Failed to create Redis client")?;
        
        Ok(Self { client })
    }
    
    pub async fn get<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        let mut conn = self.client.get_multiplexed_tokio_connection().await
            .context("Failed to connect to Redis")?;
        
        let result: Option<String> = conn.get::<_, _>(key)
            .await
            .context("Failed to get value from Redis")?;

        match result {
            Some(data) => {
                let value: T = serde_json::from_str(&data)
                    .context("Failed to deserialize cached data")?;
                debug!("Cache hit for key: {}", key);
                Ok(Some(value))
            },
            None => {
                debug!("Cache miss for key: {}", key);
                Ok(None)
            }
        }
    }
    
    pub async fn set<T: Serialize>(&self, key: &str, value: &T, ttl_seconds: Option<usize>) -> Result<()> {
        let mut conn = self.client.get_multiplexed_tokio_connection().await
            .context("Failed to connect to Redis")?;
        
        let serialized = serde_json::to_string(value)
            .context("Failed to serialize value for caching")?;
        
        match ttl_seconds {
            Some(ttl) => {
                let (): () = conn.set_ex(key, serialized, ttl as u64).await
                    .context("Failed to set value in Redis with expiry")?;
            },
            None => {
                let () = conn.set::<_, _, ()>(key, serialized).await
                    .context("Failed to set value in Redis")?;
            }
        }
        
        debug!("Cached value for key: {}", key);
        Ok(())
    }
    
    pub fn generate_key<T: Serialize>(prefix: &str, data: &T) -> Result<String> {
    let serialized = serde_json::to_string(data)
        .context("Failed to serialize data for key generation")?;
    
    let mut hasher = Sha256::new();
    hasher.update(serialized.as_bytes());
    let hash = hasher.finalize();
    
    let hash_hex = format!("{:x}", hash);
    let key = format!("{}_{}", prefix, hash_hex);
    
    info!("Generated key: {}", key);
    
    Ok(key)
}

}