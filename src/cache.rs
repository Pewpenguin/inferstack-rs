use anyhow::{Context, Result};
use deadpool_redis::redis::{cmd, AsyncCommands};
use deadpool_redis::{Config, Pool, Runtime};
use serde::{de::DeserializeOwned, Serialize};
use sha2::{Digest, Sha256};
use tracing::{debug, info};

pub struct CacheService {
    pool: Pool,
}

impl CacheService {
    pub async fn new(redis_url: &str) -> Result<Self> {
        let cfg = Config::from_url(redis_url);
        let pool = cfg.create_pool(Some(Runtime::Tokio1))?;
        Ok(Self { pool })
    }

    pub async fn get<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        let mut conn = self
            .pool
            .get()
            .await
            .context("Failed to get Redis connection from pool")?;

        let result: Option<String> = conn
            .get(key)
            .await
            .context("Failed to get value from Redis")?;

        match result {
            Some(data) => {
                let value: T =
                    serde_json::from_str(&data).context("Failed to deserialize cached data")?;
                debug!("Cache hit for key: {}", key);
                Ok(Some(value))
            }
            None => {
                debug!("Cache miss for key: {}", key);
                Ok(None)
            }
        }
    }

    pub async fn set<T: Serialize>(
        &self,
        key: &str,
        value: &T,
        ttl_seconds: Option<usize>,
    ) -> Result<()> {
        let mut conn = self
            .pool
            .get()
            .await
            .context("Failed to get Redis connection from pool")?;

        let serialized =
            serde_json::to_string(value).context("Failed to serialize value for caching")?;

        match ttl_seconds {
            Some(ttl) => {
                let _: () = cmd("SETEX")
                    .arg(key)
                    .arg(ttl)
                    .arg(&serialized)
                    .query_async(&mut conn)
                    .await
                    .context("Failed to set value in Redis with expiry")?;
            }
            None => {
                let _: () = cmd("SET")
                    .arg(key)
                    .arg(&serialized)
                    .query_async(&mut conn)
                    .await
                    .context("Failed to set value in Redis")?;
            }
        }

        debug!("Cached value for key: {}", key);
        Ok(())
    }

    pub fn generate_key<T: Serialize>(prefix: &str, data: &T) -> Result<String> {
        let serialized =
            serde_json::to_string(data).context("Failed to serialize data for key generation")?;

        let mut hasher = Sha256::new();
        hasher.update(serialized.as_bytes());
        let hash = hasher.finalize();

        let hash_hex = format!("{:x}", hash);
        let key = format!("{}_{}", prefix, hash_hex);

        info!("Generated key: {}", key);
        Ok(key)
    }
}
