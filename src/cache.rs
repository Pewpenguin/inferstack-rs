use anyhow::{Context, Result as AnyhowResult};
use deadpool_redis::redis::{cmd, AsyncCommands};
use deadpool_redis::{Config, Pool, PoolConfig, Runtime};
use serde::{de::DeserializeOwned, Serialize};
use sha2::{Digest, Sha256};
use std::time::Duration;
use tracing::{debug, info};


pub struct CacheService {
    pool: Pool,
}

impl CacheService {
    pub async fn new_with_pool_size(redis_url: &str, pool_size: u32) -> AnyhowResult<Self> {
        if pool_size == 0 {
            anyhow::bail!(
                "Redis pool size must be greater than 0; set REDIS_POOL_SIZE to a positive integer"
            );
        }
        let max_size = usize::try_from(pool_size)
            .map_err(|_| anyhow::anyhow!("Redis pool size does not fit in usize"))?;

        let mut cfg = Config::from_url(redis_url);
        cfg.pool = Some(PoolConfig::new(max_size));

        let pool = cfg.create_pool(Some(Runtime::Tokio1))?;
        info!(
            "Initialized Redis connection pool with max_size={} (REDIS_POOL_SIZE)",
            max_size
        );
        Ok(Self { pool })
    }
    
    pub async fn health_check(&self) -> AnyhowResult<()> {
        let mut conn = self.pool.get().await
            .context("Failed to get Redis connection from pool")?;
        cmd("PING").query_async::<()>(&mut conn).await
            .context("Redis health check failed")?;
        Ok(())
    }
    
    pub async fn get_with_retry<T: DeserializeOwned>(
        &self, 
        key: &str, 
        retries: u32, 
        retry_delay_ms: u64
    ) -> AnyhowResult<Option<T>> {
        let mut attempts = 0;
        let max_attempts = retries + 1;
        
        loop {
            attempts += 1;
            match self.get::<T>(key).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    if attempts >= max_attempts {
                        return Err(e);
                    }
                    debug!("Redis GET retry {}/{} for key '{}': {}", 
                           attempts, max_attempts, key, e);
                    tokio::time::sleep(Duration::from_millis(retry_delay_ms)).await;
                }
            }
        }
    }
    
    pub async fn set_with_retry<T: Serialize>(
        &self, 
        key: &str, 
        value: &T, 
        ttl_seconds: Option<usize>, 
        retries: u32, 
        retry_delay_ms: u64
    ) -> AnyhowResult<()> {
        let mut attempts = 0;
        let max_attempts = retries + 1;

        loop {
            attempts += 1;
            match self.set(key, value, ttl_seconds).await {
                Ok(_) => return Ok(()),
                Err(e) => {
                    if attempts >= max_attempts {
                        return Err(e);
                    }
                    debug!("Redis SET retry {}/{} for key '{}': {}", 
                           attempts, max_attempts, key, e);
                    tokio::time::sleep(Duration::from_millis(retry_delay_ms)).await;
                }
            }
        }
    }

    pub async fn get<T: DeserializeOwned>(&self, key: &str) -> AnyhowResult<Option<T>> {
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
    ) -> AnyhowResult<()> {
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
    
    pub fn generate_key_with_version<T: Serialize>(prefix: &str, data: &T, version: u32) -> AnyhowResult<String> {
        let serialized =
            serde_json::to_string(data).context("Failed to serialize data for key generation")?;

        let mut hasher = Sha256::new();
        hasher.update(serialized.as_bytes());
        hasher.update(version.to_string().as_bytes());
        let hash = hasher.finalize();

        let hash_hex = format!("{:x}", hash);
        let key = format!("{}_v{}{}", prefix, version, hash_hex);

        debug!("Generated versioned key: {}", key);
        Ok(key)
    }
}
