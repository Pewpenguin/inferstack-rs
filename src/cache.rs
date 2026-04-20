use deadpool_redis::redis::{cmd, AsyncCommands};
use deadpool_redis::{Config, Pool, PoolConfig, Runtime};
use serde::{de::DeserializeOwned, Serialize};
use sha2::{Digest, Sha256};
use std::time::Duration;
use tracing::{debug, info};

use crate::error::AppError;

pub struct CacheService {
    pool: Pool,
}

impl CacheService {
    pub async fn new_with_pool_size(redis_url: &str, pool_size: u32) -> Result<Self, AppError> {
        if pool_size == 0 {
            return Err(AppError::ConfigError(
                "Redis pool size must be greater than 0; set REDIS_POOL_SIZE to a positive integer"
                    .to_string(),
            ));
        }
        let max_size = usize::try_from(pool_size).map_err(|_| {
            AppError::ConfigError("Redis pool size does not fit in usize".to_string())
        })?;

        let mut cfg = Config::from_url(redis_url);
        cfg.pool = Some(PoolConfig::new(max_size));

        let pool = cfg.create_pool(Some(Runtime::Tokio1)).map_err(|e| {
            AppError::CacheError(format!("Failed to create Redis connection pool: {}", e))
        })?;
        info!(
            "Initialized Redis connection pool with max_size={} (REDIS_POOL_SIZE)",
            max_size
        );
        Ok(Self { pool })
    }

    pub async fn health_check(&self) -> Result<(), AppError> {
        let mut conn = self.pool.get().await.map_err(|e| {
            AppError::CacheError(format!("Failed to get Redis connection from pool: {}", e))
        })?;
        cmd("PING")
            .query_async::<()>(&mut conn)
            .await
            .map_err(|e| AppError::CacheError(format!("Redis health check failed: {}", e)))?;
        Ok(())
    }

    pub async fn get_with_retry<T: DeserializeOwned>(
        &self,
        key: &str,
        retries: u32,
        retry_delay_ms: u64,
    ) -> Result<Option<T>, AppError> {
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
                    debug!(
                        cache_op = "get",
                        key = key,
                        attempt = attempts,
                        max_attempts,
                        error = %e,
                        "Retrying cache get operation"
                    );
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
        retry_delay_ms: u64,
    ) -> Result<(), AppError> {
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
                    debug!(
                        cache_op = "set",
                        key = key,
                        attempt = attempts,
                        max_attempts,
                        error = %e,
                        "Retrying cache set operation"
                    );
                    tokio::time::sleep(Duration::from_millis(retry_delay_ms)).await;
                }
            }
        }
    }

    pub async fn get<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>, AppError> {
        let mut conn = self.pool.get().await.map_err(|e| {
            AppError::CacheError(format!("Failed to get Redis connection from pool: {}", e))
        })?;

        let result: Option<String> = conn
            .get(key)
            .await
            .map_err(|e| AppError::CacheError(format!("Failed to get value from Redis: {}", e)))?;

        match result {
            Some(data) => {
                let value: T = serde_json::from_str(&data).map_err(|e| {
                    AppError::CacheError(format!("Failed to deserialize cached data: {}", e))
                })?;
                info!(cache_hit = true, key = key, "Cache lookup completed");
                Ok(Some(value))
            }
            None => {
                info!(cache_hit = false, key = key, "Cache lookup completed");
                Ok(None)
            }
        }
    }

    pub async fn set<T: Serialize>(
        &self,
        key: &str,
        value: &T,
        ttl_seconds: Option<usize>,
    ) -> Result<(), AppError> {
        let mut conn = self.pool.get().await.map_err(|e| {
            AppError::CacheError(format!("Failed to get Redis connection from pool: {}", e))
        })?;

        let serialized = serde_json::to_string(value).map_err(|e| {
            AppError::CacheError(format!("Failed to serialize value for caching: {}", e))
        })?;

        match ttl_seconds {
            Some(ttl) => {
                let _: () = cmd("SETEX")
                    .arg(key)
                    .arg(ttl)
                    .arg(&serialized)
                    .query_async(&mut conn)
                    .await
                    .map_err(|e| {
                        AppError::CacheError(format!(
                            "Failed to set value in Redis with expiry: {}",
                            e
                        ))
                    })?;
            }
            None => {
                let _: () = cmd("SET")
                    .arg(key)
                    .arg(&serialized)
                    .query_async(&mut conn)
                    .await
                    .map_err(|e| {
                        AppError::CacheError(format!("Failed to set value in Redis: {}", e))
                    })?;
            }
        }

        debug!(key = key, ttl_seconds = ?ttl_seconds, "Cached value stored");
        Ok(())
    }

    pub fn generate_key_with_version<T: Serialize>(
        prefix: &str,
        data: &T,
        version: u32,
    ) -> Result<String, AppError> {
        let serialized = serde_json::to_string(data).map_err(|e| {
            AppError::CacheError(format!(
                "Failed to serialize data for key generation: {}",
                e
            ))
        })?;

        let mut hasher = Sha256::new();
        hasher.update(serialized.as_bytes());
        hasher.update(version.to_string().as_bytes());
        let hash = hasher.finalize();

        let hash_hex: String = hash.iter().map(|byte| format!("{:02x}", byte)).collect();
        let key = format!("{}_v{}{}", prefix, version, hash_hex);

        debug!("Generated versioned cache key");
        Ok(key)
    }
}
