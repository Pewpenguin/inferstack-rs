use rand::RngExt;
use tracing::info;

use crate::error::AppError;

use super::{ModelService, ModelVersion};

impl ModelService {
    pub fn select_model_version(
        &self,
        requested_version: Option<&str>,
    ) -> Result<&ModelVersion, AppError> {
        let r = rand::rng().random_range(0..100);
        info!(
            routing_roll = r,
            requested_model_version = requested_version.unwrap_or("auto"),
            "Selecting model version"
        );
        self.select_model_version_with_roll(requested_version, r)
    }

    pub fn select_model_version_with_roll(
        &self,
        requested_version: Option<&str>,
        routing_roll: u8,
    ) -> Result<&ModelVersion, AppError> {
        if let Some(version) = requested_version {
            let selected = self.models.get(version).ok_or_else(|| {
                AppError::NotFound(format!("Model version '{}' not found", version))
            })?;
            info!(
                model_version = %selected.version,
                routing_mode = "explicit",
                "Selected requested model version"
            );
            return Ok(selected);
        }

        let r = routing_roll;

        let mut prev: u8 = 0;
        for entry in &self.routing_table {
            if (prev..entry.cumulative_upper).contains(&r) {
                let selected = self.models.get(&entry.version).ok_or_else(|| {
                    AppError::InternalError("Routing entry missing loaded model".to_string())
                })?;
                info!(
                    model_version = %selected.version,
                    routing_roll = r,
                    routing_mode = "weighted",
                    "Selected model version by traffic allocation"
                );
                return Ok(selected);
            }
            prev = entry.cumulative_upper;
        }

        let selected = self.models.get(&self.default_version).ok_or_else(|| {
            AppError::InternalError("Default model version not found".to_string())
        })?;
        info!(
            model_version = %selected.version,
            routing_roll = r,
            routing_mode = "fallback_default",
            "Selected default model version after routing bounds fallback"
        );
        Ok(selected)
    }
}
