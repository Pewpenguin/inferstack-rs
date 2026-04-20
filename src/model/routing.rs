use rand::RngExt;

use crate::error::AppError;

use super::{ModelService, ModelVersion};

impl ModelService {
    pub fn select_model_version(
        &self,
        requested_version: Option<&str>,
    ) -> Result<&ModelVersion, AppError> {
        let r = rand::rng().random_range(0..100);
        self.select_model_version_with_roll(requested_version, r)
    }

    pub fn select_model_version_with_roll(
        &self,
        requested_version: Option<&str>,
        routing_roll: u8,
    ) -> Result<&ModelVersion, AppError> {
        if let Some(version) = requested_version {
            return self.models.get(version).ok_or_else(|| {
                AppError::ValidationError(format!("Model version '{}' not found", version))
            });
        }

        let r = routing_roll;

        let mut prev: u8 = 0;
        for entry in &self.routing_table {
            if (prev..entry.cumulative_upper).contains(&r) {
                return self.models.get(&entry.version).ok_or_else(|| {
                    AppError::InternalError("Routing entry missing loaded model".to_string())
                });
            }
            prev = entry.cumulative_upper;
        }

        self.models
            .get(&self.default_version)
            .ok_or_else(|| AppError::InternalError("Default model version not found".to_string()))
    }
}
