use tracing::debug;

use crate::config::NormalizeInput;
use crate::error::AppError;

use super::{ModelService, ModelVersion};

impl ModelService {
    pub(super) fn validate_input(
        &self,
        model_version: &ModelVersion,
        input_data: &[f32],
    ) -> Result<(), AppError> {
        if input_data.is_empty() {
            return Err(AppError::ValidationError(
                "Input data cannot be empty".to_string(),
            ));
        }

        let input_spec = model_version
            .spec
            .inputs
            .first()
            .ok_or_else(|| AppError::ValidationError("Model has no input tensors".to_string()))?;
        if !input_spec.shape.is_empty() {
            let dims = if input_spec.shape.len() > 1 {
                &input_spec.shape[1..]
            } else {
                &input_spec.shape[..]
            };
            if !dims.is_empty() && !dims.contains(&0) {
                let expected_size = dims.iter().product::<usize>();
                if input_data.len() != expected_size {
                    return Err(AppError::ValidationError(format!(
                        "Input '{}' size mismatch: got {}, expected {}",
                        input_spec.name,
                        input_data.len(),
                        expected_size
                    )));
                }
            }
        }

        Ok(())
    }

    pub(super) fn preprocess_input(&self, input_data: &[f32]) -> Result<Vec<f32>, AppError> {
        match self.normalize_input {
            NormalizeInput::None => Ok(input_data.to_vec()),
            NormalizeInput::MinMax => {
                let mut processed = input_data.to_vec();

                let needs_normalization =
                    processed.iter().copied().any(|x| !(0.0..=1.0).contains(&x));

                if needs_normalization {
                    debug!(
                        "Min–max normalizing input data to [0,1] range (NORMALIZE_INPUT=minmax)"
                    );

                    let min_val = processed.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max_val = processed.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let range = max_val - min_val;

                    if range > 0.0 {
                        for val in &mut processed {
                            *val = (*val - min_val) / range;
                        }
                    }
                }

                Ok(processed)
            }
        }
    }
}
