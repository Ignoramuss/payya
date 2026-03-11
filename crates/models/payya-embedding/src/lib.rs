//! Sentence-level embedding model.
//!
//! Builds on `payya-transformer` to produce fixed-size vector representations
//! of token sequences. Supports mean pooling over transformer hidden states.
//!
//! # Example
//!
//! ```
//! use payya_embedding::{EmbeddingModel, PoolingStrategy};
//! use payya_transformer::{TransformerConfig, PosEncoding};
//! use rand::SeedableRng;
//!
//! let config = TransformerConfig {
//!     vocab_size: 32,
//!     d_model: 16,
//!     n_heads: 2,
//!     n_layers: 1,
//!     d_ff: 32,
//!     max_seq_len: 64,
//!     pos_encoding: PosEncoding::Sinusoidal,
//! };
//! let mut rng = rand::rngs::StdRng::seed_from_u64(42);
//! let model = EmbeddingModel::new(config, PoolingStrategy::Mean, &mut rng);
//! let embedding = model.embed(&[0, 1, 2, 3]);
//! assert_eq!(embedding.len(), 16); // d_model
//! ```

use payya_transformer::{Transformer, TransformerConfig};
use rand::Rng;

/// How to aggregate per-token hidden states into a single vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PoolingStrategy {
    /// Average all token hidden states: sum(h_i) / seq_len.
    Mean,
    /// Use the first token's hidden state (CLS-style).
    FirstToken,
}

/// A sentence embedding model backed by a transformer encoder.
///
/// Produces a fixed-size `d_model`-dimensional vector for any input sequence.
pub struct EmbeddingModel {
    transformer: Transformer,
    pooling: PoolingStrategy,
}

impl EmbeddingModel {
    /// Create a new embedding model with random parameters.
    pub fn new(config: TransformerConfig, pooling: PoolingStrategy, rng: &mut impl Rng) -> Self {
        let transformer = Transformer::new(config, rng);
        Self {
            transformer,
            pooling,
        }
    }

    /// Create an embedding model from an existing transformer.
    pub fn from_transformer(transformer: Transformer, pooling: PoolingStrategy) -> Self {
        Self {
            transformer,
            pooling,
        }
    }

    /// Return the embedding dimensionality.
    pub fn dim(&self) -> usize {
        self.transformer.config.d_model
    }

    /// Return a reference to the underlying transformer.
    pub fn transformer(&self) -> &Transformer {
        &self.transformer
    }

    /// Return a mutable reference to the underlying transformer.
    pub fn transformer_mut(&mut self) -> &mut Transformer {
        &mut self.transformer
    }

    /// Compute a fixed-size embedding for a token sequence.
    ///
    /// Returns a vector of length `d_model`.
    ///
    /// # Panics
    ///
    /// Panics if `tokens` is empty or exceeds `max_seq_len`.
    pub fn embed(&self, tokens: &[usize]) -> Vec<f32> {
        assert!(!tokens.is_empty(), "tokens must not be empty");
        let (g, hidden_id) = self.transformer.forward_hidden(tokens);
        let hidden = g.data(hidden_id);
        let d = self.transformer.config.d_model;
        let seq = tokens.len();
        assert_eq!(
            hidden.len(),
            seq * d,
            "hidden state size mismatch: got {}, expected {}",
            hidden.len(),
            seq * d
        );

        self.pool(hidden, seq, d)
    }

    /// Apply pooling over hidden states (seq, d_model) → (d_model,).
    fn pool(&self, hidden: &[f32], seq: usize, d: usize) -> Vec<f32> {
        match self.pooling {
            PoolingStrategy::Mean => {
                let mut result = vec![0.0f32; d];
                for row in 0..seq {
                    for col in 0..d {
                        result[col] += hidden[row * d + col];
                    }
                }
                let scale = 1.0 / seq as f32;
                for val in &mut result {
                    *val *= scale;
                }
                result
            }
            PoolingStrategy::FirstToken => hidden[..d].to_vec(),
        }
    }
}

/// Compute cosine similarity between two vectors.
///
/// Returns a value in [-1, 1]. Identical directions → 1.0.
///
/// # Panics
///
/// Panics if vectors have different lengths or either has zero norm.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "vectors must have same length: {} vs {}",
        a.len(),
        b.len()
    );
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        norm_a > 0.0,
        "first vector has zero norm — cannot compute cosine similarity"
    );
    assert!(
        norm_b > 0.0,
        "second vector has zero norm — cannot compute cosine similarity"
    );
    dot / (norm_a * norm_b)
}

/// Normalize a vector to unit length (L2 normalization).
///
/// # Panics
///
/// Panics if the vector has zero norm.
pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(norm > 0.0, "cannot normalize zero vector");
    v.iter().map(|x| x / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use payya_transformer::{PosEncoding, TransformerConfig};
    use rand::SeedableRng;

    fn test_config() -> TransformerConfig {
        TransformerConfig {
            vocab_size: 32,
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            d_ff: 32,
            max_seq_len: 64,
            pos_encoding: PosEncoding::Sinusoidal,
        }
    }

    fn seeded_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

    #[test]
    fn embed_returns_correct_dim() {
        let mut rng = seeded_rng();
        let model = EmbeddingModel::new(test_config(), PoolingStrategy::Mean, &mut rng);
        let emb = model.embed(&[0, 1, 2, 3, 4]);
        assert_eq!(emb.len(), 16);
    }

    #[test]
    fn embed_values_are_finite() {
        let mut rng = seeded_rng();
        let model = EmbeddingModel::new(test_config(), PoolingStrategy::Mean, &mut rng);
        let emb = model.embed(&[0, 1, 2, 3]);
        for &val in &emb {
            assert!(val.is_finite(), "embedding value must be finite, got {val}");
        }
    }

    #[test]
    fn first_token_pooling_differs_from_mean() {
        let mut rng = seeded_rng();
        let config = test_config();
        let model_mean = EmbeddingModel::new(config.clone(), PoolingStrategy::Mean, &mut rng);
        // Same params for both (same rng state won't work, so use from_transformer).
        let model_first = EmbeddingModel::from_transformer(
            model_mean.transformer().clone(),
            PoolingStrategy::FirstToken,
        );

        let tokens = &[0, 1, 2, 3, 4];
        let emb_mean = model_mean.embed(tokens);
        let emb_first = model_first.embed(tokens);

        // With more than one token, mean and first-token should differ.
        assert_ne!(
            emb_mean, emb_first,
            "mean and first-token pooling should differ for multi-token input"
        );
    }

    #[test]
    fn single_token_mean_equals_first() {
        let mut rng = seeded_rng();
        let config = test_config();
        let model = EmbeddingModel::new(config, PoolingStrategy::Mean, &mut rng);
        let model_first = EmbeddingModel::from_transformer(
            model.transformer().clone(),
            PoolingStrategy::FirstToken,
        );

        let emb_mean = model.embed(&[5]);
        let emb_first = model_first.embed(&[5]);

        for (a, b) in emb_mean.iter().zip(emb_first.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "single token: mean={a} should equal first={b}"
            );
        }
    }

    #[test]
    fn different_inputs_produce_different_embeddings() {
        let mut rng = seeded_rng();
        let model = EmbeddingModel::new(test_config(), PoolingStrategy::Mean, &mut rng);
        let emb1 = model.embed(&[0, 1, 2]);
        let emb2 = model.embed(&[3, 4, 5]);
        assert_ne!(
            emb1, emb2,
            "different inputs should produce different embeddings"
        );
    }

    #[test]
    fn cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "cosine similarity of identical vectors should be 1.0, got {sim}"
        );
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-6,
            "cosine similarity of orthogonal vectors should be 0.0, got {sim}"
        );
    }

    #[test]
    fn cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b: Vec<f32> = a.iter().map(|x| -x).collect();
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim + 1.0).abs() < 1e-6,
            "cosine similarity of opposite vectors should be -1.0, got {sim}"
        );
    }

    #[test]
    fn l2_normalize_unit_length() {
        let v = vec![3.0, 4.0];
        let n = l2_normalize(&v);
        let norm: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "normalized vector should have unit length, got {norm}"
        );
    }

    #[test]
    #[should_panic(expected = "vectors must have same length")]
    fn cosine_similarity_length_mismatch() {
        cosine_similarity(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    #[should_panic(expected = "zero norm")]
    fn cosine_similarity_zero_vector() {
        cosine_similarity(&[0.0, 0.0], &[1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "tokens must not be empty")]
    fn embed_empty_tokens_panics() {
        let mut rng = seeded_rng();
        let model = EmbeddingModel::new(test_config(), PoolingStrategy::Mean, &mut rng);
        model.embed(&[]);
    }

    #[test]
    fn embed_nonzero_norm() {
        let mut rng = seeded_rng();
        let model = EmbeddingModel::new(test_config(), PoolingStrategy::Mean, &mut rng);
        let emb = model.embed(&[0, 1, 2]);
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm > 0.0, "embedding should have non-zero norm");
    }

    #[test]
    fn similar_inputs_more_similar_than_random() {
        // Two overlapping sequences should be more similar than disjoint ones.
        let mut rng = seeded_rng();
        let model = EmbeddingModel::new(test_config(), PoolingStrategy::Mean, &mut rng);

        let emb_a = model.embed(&[0, 1, 2, 3]);
        let emb_b = model.embed(&[0, 1, 2, 4]); // one token different
        let emb_c = model.embed(&[10, 11, 12, 13]); // totally different

        let sim_ab = cosine_similarity(&emb_a, &emb_b);
        let sim_ac = cosine_similarity(&emb_a, &emb_c);

        // With random weights this won't always hold, but with a seeded RNG
        // and overlapping input, it should. If not, that's fine — this is a
        // sanity check, not a hard invariant.
        assert!(
            sim_ab > sim_ac,
            "overlapping inputs should be more similar: sim_ab={sim_ab}, sim_ac={sim_ac}"
        );
    }
}
