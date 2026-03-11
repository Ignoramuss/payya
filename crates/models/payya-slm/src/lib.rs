//! Small Language Model — end-to-end trainable language model.
//!
//! Wires together `payya-transformer`, `payya-tokenizer`, and
//! `payya-logit-processor` into a single trainable language model with:
//!
//! - Text-in, text-out training and generation
//! - Learning rate warmup and gradient clipping
//! - Checkpoint save/load in JSON format
//!
//! # Example
//!
//! ```
//! use payya_slm::{Slm, SlmConfig};
//!
//! let config = SlmConfig {
//!     vocab_size: 280,
//!     d_model: 32,
//!     n_heads: 2,
//!     n_layers: 1,
//!     d_ff: 64,
//!     max_seq_len: 64,
//! };
//! let mut slm = Slm::new(config, 42);
//! ```

use payya_logit_processor::LogitProcessor;
use payya_tokenizer::Tokenizer;
use payya_transformer::{PosEncoding, Transformer, TransformerConfig, TransformerParams};
use rand::Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

// ── Configuration ──────────────────────────────────────────────────────

/// SLM hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlmConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
}

/// Training hyperparameters.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Base learning rate.
    pub lr: f32,
    /// Number of warmup steps (linear warmup from 0 to lr).
    pub warmup_steps: usize,
    /// Maximum gradient norm for clipping. None = no clipping.
    pub max_grad_norm: Option<f32>,
    /// Weight decay coefficient. 0.0 = no decay.
    pub weight_decay: f32,
    /// Sliding window size for training chunks.
    pub window_size: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            warmup_steps: 100,
            max_grad_norm: Some(1.0),
            weight_decay: 0.01,
            window_size: 64,
        }
    }
}

// ── Checkpoint ─────────────────────────────────────────────────────────

/// Serializable checkpoint containing model config, parameters, tokenizer,
/// and training state.
#[derive(Serialize, Deserialize)]
pub struct Checkpoint {
    pub config: SlmConfig,
    pub params: TransformerParams,
    pub tokenizer: Option<String>,
    pub step: usize,
}

impl Checkpoint {
    /// Serialize to JSON bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("checkpoint serialization should not fail")
    }

    /// Deserialize from JSON bytes.
    ///
    /// # Panics
    ///
    /// Panics if the bytes do not contain a valid checkpoint.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        serde_json::from_slice(bytes).expect("invalid checkpoint data")
    }
}

// ── SLM ────────────────────────────────────────────────────────────────

/// A Small Language Model: transformer + tokenizer + training utilities.
pub struct Slm {
    transformer: Transformer,
    tokenizer: Option<Tokenizer>,
    config: SlmConfig,
    step: usize,
}

impl Slm {
    /// Create a new SLM with random parameters.
    pub fn new(config: SlmConfig, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let tf_config = Self::make_tf_config(&config);
        let transformer = Transformer::new(tf_config, &mut rng);
        Self {
            transformer,
            tokenizer: None,
            config,
            step: 0,
        }
    }

    /// Create an SLM with a tokenizer. The config's vocab_size is set to match.
    pub fn with_tokenizer(mut config: SlmConfig, tokenizer: Tokenizer, seed: u64) -> Self {
        config.vocab_size = tokenizer.vocab_size();
        let mut slm = Self::new(config, seed);
        slm.tokenizer = Some(tokenizer);
        slm
    }

    /// Restore from a checkpoint.
    pub fn from_checkpoint(checkpoint: Checkpoint) -> Self {
        let tf_config = Self::make_tf_config(&checkpoint.config);
        let transformer = Transformer::from_params(tf_config, checkpoint.params);
        let tokenizer = checkpoint
            .tokenizer
            .as_ref()
            .map(|json| Tokenizer::from_json(json));
        Self {
            transformer,
            tokenizer,
            config: checkpoint.config,
            step: checkpoint.step,
        }
    }

    /// Save a checkpoint.
    pub fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            config: self.config.clone(),
            params: self.transformer.params.clone(),
            tokenizer: self.tokenizer.as_ref().map(|t| t.to_json()),
            step: self.step,
        }
    }

    /// Return the current training step.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Return the config.
    pub fn config(&self) -> &SlmConfig {
        &self.config
    }

    /// Return a reference to the tokenizer, if set.
    pub fn tokenizer(&self) -> Option<&Tokenizer> {
        self.tokenizer.as_ref()
    }

    /// Return a reference to the underlying transformer.
    pub fn transformer(&self) -> &Transformer {
        &self.transformer
    }

    fn make_tf_config(config: &SlmConfig) -> TransformerConfig {
        TransformerConfig {
            vocab_size: config.vocab_size,
            d_model: config.d_model,
            n_heads: config.n_heads,
            n_layers: config.n_layers,
            d_ff: config.d_ff,
            max_seq_len: config.max_seq_len,
            pos_encoding: PosEncoding::Sinusoidal,
        }
    }

    // ── Training ────────────────────────────────────────────────────────

    /// Run one training step on raw token IDs with the given training config.
    ///
    /// Returns the loss value. Internally handles LR warmup, gradient clipping,
    /// and weight decay.
    pub fn train_step_ids(&mut self, tokens: &[usize], train_config: &TrainConfig) -> f32 {
        assert!(
            tokens.len() >= 2,
            "need at least 2 tokens for next-token prediction"
        );

        let input = &tokens[..tokens.len() - 1];
        let targets = &tokens[1..];

        let (mut g, logits) = self.transformer.forward(input);
        let loss = g.cross_entropy(logits, targets);
        let loss_val = g.data(loss)[0];
        g.backward(loss);

        // Compute effective learning rate with warmup.
        let lr = if self.step < train_config.warmup_steps && train_config.warmup_steps > 0 {
            train_config.lr * (self.step as f32 + 1.0) / train_config.warmup_steps as f32
        } else {
            train_config.lr
        };

        // Compute global gradient norm for clipping.
        let grad_scale = if let Some(max_norm) = train_config.max_grad_norm {
            let global_norm = compute_grad_norm(&g);
            if global_norm > max_norm {
                max_norm / (global_norm + 1e-8)
            } else {
                1.0
            }
        } else {
            1.0
        };

        // SGD with weight decay and gradient clipping.
        update_params(
            &mut self.transformer,
            &g,
            lr,
            grad_scale,
            train_config.weight_decay,
        );

        self.step += 1;
        loss_val
    }

    /// Train on a text corpus for the given number of steps.
    ///
    /// Requires a tokenizer to be set. Returns loss values for each step.
    ///
    /// # Panics
    ///
    /// Panics if no tokenizer is set or the corpus is too short.
    pub fn train_text(
        &mut self,
        corpus: &str,
        num_steps: usize,
        train_config: &TrainConfig,
    ) -> Vec<f32> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .expect("tokenizer must be set for text training");
        let token_ids: Vec<usize> = tokenizer
            .encode(corpus)
            .iter()
            .map(|&t| t as usize)
            .collect();
        assert!(token_ids.len() >= 2, "corpus produces fewer than 2 tokens");

        let window = train_config.window_size.min(token_ids.len());
        let max_start = if token_ids.len() > window {
            token_ids.len() - window
        } else {
            0
        };

        let mut losses = Vec::with_capacity(num_steps);
        for i in 0..num_steps {
            let start = if max_start > 0 {
                (i * 7) % (max_start + 1) // deterministic stride through corpus
            } else {
                0
            };
            let end = (start + window).min(token_ids.len());
            let chunk = &token_ids[start..end];
            if chunk.len() < 2 {
                continue;
            }
            let loss = self.train_step_ids(chunk, train_config);
            losses.push(loss);
        }
        losses
    }

    // ── Generation ──────────────────────────────────────────────────────

    /// Generate text from a prompt string.
    ///
    /// # Panics
    ///
    /// Panics if no tokenizer is set.
    pub fn generate_text(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        processor: &LogitProcessor,
        rng: &mut impl Rng,
    ) -> String {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .expect("tokenizer must be set for text generation");
        let prompt_tokens: Vec<usize> = tokenizer
            .encode(prompt)
            .iter()
            .map(|&t| t as usize)
            .collect();
        assert!(
            !prompt_tokens.is_empty(),
            "prompt must produce at least one token"
        );

        let generated = self
            .transformer
            .generate(&prompt_tokens, max_new_tokens, processor, rng);
        let generated_ids: Vec<u32> = generated.iter().map(|&t| t as u32).collect();
        tokenizer.decode(&generated_ids)
    }

    /// Generate token IDs from prompt token IDs.
    pub fn generate_ids(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        processor: &LogitProcessor,
        rng: &mut impl Rng,
    ) -> Vec<usize> {
        self.transformer
            .generate(prompt, max_new_tokens, processor, rng)
    }
}

// ── Parameter update utilities ─────────────────────────────────────────

/// Compute the global L2 norm of all gradients.
fn compute_grad_norm(g: &payya_autograd::Graph) -> f32 {
    let mut sum_sq = 0.0f32;
    let num_nodes = g.num_nodes();
    for node_id in 0..num_nodes {
        let tid = payya_autograd::TensorId::from_raw(node_id);
        if !g.is_param(tid) || !g.has_grad(tid) {
            continue;
        }
        let grad = g.grad(tid);
        for &val in grad {
            sum_sq += val * val;
        }
    }
    sum_sq.sqrt()
}

/// Apply SGD update with gradient clipping and weight decay.
fn update_params(
    transformer: &mut Transformer,
    g: &payya_autograd::Graph,
    lr: f32,
    grad_scale: f32,
    weight_decay: f32,
) {
    let mut param_vecs = collect_param_refs_mut(transformer);
    let mut param_idx = 0;

    let num_nodes = g.num_nodes();
    for node_id in 0..num_nodes {
        let tid = payya_autograd::TensorId::from_raw(node_id);
        if !g.is_param(tid) {
            continue;
        }
        if param_idx >= param_vecs.len() {
            break;
        }
        if !g.has_grad(tid) {
            param_idx += 1;
            continue;
        }
        let grad = g.grad(tid);
        let params = &mut param_vecs[param_idx];
        assert_eq!(
            params.len(),
            grad.len(),
            "param/grad size mismatch at param_idx={param_idx}"
        );
        for (p, &gr) in params.iter_mut().zip(grad.iter()) {
            let clipped_grad = gr * grad_scale;
            // Weight decay (decoupled, AdamW-style).
            if weight_decay > 0.0 {
                *p *= 1.0 - lr * weight_decay;
            }
            *p -= lr * clipped_grad;
        }
        param_idx += 1;
    }
}

/// Collect mutable references to all parameter vectors in graph construction order.
/// Mirrors the order used by `Transformer::forward`.
fn collect_param_refs_mut(transformer: &mut Transformer) -> Vec<&mut Vec<f32>> {
    let mut refs: Vec<&mut Vec<f32>> = Vec::new();

    refs.push(&mut transformer.params.token_emb);

    for layer in &mut transformer.params.layers {
        refs.push(&mut layer.ln1_gamma);
        refs.push(&mut layer.ln1_beta);
        refs.push(&mut layer.wq);
        refs.push(&mut layer.bq);
        refs.push(&mut layer.wk);
        refs.push(&mut layer.bk);
        refs.push(&mut layer.wv);
        refs.push(&mut layer.bv);
        refs.push(&mut layer.wo);
        refs.push(&mut layer.bo);
        refs.push(&mut layer.ln2_gamma);
        refs.push(&mut layer.ln2_beta);
        refs.push(&mut layer.w1);
        refs.push(&mut layer.b1);
        refs.push(&mut layer.w2);
        refs.push(&mut layer.b2);
    }

    refs.push(&mut transformer.params.final_ln_gamma);
    refs.push(&mut transformer.params.final_ln_beta);
    refs.push(&mut transformer.params.output_weight);
    refs.push(&mut transformer.params.output_bias);

    refs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> SlmConfig {
        SlmConfig {
            vocab_size: 32,
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            d_ff: 32,
            max_seq_len: 64,
        }
    }

    fn train_config() -> TrainConfig {
        TrainConfig {
            lr: 0.01,
            warmup_steps: 5,
            max_grad_norm: Some(1.0),
            weight_decay: 0.0,
            window_size: 32,
        }
    }

    #[test]
    fn new_creates_model() {
        let slm = Slm::new(test_config(), 42);
        assert_eq!(slm.step(), 0);
        assert!(slm.tokenizer().is_none());
    }

    #[test]
    fn with_tokenizer_sets_vocab_size() {
        let tokenizer = Tokenizer::train("hello world hello world hello", 260);
        let vocab = tokenizer.vocab_size();
        let config = SlmConfig {
            vocab_size: 999, // will be overridden
            ..test_config()
        };
        let slm = Slm::with_tokenizer(config, tokenizer, 42);
        assert_eq!(slm.config().vocab_size, vocab);
        assert!(slm.tokenizer().is_some());
    }

    #[test]
    fn train_step_ids_returns_finite_loss() {
        let mut slm = Slm::new(test_config(), 42);
        let tc = train_config();
        let tokens: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let loss = slm.train_step_ids(&tokens, &tc);
        assert!(loss.is_finite(), "loss must be finite, got {loss}");
        assert!(
            loss > 0.0,
            "cross-entropy loss must be positive, got {loss}"
        );
        assert_eq!(slm.step(), 1);
    }

    #[test]
    fn training_loss_decreases() {
        let mut slm = Slm::new(test_config(), 42);
        let tc = TrainConfig {
            lr: 0.01,
            warmup_steps: 0,
            max_grad_norm: None,
            weight_decay: 0.0,
            window_size: 32,
        };
        let pattern: Vec<usize> = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
        let mut losses = Vec::new();
        for _ in 0..30 {
            let loss = slm.train_step_ids(&pattern, &tc);
            losses.push(loss);
        }
        let first = losses[0];
        let last = *losses.last().unwrap();
        assert!(
            last < first,
            "loss should decrease: first={first}, last={last}"
        );
    }

    #[test]
    fn warmup_increases_lr() {
        let mut slm = Slm::new(test_config(), 42);
        let tc = TrainConfig {
            lr: 0.1,
            warmup_steps: 10,
            max_grad_norm: None,
            weight_decay: 0.0,
            window_size: 32,
        };
        let tokens: Vec<usize> = vec![0, 1, 2, 3, 4, 5];

        // Step 0: effective lr = 0.1 * 1/10 = 0.01
        // Step 9: effective lr = 0.1 * 10/10 = 0.1
        // Just verify training doesn't crash during warmup.
        for _ in 0..12 {
            let loss = slm.train_step_ids(&tokens, &tc);
            assert!(loss.is_finite());
        }
    }

    #[test]
    fn gradient_clipping_prevents_explosion() {
        let mut slm = Slm::new(test_config(), 42);
        let tc = TrainConfig {
            lr: 10.0, // absurdly high LR
            warmup_steps: 0,
            max_grad_norm: Some(0.1), // but very tight clipping
            weight_decay: 0.0,
            window_size: 32,
        };
        let tokens: Vec<usize> = vec![0, 1, 2, 3, 4, 5];
        // With high LR but tight clipping, the model shouldn't explode.
        for _ in 0..5 {
            let loss = slm.train_step_ids(&tokens, &tc);
            assert!(
                loss.is_finite(),
                "loss should be finite with gradient clipping, got {loss}"
            );
        }
    }

    #[test]
    fn weight_decay_shrinks_params() {
        let slm_no_decay = Slm::new(test_config(), 42);
        let params_before: f32 = slm_no_decay
            .transformer()
            .params
            .token_emb
            .iter()
            .map(|x| x * x)
            .sum();

        let mut slm = Slm::new(test_config(), 42);
        let tc = TrainConfig {
            lr: 0.01,
            warmup_steps: 0,
            max_grad_norm: None,
            weight_decay: 0.1,
            window_size: 32,
        };
        let tokens: Vec<usize> = vec![0, 1, 2, 3];
        for _ in 0..10 {
            slm.train_step_ids(&tokens, &tc);
        }
        let params_after: f32 = slm
            .transformer()
            .params
            .token_emb
            .iter()
            .map(|x| x * x)
            .sum();

        assert!(
            params_after < params_before,
            "weight decay should shrink param norm: before={params_before}, after={params_after}"
        );
    }

    #[test]
    fn train_text_works() {
        let corpus = "the cat sat on the mat and the dog ran in the park";
        let tokenizer = Tokenizer::train(corpus, 270);
        let config = SlmConfig {
            vocab_size: tokenizer.vocab_size(),
            ..test_config()
        };
        let mut slm = Slm::with_tokenizer(config, tokenizer, 42);
        let tc = train_config();
        let losses = slm.train_text(corpus, 20, &tc);
        assert_eq!(losses.len(), 20);
        for &loss in &losses {
            assert!(loss.is_finite());
        }
    }

    #[test]
    fn generate_ids_produces_valid_tokens() {
        let slm = Slm::new(test_config(), 42);
        let processor = LogitProcessor::new().with_temperature(1.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let prompt = vec![0, 1, 2];
        let generated = slm.generate_ids(&prompt, 10, &processor, &mut rng);
        assert_eq!(generated.len(), 13); // 3 + 10
        for &tok in &generated {
            assert!(tok < slm.config().vocab_size, "token {tok} out of range");
        }
    }

    #[test]
    fn generate_text_works() {
        let corpus = "the cat sat on the mat and the dog ran in the park";
        let tokenizer = Tokenizer::train(corpus, 270);
        let config = SlmConfig {
            vocab_size: tokenizer.vocab_size(),
            ..test_config()
        };
        let slm = Slm::with_tokenizer(config, tokenizer, 42);
        let processor = LogitProcessor::new().with_temperature(0.8);
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let text = slm.generate_text("the", 10, &processor, &mut rng);
        assert!(!text.is_empty());
    }

    #[test]
    fn checkpoint_roundtrip() {
        let corpus = "the cat sat on the mat and the dog ran in the park";
        let tokenizer = Tokenizer::train(corpus, 270);
        let config = SlmConfig {
            vocab_size: tokenizer.vocab_size(),
            ..test_config()
        };
        let mut slm = Slm::with_tokenizer(config, tokenizer, 42);
        let tc = train_config();

        // Train a few steps.
        slm.train_text(corpus, 5, &tc);
        let step_before = slm.step();

        // Save checkpoint.
        let ckpt = slm.checkpoint();
        let bytes = ckpt.to_bytes();
        assert!(!bytes.is_empty());

        // Restore and verify.
        let ckpt2 = Checkpoint::from_bytes(&bytes);
        let slm2 = Slm::from_checkpoint(ckpt2);
        assert_eq!(slm2.step(), step_before);
        assert!(slm2.tokenizer().is_some());

        // Params should match exactly.
        assert_eq!(
            slm.transformer().params.token_emb,
            slm2.transformer().params.token_emb
        );
    }

    #[test]
    fn checkpoint_roundtrip_no_loss_spike() {
        let config = test_config();
        let mut slm = Slm::new(config, 42);
        let tc = TrainConfig {
            lr: 0.005,
            warmup_steps: 0,
            max_grad_norm: Some(1.0),
            weight_decay: 0.0,
            window_size: 32,
        };
        let pattern: Vec<usize> = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];

        // Train to get loss down.
        let mut loss_before_save = 0.0;
        for _ in 0..20 {
            loss_before_save = slm.train_step_ids(&pattern, &tc);
        }

        // Save & restore.
        let ckpt = slm.checkpoint();
        let bytes = ckpt.to_bytes();
        let mut slm2 = Slm::from_checkpoint(Checkpoint::from_bytes(&bytes));

        // Continue training — loss should not spike.
        let loss_after_restore = slm2.train_step_ids(&pattern, &tc);
        let spike_ratio = loss_after_restore / loss_before_save;
        assert!(
            spike_ratio < 2.0,
            "loss spiked after restore: before={loss_before_save}, after={loss_after_restore}, ratio={spike_ratio}"
        );
    }

    #[test]
    fn train_text_loss_improves() {
        let corpus = "aaaa bbbb cccc aaaa bbbb cccc aaaa bbbb cccc aaaa bbbb cccc";
        let tokenizer = Tokenizer::train(corpus, 265);
        let config = SlmConfig {
            vocab_size: tokenizer.vocab_size(),
            d_model: 32,
            n_heads: 2,
            n_layers: 2,
            d_ff: 64,
            max_seq_len: 64,
        };
        let mut slm = Slm::with_tokenizer(config, tokenizer, 42);
        let tc = TrainConfig {
            lr: 0.005,
            warmup_steps: 0,
            max_grad_norm: Some(1.0),
            weight_decay: 0.0,
            window_size: 32,
        };
        let losses = slm.train_text(corpus, 50, &tc);

        // Average of first 5 losses should be higher than average of last 5.
        let first_avg: f32 = losses[..5].iter().sum::<f32>() / 5.0;
        let last_avg: f32 = losses[losses.len() - 5..].iter().sum::<f32>() / 5.0;
        assert!(
            last_avg < first_avg,
            "loss should improve over training: first_avg={first_avg}, last_avg={last_avg}"
        );
    }

    #[test]
    #[should_panic(expected = "need at least 2 tokens")]
    fn train_step_too_short_panics() {
        let mut slm = Slm::new(test_config(), 42);
        let tc = train_config();
        slm.train_step_ids(&[0], &tc);
    }
}
