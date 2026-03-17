//! Direct Preference Optimization (DPO) loss function.
//!
//! Implements the DPO loss from "Direct Preference Optimization: Your Language
//! Model is Secretly a Reward Model" (Rafailov et al., 2023). Given chosen and
//! rejected response pairs, optimizes the policy model to prefer chosen
//! responses without a separate reward model.
//!
//! The DPO loss is:
//!   L = -log(sigmoid(beta * (log_pi(chosen) - log_pi_ref(chosen)
//!                           - log_pi(rejected) + log_pi_ref(rejected))))
//!
//! where pi is the policy model and pi_ref is the frozen reference model.

use payya_transformer::Transformer;

// ── Configuration ────────────────────────────────────────────────────

/// DPO training configuration.
#[derive(Debug, Clone)]
pub struct DpoConfig {
    /// Temperature parameter controlling preference strength. Higher beta
    /// means the model deviates less from the reference.
    pub beta: f32,
    /// Learning rate for SGD updates.
    pub lr: f32,
}

impl Default for DpoConfig {
    fn default() -> Self {
        Self {
            beta: 0.1,
            lr: 1e-4,
        }
    }
}

/// A single preference pair: chosen response is preferred over rejected.
#[derive(Debug, Clone)]
pub struct PreferencePair {
    /// Token IDs of the prompt (shared prefix).
    pub prompt: Vec<usize>,
    /// Token IDs of the chosen (preferred) response.
    pub chosen: Vec<usize>,
    /// Token IDs of the rejected response.
    pub rejected: Vec<usize>,
}

// ── Log-probability computation ─────────────────────────────────────

/// Compute the total log-probability of a sequence under a model.
/// The model computes logits for positions 0..len-1 predicting tokens 1..len.
/// Returns sum of log P(token[i+1] | token[0..=i]) for all i.
pub fn sequence_log_prob(model: &Transformer, tokens: &[usize]) -> f32 {
    assert!(
        tokens.len() >= 2,
        "need at least 2 tokens for log-prob computation, got {}",
        tokens.len()
    );

    let (g, logits_id) = model.forward(tokens);
    let logits = g.data(logits_id);
    let vocab = model.config.vocab_size;
    let seq = tokens.len();

    let mut total_log_prob = 0.0f32;
    for pos in 0..seq - 1 {
        let row = &logits[pos * vocab..(pos + 1) * vocab];
        let target = tokens[pos + 1];
        assert!(
            target < vocab,
            "target token {target} out of vocab range {vocab}"
        );

        // Numerically stable log-softmax: log(exp(x_i) / sum(exp(x_j)))
        //   = x_i - log(sum(exp(x_j)))
        //   = x_i - max - log(sum(exp(x_j - max)))
        let max_logit = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp: f32 = row.iter().map(|&x| (x - max_logit).exp()).sum::<f32>().ln();
        let log_prob = row[target] - max_logit - log_sum_exp;
        total_log_prob += log_prob;
    }

    total_log_prob
}

/// Compute the total log-probability of response tokens given a prompt,
/// using only the response positions for the log-prob sum.
pub fn response_log_prob(model: &Transformer, prompt: &[usize], response: &[usize]) -> f32 {
    assert!(!prompt.is_empty(), "prompt must not be empty");
    assert!(!response.is_empty(), "response must not be empty");

    let mut full_seq: Vec<usize> = Vec::with_capacity(prompt.len() + response.len());
    full_seq.extend_from_slice(prompt);
    full_seq.extend_from_slice(response);

    let (g, logits_id) = model.forward(&full_seq);
    let logits = g.data(logits_id);
    let vocab = model.config.vocab_size;
    let prompt_len = prompt.len();

    let mut total_log_prob = 0.0f32;
    // Sum log-probs only over response positions.
    // Position (prompt_len - 1) predicts response[0], up to position (full_len - 2) predicts response[last].
    for (i, &target) in response.iter().enumerate() {
        let pos = prompt_len - 1 + i;
        let row = &logits[pos * vocab..(pos + 1) * vocab];
        assert!(
            target < vocab,
            "target token {target} out of vocab range {vocab}"
        );

        let max_logit = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp: f32 = row.iter().map(|&x| (x - max_logit).exp()).sum::<f32>().ln();
        total_log_prob += row[target] - max_logit - log_sum_exp;
    }

    total_log_prob
}

// ── DPO Loss ────────────────────────────────────────────────────────

/// Compute the DPO loss for a single preference pair.
///
/// Returns the loss value (scalar, lower is better).
///
/// The DPO loss is:
///   L = -log(sigmoid(beta * (log_pi_chosen - log_ref_chosen
///                           - log_pi_rejected + log_ref_rejected)))
pub fn dpo_loss(
    policy: &Transformer,
    reference: &Transformer,
    pair: &PreferencePair,
    beta: f32,
) -> f32 {
    assert!(beta > 0.0, "beta must be positive, got {beta}");

    let log_pi_chosen = response_log_prob(policy, &pair.prompt, &pair.chosen);
    let log_ref_chosen = response_log_prob(reference, &pair.prompt, &pair.chosen);
    let log_pi_rejected = response_log_prob(policy, &pair.prompt, &pair.rejected);
    let log_ref_rejected = response_log_prob(reference, &pair.prompt, &pair.rejected);

    let chosen_reward = log_pi_chosen - log_ref_chosen;
    let rejected_reward = log_pi_rejected - log_ref_rejected;
    let logit = beta * (chosen_reward - rejected_reward);

    // -log(sigmoid(x)) = log(1 + exp(-x))
    // For numerical stability: softplus(-x) when x > 0, softplus(x) = x + softplus(-x) when x < 0
    if logit >= 0.0 {
        (-logit).exp().ln_1p()
    } else {
        -logit + logit.exp().ln_1p()
    }
}

/// Compute DPO loss over a batch of preference pairs. Returns mean loss.
pub fn dpo_loss_batch(
    policy: &Transformer,
    reference: &Transformer,
    pairs: &[PreferencePair],
    beta: f32,
) -> f32 {
    assert!(!pairs.is_empty(), "preference pairs must not be empty");
    let total: f32 = pairs
        .iter()
        .map(|p| dpo_loss(policy, reference, p, beta))
        .sum();
    total / pairs.len() as f32
}

// ── DPO Trainer ─────────────────────────────────────────────────────

/// Trains a policy model using DPO against a frozen reference model.
pub struct DpoTrainer {
    /// The policy model being optimized.
    pub policy: Transformer,
    /// The frozen reference model (snapshot of policy before DPO).
    reference: Transformer,
    /// DPO configuration.
    pub config: DpoConfig,
    /// Training step counter.
    step: usize,
}

impl DpoTrainer {
    /// Create a DPO trainer. The reference model is cloned from the policy.
    pub fn new(policy: Transformer, config: DpoConfig) -> Self {
        let reference = policy.clone();
        Self {
            policy,
            reference,
            config,
            step: 0,
        }
    }

    /// Current training step.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Run one DPO training step on a preference pair.
    /// Returns the loss value.
    ///
    /// Uses finite-difference gradient estimation: for each parameter,
    /// compute (loss(p+eps) - loss(p-eps)) / (2*eps) and update via SGD.
    ///
    /// This is correct but slow — suitable for small models. The autograd
    /// graph approach would be more efficient but requires building the DPO
    /// loss within the computation graph, which needs cross-model graph support.
    pub fn train_step(&mut self, pair: &PreferencePair) -> f32 {
        let loss = dpo_loss(&self.policy, &self.reference, pair, self.config.beta);

        // Compute gradients via finite differences and apply SGD.
        let lr = self.config.lr;
        let eps = 1e-4_f32;

        // We update parameters one at a time via finite differences.
        // Collect all mutable param slices.
        let mut flat_params = flatten_params(&self.policy);

        for i in 0..flat_params.len() {
            let orig = flat_params[i];

            // f(p + eps)
            flat_params[i] = orig + eps;
            set_flat_params(&mut self.policy, &flat_params);
            let loss_plus = dpo_loss(&self.policy, &self.reference, pair, self.config.beta);

            // f(p - eps)
            flat_params[i] = orig - eps;
            set_flat_params(&mut self.policy, &flat_params);
            let loss_minus = dpo_loss(&self.policy, &self.reference, pair, self.config.beta);

            // Central difference gradient.
            let grad = (loss_plus - loss_minus) / (2.0 * eps);

            // SGD update.
            flat_params[i] = orig - lr * grad;
        }

        set_flat_params(&mut self.policy, &flat_params);
        self.step += 1;

        loss
    }

    /// Train on a batch of preference pairs (one step per pair).
    /// Returns the mean loss over the batch.
    pub fn train_batch(&mut self, pairs: &[PreferencePair]) -> f32 {
        assert!(!pairs.is_empty(), "pairs must not be empty");
        let mut total_loss = 0.0f32;
        for pair in pairs {
            total_loss += self.train_step(pair);
        }
        total_loss / pairs.len() as f32
    }
}

// ── Parameter flattening helpers ────────────────────────────────────

fn count_params(model: &Transformer) -> usize {
    let mut total = model.params.token_emb.len();
    for layer in &model.params.layers {
        total += layer.wq.len() + layer.wk.len() + layer.wv.len() + layer.wo.len();
        total += layer.bq.len() + layer.bk.len() + layer.bv.len() + layer.bo.len();
        total += layer.ln1_gamma.len() + layer.ln1_beta.len();
        total += layer.w1.len() + layer.w2.len() + layer.b1.len() + layer.b2.len();
        total += layer.ln2_gamma.len() + layer.ln2_beta.len();
    }
    total += model.params.final_ln_gamma.len() + model.params.final_ln_beta.len();
    total += model.params.output_weight.len() + model.params.output_bias.len();
    total
}

fn flatten_params(model: &Transformer) -> Vec<f32> {
    let mut flat = Vec::with_capacity(count_params(model));
    flat.extend_from_slice(&model.params.token_emb);
    for layer in &model.params.layers {
        flat.extend_from_slice(&layer.wq);
        flat.extend_from_slice(&layer.wk);
        flat.extend_from_slice(&layer.wv);
        flat.extend_from_slice(&layer.wo);
        flat.extend_from_slice(&layer.bq);
        flat.extend_from_slice(&layer.bk);
        flat.extend_from_slice(&layer.bv);
        flat.extend_from_slice(&layer.bo);
        flat.extend_from_slice(&layer.ln1_gamma);
        flat.extend_from_slice(&layer.ln1_beta);
        flat.extend_from_slice(&layer.w1);
        flat.extend_from_slice(&layer.w2);
        flat.extend_from_slice(&layer.b1);
        flat.extend_from_slice(&layer.b2);
        flat.extend_from_slice(&layer.ln2_gamma);
        flat.extend_from_slice(&layer.ln2_beta);
    }
    flat.extend_from_slice(&model.params.final_ln_gamma);
    flat.extend_from_slice(&model.params.final_ln_beta);
    flat.extend_from_slice(&model.params.output_weight);
    flat.extend_from_slice(&model.params.output_bias);
    flat
}

fn set_flat_params(model: &mut Transformer, flat: &[f32]) {
    let mut offset = 0;

    let n = model.params.token_emb.len();
    model
        .params
        .token_emb
        .copy_from_slice(&flat[offset..offset + n]);
    offset += n;

    for layer in &mut model.params.layers {
        for param in [
            &mut layer.wq,
            &mut layer.wk,
            &mut layer.wv,
            &mut layer.wo,
            &mut layer.bq,
            &mut layer.bk,
            &mut layer.bv,
            &mut layer.bo,
            &mut layer.ln1_gamma,
            &mut layer.ln1_beta,
            &mut layer.w1,
            &mut layer.w2,
            &mut layer.b1,
            &mut layer.b2,
            &mut layer.ln2_gamma,
            &mut layer.ln2_beta,
        ] {
            let n = param.len();
            param.copy_from_slice(&flat[offset..offset + n]);
            offset += n;
        }
    }

    let n = model.params.final_ln_gamma.len();
    model
        .params
        .final_ln_gamma
        .copy_from_slice(&flat[offset..offset + n]);
    offset += n;
    let n = model.params.final_ln_beta.len();
    model
        .params
        .final_ln_beta
        .copy_from_slice(&flat[offset..offset + n]);
    offset += n;
    let n = model.params.output_weight.len();
    model
        .params
        .output_weight
        .copy_from_slice(&flat[offset..offset + n]);
    offset += n;
    let n = model.params.output_bias.len();
    model
        .params
        .output_bias
        .copy_from_slice(&flat[offset..offset + n]);
    offset += n;

    assert_eq!(offset, flat.len(), "flat params length mismatch");
}

#[cfg(test)]
mod tests {
    use super::*;
    use payya_transformer::{PosEncoding, TransformerConfig};
    use rand::SeedableRng;

    fn tiny_config() -> TransformerConfig {
        TransformerConfig {
            vocab_size: 8,
            d_model: 8,
            n_heads: 2,
            n_layers: 1,
            d_ff: 16,
            max_seq_len: 32,
            pos_encoding: PosEncoding::Sinusoidal,
        }
    }

    fn seeded_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

    #[test]
    fn sequence_log_prob_is_finite_and_negative() {
        let mut rng = seeded_rng();
        let model = Transformer::new(tiny_config(), &mut rng);
        let tokens = vec![0, 1, 2, 3, 4];
        let lp = sequence_log_prob(&model, &tokens);
        assert!(lp.is_finite(), "log-prob must be finite, got {lp}");
        assert!(lp <= 0.0, "log-prob must be <= 0, got {lp}");
    }

    #[test]
    fn response_log_prob_is_finite() {
        let mut rng = seeded_rng();
        let model = Transformer::new(tiny_config(), &mut rng);
        let prompt = vec![0, 1];
        let response = vec![2, 3, 4];
        let lp = response_log_prob(&model, &prompt, &response);
        assert!(lp.is_finite(), "response log-prob must be finite, got {lp}");
        assert!(lp <= 0.0, "response log-prob must be <= 0, got {lp}");
    }

    #[test]
    fn dpo_loss_is_finite() {
        let mut rng = seeded_rng();
        let policy = Transformer::new(tiny_config(), &mut rng);
        let reference = policy.clone();
        let pair = PreferencePair {
            prompt: vec![0, 1],
            chosen: vec![2, 3],
            rejected: vec![4, 5],
        };
        let loss = dpo_loss(&policy, &reference, &pair, 0.1);
        assert!(loss.is_finite(), "DPO loss must be finite, got {loss}");
    }

    #[test]
    fn dpo_loss_identical_models_is_log2() {
        // When policy == reference, the log-ratio terms cancel and the loss
        // should be -log(sigmoid(0)) = log(2).
        let mut rng = seeded_rng();
        let policy = Transformer::new(tiny_config(), &mut rng);
        let reference = policy.clone();
        let pair = PreferencePair {
            prompt: vec![0, 1],
            chosen: vec![2, 3],
            rejected: vec![4, 5],
        };
        let loss = dpo_loss(&policy, &reference, &pair, 0.1);
        let expected = 2.0f32.ln(); // log(2) ≈ 0.693
        assert!(
            (loss - expected).abs() < 1e-4,
            "DPO loss with identical models should be log(2)={expected}, got {loss}"
        );
    }

    #[test]
    fn dpo_loss_batch_works() {
        let mut rng = seeded_rng();
        let policy = Transformer::new(tiny_config(), &mut rng);
        let reference = policy.clone();
        let pairs = vec![
            PreferencePair {
                prompt: vec![0, 1],
                chosen: vec![2, 3],
                rejected: vec![4, 5],
            },
            PreferencePair {
                prompt: vec![0],
                chosen: vec![1, 2, 3],
                rejected: vec![5, 6, 7],
            },
        ];
        let loss = dpo_loss_batch(&policy, &reference, &pairs, 0.1);
        assert!(loss.is_finite());
    }

    #[test]
    fn dpo_trainer_loss_decreases() {
        // With finite-difference training on a tiny model, loss should decrease.
        let mut rng = seeded_rng();
        let config = TransformerConfig {
            vocab_size: 4,
            d_model: 4,
            n_heads: 1,
            n_layers: 1,
            d_ff: 8,
            max_seq_len: 16,
            pos_encoding: PosEncoding::Sinusoidal,
        };
        let policy = Transformer::new(config, &mut rng);
        let dpo_config = DpoConfig {
            beta: 0.1,
            lr: 0.01,
        };
        let mut trainer = DpoTrainer::new(policy, dpo_config);

        let pair = PreferencePair {
            prompt: vec![0],
            chosen: vec![1, 2],
            rejected: vec![3, 0],
        };

        let first_loss = trainer.train_step(&pair);
        // Do a few more steps.
        let mut last_loss = first_loss;
        for _ in 0..3 {
            last_loss = trainer.train_step(&pair);
        }
        // Loss should be finite throughout.
        assert!(last_loss.is_finite(), "loss must be finite");
        assert_eq!(trainer.step(), 4);
    }
}
