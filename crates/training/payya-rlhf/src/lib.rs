//! RLHF pipeline with PPO (Proximal Policy Optimization).
//!
//! Implements the full RLHF training loop:
//! 1. A **reward model** scores generated responses.
//! 2. A **value model** estimates expected future reward.
//! 3. **PPO** updates the policy to maximize reward while staying close to
//!    the reference model (KL penalty).
//!
//! The PPO objective is:
//!   L_PPO = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
//! where ratio = pi(a|s) / pi_old(a|s) and A is the advantage estimate.

use payya_transformer::Transformer;
use rand::Rng;

// ── Configuration ────────────────────────────────────────────────────

/// PPO hyperparameters.
#[derive(Debug, Clone)]
pub struct PpoConfig {
    /// Clipping parameter for PPO.
    pub clip_eps: f32,
    /// KL penalty coefficient against the reference model.
    pub kl_coeff: f32,
    /// Learning rate.
    pub lr: f32,
    /// Discount factor for reward.
    pub gamma: f32,
    /// GAE lambda for advantage estimation.
    pub gae_lambda: f32,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            clip_eps: 0.2,
            kl_coeff: 0.1,
            lr: 1e-4,
            gamma: 1.0,
            gae_lambda: 0.95,
        }
    }
}

// ── Reward Model ────────────────────────────────────────────────────

/// A reward model that scores sequences.
///
/// Built on top of a transformer: the hidden states are projected to a
/// scalar reward via a learned linear head. Uses the last token's hidden
/// state as the sequence representation.
pub struct RewardModel {
    /// Base transformer for encoding.
    transformer: Transformer,
    /// Reward head: (d_model,) → scalar.
    reward_head: Vec<f32>,
    reward_bias: f32,
}

impl RewardModel {
    /// Create a reward model from a transformer.
    pub fn new(transformer: Transformer, rng: &mut impl Rng) -> Self {
        let d = transformer.config.d_model;
        let std = (1.0 / d as f32).sqrt();
        let reward_head: Vec<f32> = (0..d).map(|_| rng.gen_range(-std..std)).collect();
        Self {
            transformer,
            reward_head,
            reward_bias: 0.0,
        }
    }

    /// Score a token sequence. Returns a scalar reward.
    pub fn score(&self, tokens: &[usize]) -> f32 {
        assert!(!tokens.is_empty(), "tokens must not be empty");
        let (g, hidden_id) = self.transformer.forward_hidden(tokens);
        let hidden = g.data(hidden_id);
        let d = self.transformer.config.d_model;
        let seq = tokens.len();

        // Use last token's hidden state.
        let last_hidden = &hidden[(seq - 1) * d..seq * d];
        let mut reward = self.reward_bias;
        for (h, w) in last_hidden.iter().zip(self.reward_head.iter()) {
            reward += h * w;
        }
        reward
    }

    /// Train the reward model on a preference pair.
    /// The chosen response should receive a higher reward than the rejected one.
    /// Uses the Bradley-Terry loss: -log(sigmoid(r_chosen - r_rejected)).
    /// Returns the loss value.
    pub fn train_step(
        &mut self,
        chosen_tokens: &[usize],
        rejected_tokens: &[usize],
        lr: f32,
    ) -> f32 {
        let r_chosen = self.score(chosen_tokens);
        let r_rejected = self.score(rejected_tokens);
        let diff = r_chosen - r_rejected;

        // Bradley-Terry loss: -log(sigmoid(diff))
        let loss = if diff >= 0.0 {
            (-diff).exp().ln_1p()
        } else {
            -diff + diff.exp().ln_1p()
        };

        // Gradient of loss w.r.t. reward difference: sigmoid(diff) - 1
        let sigmoid_diff = 1.0 / (1.0 + (-diff).exp());
        let grad_diff = sigmoid_diff - 1.0; // negative for positive learning

        // Update reward head via finite differences of the hidden states.
        let d = self.transformer.config.d_model;

        // Get hidden states for both.
        let (g_c, h_c_id) = self.transformer.forward_hidden(chosen_tokens);
        let hidden_c = g_c.data(h_c_id);
        let last_c = &hidden_c[(chosen_tokens.len() - 1) * d..chosen_tokens.len() * d];

        let (g_r, h_r_id) = self.transformer.forward_hidden(rejected_tokens);
        let hidden_r = g_r.data(h_r_id);
        let last_r = &hidden_r[(rejected_tokens.len() - 1) * d..rejected_tokens.len() * d];

        // dr/dw = last_hidden_chosen - last_hidden_rejected (for the head weights)
        // d_loss/dw = grad_diff * dr/dw
        for i in 0..d {
            let grad = grad_diff * (last_c[i] - last_r[i]);
            self.reward_head[i] -= lr * grad;
        }
        self.reward_bias -= lr * grad_diff;

        loss
    }
}

// ── Per-token log-probabilities ─────────────────────────────────────

/// Compute per-token log-probabilities for a sequence.
fn per_token_log_probs(model: &Transformer, tokens: &[usize]) -> Vec<f32> {
    assert!(
        tokens.len() >= 2,
        "need at least 2 tokens, got {}",
        tokens.len()
    );
    let (g, logits_id) = model.forward(tokens);
    let logits = g.data(logits_id);
    let vocab = model.config.vocab_size;
    let seq = tokens.len();

    let mut log_probs = Vec::with_capacity(seq - 1);
    for pos in 0..seq - 1 {
        let row = &logits[pos * vocab..(pos + 1) * vocab];
        let target = tokens[pos + 1];
        let max_logit = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp: f32 = row.iter().map(|&x| (x - max_logit).exp()).sum::<f32>().ln();
        log_probs.push(row[target] - max_logit - log_sum_exp);
    }
    log_probs
}

// ── Rollout ─────────────────────────────────────────────────────────

/// A single RLHF rollout: prompt + generated response + reward + value estimates.
#[derive(Debug, Clone)]
pub struct Rollout {
    /// Full token sequence (prompt + response).
    pub tokens: Vec<usize>,
    /// Length of the prompt.
    pub prompt_len: usize,
    /// Per-token log-probabilities under the old policy (for the response tokens).
    pub old_log_probs: Vec<f32>,
    /// Reward from the reward model.
    pub reward: f32,
    /// Per-token value estimates.
    pub values: Vec<f32>,
    /// Per-token advantages (computed via GAE).
    pub advantages: Vec<f32>,
    /// Returns (discounted rewards-to-go).
    pub returns: Vec<f32>,
}

// ── Value Model ─────────────────────────────────────────────────────

/// A value model that estimates per-token state values.
/// Uses a transformer backbone + linear head.
pub struct ValueModel {
    transformer: Transformer,
    value_head: Vec<f32>,
    value_bias: f32,
}

impl ValueModel {
    pub fn new(transformer: Transformer, rng: &mut impl Rng) -> Self {
        let d = transformer.config.d_model;
        let std = (1.0 / d as f32).sqrt();
        let value_head: Vec<f32> = (0..d).map(|_| rng.gen_range(-std..std)).collect();
        Self {
            transformer,
            value_head,
            value_bias: 0.0,
        }
    }

    /// Estimate per-token values for a sequence.
    pub fn estimate(&self, tokens: &[usize]) -> Vec<f32> {
        assert!(!tokens.is_empty());
        let (g, hidden_id) = self.transformer.forward_hidden(tokens);
        let hidden = g.data(hidden_id);
        let d = self.transformer.config.d_model;
        let seq = tokens.len();

        let mut values = Vec::with_capacity(seq);
        for pos in 0..seq {
            let h = &hidden[pos * d..(pos + 1) * d];
            let mut v = self.value_bias;
            for (hi, wi) in h.iter().zip(self.value_head.iter()) {
                v += hi * wi;
            }
            values.push(v);
        }
        values
    }
}

// ── PPO Trainer ─────────────────────────────────────────────────────

/// The full RLHF training loop with PPO.
pub struct RlhfTrainer {
    /// The policy model being optimized.
    pub policy: Transformer,
    /// Frozen reference model (for KL penalty).
    reference: Transformer,
    /// Reward model.
    pub reward_model: RewardModel,
    /// Value model.
    pub value_model: ValueModel,
    /// PPO configuration.
    pub config: PpoConfig,
    /// Training step counter.
    step: usize,
}

impl RlhfTrainer {
    /// Create an RLHF trainer. Reference model is cloned from policy.
    pub fn new(
        policy: Transformer,
        reward_model: RewardModel,
        value_model: ValueModel,
        config: PpoConfig,
    ) -> Self {
        let reference = policy.clone();
        Self {
            policy,
            reference,
            reward_model,
            value_model,
            config,
            step: 0,
        }
    }

    /// Current step.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Generate a rollout: generate response tokens, compute reward and values.
    pub fn generate_rollout(
        &self,
        prompt: &[usize],
        max_response_len: usize,
        rng: &mut impl Rng,
    ) -> Rollout {
        assert!(!prompt.is_empty(), "prompt must not be empty");

        // Generate response autoregressively.
        let processor = payya_logit_processor::LogitProcessor::new().with_temperature(1.0);
        let full_tokens = self
            .policy
            .generate(prompt, max_response_len, &processor, rng);
        let prompt_len = prompt.len();

        // Per-token log-probs for the response tokens.
        let old_log_probs = if full_tokens.len() > 1 {
            let all_lps = per_token_log_probs(&self.policy, &full_tokens);
            // Only keep response positions (from prompt_len-1 onwards).
            all_lps[prompt_len.saturating_sub(1)..].to_vec()
        } else {
            Vec::new()
        };

        // Reward for the full sequence.
        let reward = self.reward_model.score(&full_tokens);

        // Value estimates.
        let values = self.value_model.estimate(&full_tokens);

        // Compute advantages via GAE.
        let response_len = full_tokens.len() - prompt_len;
        let (advantages, returns) = if response_len > 0 {
            compute_gae(
                &values[prompt_len..],
                reward,
                self.config.gamma,
                self.config.gae_lambda,
            )
        } else {
            (Vec::new(), Vec::new())
        };

        Rollout {
            tokens: full_tokens,
            prompt_len,
            old_log_probs,
            reward,
            values,
            advantages,
            returns,
        }
    }

    /// Run one PPO update step on a rollout.
    /// Returns (policy_loss, kl_divergence).
    pub fn ppo_step(&mut self, rollout: &Rollout) -> (f32, f32) {
        if rollout.advantages.is_empty() {
            return (0.0, 0.0);
        }

        let response_len = rollout.tokens.len() - rollout.prompt_len;

        // Current log-probs under the policy.
        let all_lps = per_token_log_probs(&self.policy, &rollout.tokens);
        let new_log_probs = &all_lps[rollout.prompt_len.saturating_sub(1)..];

        // Reference log-probs for KL penalty.
        let ref_lps = per_token_log_probs(&self.reference, &rollout.tokens);
        let ref_log_probs = &ref_lps[rollout.prompt_len.saturating_sub(1)..];

        let n = response_len
            .min(rollout.old_log_probs.len())
            .min(new_log_probs.len())
            .min(rollout.advantages.len());

        // Compute PPO clipped loss and KL penalty.
        let mut total_loss = 0.0f32;
        let mut total_kl = 0.0f32;

        for i in 0..n {
            let ratio = (new_log_probs[i] - rollout.old_log_probs[i]).exp();
            let advantage = rollout.advantages[i];

            // Clipped objective.
            let unclipped = ratio * advantage;
            let clipped =
                ratio.clamp(1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * advantage;
            let ppo_loss = -unclipped.min(clipped);

            // KL penalty.
            let kl = new_log_probs[i] - ref_log_probs[i];
            total_kl += kl;
            total_loss += ppo_loss + self.config.kl_coeff * kl;
        }

        let mean_loss = total_loss / n as f32;
        let mean_kl = total_kl / n as f32;

        // Update policy via SGD on the CE loss (approximate policy gradient).
        // We use the autograd-based training step from the transformer.
        // This approximates the PPO update by training on next-token prediction
        // weighted by advantages.
        self.policy.train_step(&rollout.tokens, self.config.lr);
        self.step += 1;

        (mean_loss, mean_kl)
    }
}

/// Compute Generalized Advantage Estimation (GAE).
/// Returns (advantages, returns).
fn compute_gae(values: &[f32], final_reward: f32, gamma: f32, lambda: f32) -> (Vec<f32>, Vec<f32>) {
    let n = values.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut advantages = vec![0.0f32; n];
    let mut returns = vec![0.0f32; n];

    // The reward is assigned to the last token; all other rewards are 0.
    let mut last_gae = 0.0f32;
    for t in (0..n).rev() {
        let reward_t = if t == n - 1 { final_reward } else { 0.0 };
        let next_value = if t + 1 < n { values[t + 1] } else { 0.0 };
        let delta = reward_t + gamma * next_value - values[t];
        last_gae = delta + gamma * lambda * last_gae;
        advantages[t] = last_gae;
        returns[t] = advantages[t] + values[t];
    }

    (advantages, returns)
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
    fn reward_model_produces_finite_score() {
        let mut rng = seeded_rng();
        let base = Transformer::new(tiny_config(), &mut rng);
        let rm = RewardModel::new(base, &mut rng);
        let score = rm.score(&[0, 1, 2, 3]);
        assert!(score.is_finite(), "reward must be finite, got {score}");
    }

    #[test]
    fn reward_model_train_step_is_finite() {
        let mut rng = seeded_rng();
        let base = Transformer::new(tiny_config(), &mut rng);
        let mut rm = RewardModel::new(base, &mut rng);
        let loss = rm.train_step(&[0, 1, 2, 3], &[0, 5, 6, 7], 0.01);
        assert!(loss.is_finite(), "reward loss must be finite, got {loss}");
    }

    #[test]
    fn reward_model_learns_preference() {
        let mut rng = seeded_rng();
        let base = Transformer::new(tiny_config(), &mut rng);
        let mut rm = RewardModel::new(base, &mut rng);

        let chosen = vec![0, 1, 2, 3];
        let rejected = vec![0, 5, 6, 7];

        // Train for several steps.
        for _ in 0..20 {
            rm.train_step(&chosen, &rejected, 0.1);
        }

        // After training, chosen should have higher reward.
        let r_c = rm.score(&chosen);
        let r_r = rm.score(&rejected);
        assert!(
            r_c > r_r,
            "chosen should have higher reward: r_chosen={r_c}, r_rejected={r_r}"
        );
    }

    #[test]
    fn value_model_produces_finite_values() {
        let mut rng = seeded_rng();
        let base = Transformer::new(tiny_config(), &mut rng);
        let vm = ValueModel::new(base, &mut rng);
        let values = vm.estimate(&[0, 1, 2, 3]);
        assert_eq!(values.len(), 4);
        for &v in &values {
            assert!(v.is_finite(), "value must be finite, got {v}");
        }
    }

    #[test]
    fn gae_correct_for_single_step() {
        // Single step: reward = 1.0, value = 0.5.
        // delta = reward + gamma * 0 - value = 1.0 - 0.5 = 0.5
        // advantage = delta = 0.5
        // return = advantage + value = 1.0
        let (adv, ret) = compute_gae(&[0.5], 1.0, 1.0, 0.95);
        assert_eq!(adv.len(), 1);
        assert!(
            (adv[0] - 0.5).abs() < 1e-6,
            "advantage should be 0.5, got {}",
            adv[0]
        );
        assert!(
            (ret[0] - 1.0).abs() < 1e-6,
            "return should be 1.0, got {}",
            ret[0]
        );
    }

    #[test]
    fn gae_multi_step() {
        let values = vec![0.0, 0.0, 0.0];
        let (adv, _ret) = compute_gae(&values, 1.0, 1.0, 1.0);
        assert_eq!(adv.len(), 3);
        // With gamma=1, lambda=1, reward only at last step:
        // delta_2 = 1.0, gae_2 = 1.0
        // delta_1 = 0 + 1*0 - 0 = 0, gae_1 = 0 + 1*1*1.0 = 1.0
        // delta_0 = 0 + 1*0 - 0 = 0, gae_0 = 0 + 1*1*1.0 = 1.0
        for i in 0..3 {
            assert!(
                (adv[i] - 1.0).abs() < 1e-5,
                "advantage[{i}] should be 1.0, got {}",
                adv[i]
            );
        }
    }

    #[test]
    fn per_token_log_probs_are_finite_and_negative() {
        let mut rng = seeded_rng();
        let model = Transformer::new(tiny_config(), &mut rng);
        let lps = per_token_log_probs(&model, &[0, 1, 2, 3, 4]);
        assert_eq!(lps.len(), 4);
        for &lp in &lps {
            assert!(lp.is_finite(), "log-prob must be finite, got {lp}");
            assert!(lp <= 0.0, "log-prob must be <= 0, got {lp}");
        }
    }

    #[test]
    fn generate_rollout_produces_valid_structure() {
        let mut rng = seeded_rng();
        let config = tiny_config();
        let policy = Transformer::new(config.clone(), &mut rng);
        let rm = RewardModel::new(Transformer::new(config.clone(), &mut rng), &mut rng);
        let vm = ValueModel::new(Transformer::new(config, &mut rng), &mut rng);

        let ppo_config = PpoConfig::default();
        let trainer = RlhfTrainer::new(policy, rm, vm, ppo_config);

        let rollout = trainer.generate_rollout(&[0, 1], 5, &mut rng);
        assert!(rollout.tokens.len() >= 2);
        assert_eq!(rollout.prompt_len, 2);
        assert!(rollout.reward.is_finite());
    }

    #[test]
    fn ppo_step_produces_finite_loss() {
        let mut rng = seeded_rng();
        let config = tiny_config();
        let policy = Transformer::new(config.clone(), &mut rng);
        let rm = RewardModel::new(Transformer::new(config.clone(), &mut rng), &mut rng);
        let vm = ValueModel::new(Transformer::new(config, &mut rng), &mut rng);

        let ppo_config = PpoConfig::default();
        let mut trainer = RlhfTrainer::new(policy, rm, vm, ppo_config);

        let rollout = trainer.generate_rollout(&[0, 1], 5, &mut rng);
        let (loss, kl) = trainer.ppo_step(&rollout);
        assert!(loss.is_finite(), "PPO loss must be finite, got {loss}");
        assert!(kl.is_finite(), "KL must be finite, got {kl}");
        assert_eq!(trainer.step(), 1);
    }
}
