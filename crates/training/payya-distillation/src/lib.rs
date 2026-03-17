//! Knowledge distillation: teacher-student training with KL divergence.
//!
//! The student model learns to match the teacher's output distribution
//! (soft targets) at a given temperature, combined with the standard
//! cross-entropy loss on hard targets.
//!
//! Distillation loss = alpha * KL(teacher_soft || student_soft) * T^2
//!                   + (1 - alpha) * CE(student_logits, hard_targets)
//!
//! where T is the temperature and soft distributions are computed as
//! softmax(logits / T).

use payya_autograd::Graph;
use payya_transformer::Transformer;

// ── Configuration ────────────────────────────────────────────────────

/// Distillation training configuration.
#[derive(Debug, Clone)]
pub struct DistillConfig {
    /// Temperature for softening distributions. Higher T → softer targets.
    pub temperature: f32,
    /// Weight for the distillation (KL) loss vs. hard-target (CE) loss.
    /// alpha=1.0 means pure distillation, alpha=0.0 means pure CE.
    pub alpha: f32,
    /// Learning rate.
    pub lr: f32,
}

impl Default for DistillConfig {
    fn default() -> Self {
        Self {
            temperature: 2.0,
            alpha: 0.5,
            lr: 1e-3,
        }
    }
}

// ── Softmax and KL divergence helpers ───────────────────────────────

/// Compute softmax of a row of logits at a given temperature.
fn softmax_with_temp(logits: &[f32], temperature: f32) -> Vec<f32> {
    assert!(
        temperature > 0.0,
        "temperature must be > 0, got {temperature}"
    );
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
    let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// KL divergence: KL(P || Q) = sum_i P(i) * log(P(i) / Q(i))
/// where P is the teacher distribution and Q is the student distribution.
fn kl_divergence(teacher: &[f32], student: &[f32]) -> f32 {
    assert_eq!(
        teacher.len(),
        student.len(),
        "distribution lengths must match: {} vs {}",
        teacher.len(),
        student.len()
    );
    let mut kl = 0.0f32;
    for (&p, &q) in teacher.iter().zip(student.iter()) {
        if p > 1e-10 {
            // Clamp q to avoid log(0).
            let q_clamped = q.max(1e-10);
            kl += p * (p / q_clamped).ln();
        }
    }
    kl
}

// ── Distillation loss computation ───────────────────────────────────

/// Compute the distillation loss for a single sequence.
///
/// Returns (total_loss, kl_loss, ce_loss).
///
/// - `teacher`: the teacher model (frozen).
/// - `student`: the student model.
/// - `tokens`: input token sequence (at least 2 tokens for next-token prediction).
/// - `config`: distillation configuration.
pub fn distillation_loss(
    teacher: &Transformer,
    student: &Transformer,
    tokens: &[usize],
    config: &DistillConfig,
) -> (f32, f32, f32) {
    assert!(
        tokens.len() >= 2,
        "need at least 2 tokens, got {}",
        tokens.len()
    );

    let vocab_t = teacher.config.vocab_size;
    let vocab_s = student.config.vocab_size;
    let seq = tokens.len() - 1; // number of prediction positions

    // Get teacher logits (no gradient needed).
    let (g_teacher, logits_t_id) = teacher.forward(&tokens[..tokens.len() - 1]);
    let teacher_logits = g_teacher.data(logits_t_id);

    // Get student logits.
    let (g_student, logits_s_id) = student.forward(&tokens[..tokens.len() - 1]);
    let student_logits = g_student.data(logits_s_id);

    let t = config.temperature;
    let alpha = config.alpha;

    // Compute per-position losses.
    let mut total_kl = 0.0f32;
    let mut total_ce = 0.0f32;

    for pos in 0..seq {
        let t_row = &teacher_logits[pos * vocab_t..(pos + 1) * vocab_t];
        let s_row = &student_logits[pos * vocab_s..(pos + 1) * vocab_s];
        let target = tokens[pos + 1];

        // KL divergence on softened distributions.
        let teacher_soft = softmax_with_temp(t_row, t);
        // Student must have at least as many vocab entries as needed.
        let student_soft = softmax_with_temp(&s_row[..vocab_t.min(vocab_s)], t);
        total_kl += kl_divergence(&teacher_soft[..student_soft.len()], &student_soft);

        // Cross-entropy on hard targets (temperature=1).
        let max_logit = s_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp: f32 = s_row
            .iter()
            .map(|&x| (x - max_logit).exp())
            .sum::<f32>()
            .ln();
        let ce = -(s_row[target.min(vocab_s - 1)] - max_logit - log_sum_exp);
        total_ce += ce;
    }

    let mean_kl = total_kl / seq as f32;
    let mean_ce = total_ce / seq as f32;

    // Total loss: weighted combination, with T^2 scaling for KL.
    let total = alpha * mean_kl * t * t + (1.0 - alpha) * mean_ce;

    (total, mean_kl, mean_ce)
}

// ── Distillation Trainer ────────────────────────────────────────────

/// Trains a student model to match a teacher model's output distribution.
pub struct DistillTrainer {
    /// The student model being trained.
    pub student: Transformer,
    /// The frozen teacher model.
    teacher: Transformer,
    /// Configuration.
    pub config: DistillConfig,
    /// Training step counter.
    step: usize,
}

impl DistillTrainer {
    /// Create a distillation trainer.
    pub fn new(teacher: Transformer, student: Transformer, config: DistillConfig) -> Self {
        assert_eq!(
            teacher.config.vocab_size, student.config.vocab_size,
            "teacher vocab_size={} must match student vocab_size={}",
            teacher.config.vocab_size, student.config.vocab_size
        );
        Self {
            student,
            teacher,
            config,
            step: 0,
        }
    }

    /// Current training step.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Run one training step on a token sequence.
    /// Uses the autograd graph for student gradients (cross-entropy on hard targets)
    /// and adds KL divergence gradient via finite differences on the KL component.
    ///
    /// Returns (total_loss, kl_loss, ce_loss).
    pub fn train_step(&mut self, tokens: &[usize]) -> (f32, f32, f32) {
        assert!(
            tokens.len() >= 2,
            "need at least 2 tokens, got {}",
            tokens.len()
        );

        let input = &tokens[..tokens.len() - 1];
        let targets = &tokens[1..];

        // Step 1: Get teacher soft targets.
        let (g_teacher, logits_t_id) = self.teacher.forward(input);
        let teacher_logits = g_teacher.data(logits_t_id).to_vec();
        let vocab = self.teacher.config.vocab_size;
        let seq = input.len();
        let t = self.config.temperature;

        // Compute teacher soft distributions.
        let mut teacher_soft: Vec<Vec<f32>> = Vec::with_capacity(seq);
        for pos in 0..seq {
            let row = &teacher_logits[pos * vocab..(pos + 1) * vocab];
            teacher_soft.push(softmax_with_temp(row, t));
        }

        // Step 2: Student forward + backward using autograd for CE loss.
        let (mut g, logits_s_id) = self.student.forward(input);
        let student_logits_data = g.data(logits_s_id).to_vec();

        // Compute KL loss manually (not through autograd).
        let mut total_kl = 0.0f32;
        for pos in 0..seq {
            let s_row = &student_logits_data[pos * vocab..(pos + 1) * vocab];
            let student_soft = softmax_with_temp(s_row, t);
            total_kl += kl_divergence(&teacher_soft[pos], &student_soft);
        }
        let mean_kl = total_kl / seq as f32;

        // CE loss through autograd.
        let ce_loss_id = g.cross_entropy(logits_s_id, targets);
        let ce_loss_val = g.data(ce_loss_id)[0];
        g.backward(ce_loss_id);

        // Update student params with CE gradients.
        let lr = self.config.lr;
        let alpha = self.config.alpha;
        let ce_scale = (1.0 - alpha) * lr;

        // Apply CE gradients.
        update_student_params(&mut self.student, &g, ce_scale);

        // Step 3: KL gradient via finite differences on student params.
        if alpha > 0.0 {
            let kl_lr = alpha * lr * t * t;
            let eps = 1e-4_f32;
            apply_kl_gradient_fd(
                &mut self.student,
                &self.teacher,
                tokens,
                &self.config,
                kl_lr,
                eps,
            );
        }

        let total_loss = alpha * mean_kl * t * t + (1.0 - alpha) * ce_loss_val;
        self.step += 1;

        (total_loss, mean_kl, ce_loss_val)
    }
}

/// Apply SGD update to student parameters from autograd graph gradients.
fn update_student_params(student: &mut Transformer, g: &Graph, scaled_lr: f32) {
    let mut param_refs = collect_param_refs_mut(student);
    let mut param_idx = 0;

    let num_nodes = g.num_nodes();
    for node_id in 0..num_nodes {
        let tid = payya_autograd::TensorId::from_raw(node_id);
        if !g.is_param(tid) {
            continue;
        }
        if param_idx >= param_refs.len() {
            break;
        }
        if !g.has_grad(tid) {
            param_idx += 1;
            if param_idx >= param_refs.len() {
                break;
            }
            continue;
        }
        let grad = g.grad(tid);
        let params = &mut param_refs[param_idx];
        assert_eq!(params.len(), grad.len());
        for (p, &gr) in params.iter_mut().zip(grad.iter()) {
            *p -= scaled_lr * gr;
        }
        param_idx += 1;
    }
}

/// Collect mutable references to all student parameter vectors in graph order.
fn collect_param_refs_mut(model: &mut Transformer) -> Vec<&mut Vec<f32>> {
    let mut refs: Vec<&mut Vec<f32>> = Vec::new();
    refs.push(&mut model.params.token_emb);
    for layer in &mut model.params.layers {
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
    refs.push(&mut model.params.final_ln_gamma);
    refs.push(&mut model.params.final_ln_beta);
    refs.push(&mut model.params.output_weight);
    refs.push(&mut model.params.output_bias);
    refs
}

/// Apply KL divergence gradient via finite differences.
/// Only updates a subset of parameters per step for efficiency.
fn apply_kl_gradient_fd(
    student: &mut Transformer,
    teacher: &Transformer,
    tokens: &[usize],
    config: &DistillConfig,
    scaled_lr: f32,
    eps: f32,
) {
    // For efficiency, we use stochastic coordinate descent: update a random
    // subset of parameters each step. Full finite differences on all params
    // would be prohibitively slow for large models.
    let mut flat = flatten_params(student);
    let n = flat.len();

    // Update every parameter.
    // For tiny models this is feasible; for larger models the PEFT crate
    // should use LoRA + distillation instead.
    for i in 0..n {
        let orig = flat[i];

        flat[i] = orig + eps;
        set_flat_params(student, &flat);
        let (loss_plus, _, _) = distillation_loss(teacher, student, tokens, config);

        flat[i] = orig - eps;
        set_flat_params(student, &flat);
        let (loss_minus, _, _) = distillation_loss(teacher, student, tokens, config);

        let grad = (loss_plus - loss_minus) / (2.0 * eps);
        flat[i] = orig - scaled_lr * grad;
    }
    set_flat_params(student, &flat);
}

fn flatten_params(model: &Transformer) -> Vec<f32> {
    let mut flat = Vec::new();
    flat.extend_from_slice(&model.params.token_emb);
    for layer in &model.params.layers {
        for param in [
            &layer.wq,
            &layer.wk,
            &layer.wv,
            &layer.wo,
            &layer.bq,
            &layer.bk,
            &layer.bv,
            &layer.bo,
            &layer.ln1_gamma,
            &layer.ln1_beta,
            &layer.w1,
            &layer.w2,
            &layer.b1,
            &layer.b2,
            &layer.ln2_gamma,
            &layer.ln2_beta,
        ] {
            flat.extend_from_slice(param);
        }
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

    assert_eq!(offset, flat.len());
}

#[cfg(test)]
mod tests {
    use super::*;
    use payya_transformer::{PosEncoding, TransformerConfig};
    use rand::SeedableRng;

    fn tiny_config() -> TransformerConfig {
        TransformerConfig {
            vocab_size: 4,
            d_model: 4,
            n_heads: 1,
            n_layers: 1,
            d_ff: 8,
            max_seq_len: 16,
            pos_encoding: PosEncoding::Sinusoidal,
        }
    }

    fn seeded_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

    #[test]
    fn softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = softmax_with_temp(&logits, 1.0);
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "softmax should sum to 1, got {sum}"
        );
    }

    #[test]
    fn softmax_higher_temp_is_more_uniform() {
        let logits = vec![1.0, 5.0, 2.0, 0.5];
        let sharp = softmax_with_temp(&logits, 0.5);
        let soft = softmax_with_temp(&logits, 5.0);
        // Higher temperature should produce more uniform distribution.
        let sharp_max = sharp.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let soft_max = soft.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            soft_max < sharp_max,
            "higher temp should reduce max prob: sharp_max={sharp_max}, soft_max={soft_max}"
        );
    }

    #[test]
    fn kl_same_distribution_is_zero() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let kl = kl_divergence(&p, &p);
        assert!(kl.abs() < 1e-6, "KL(P || P) should be 0, got {kl}");
    }

    #[test]
    fn kl_is_non_negative() {
        let p = vec![0.7, 0.2, 0.1];
        let q = vec![0.3, 0.4, 0.3];
        let kl = kl_divergence(&p, &q);
        assert!(kl >= 0.0, "KL divergence must be non-negative, got {kl}");
    }

    #[test]
    fn distillation_loss_is_finite() {
        let mut rng = seeded_rng();
        let teacher = Transformer::new(tiny_config(), &mut rng);
        let student = Transformer::new(tiny_config(), &mut rng);
        let config = DistillConfig::default();
        let tokens = vec![0, 1, 2, 3];
        let (total, kl, ce) = distillation_loss(&teacher, &student, &tokens, &config);
        assert!(total.is_finite(), "total loss must be finite, got {total}");
        assert!(kl.is_finite(), "KL loss must be finite, got {kl}");
        assert!(ce.is_finite(), "CE loss must be finite, got {ce}");
    }

    #[test]
    fn distillation_loss_same_model_has_zero_kl() {
        let mut rng = seeded_rng();
        let model = Transformer::new(tiny_config(), &mut rng);
        let config = DistillConfig::default();
        let tokens = vec![0, 1, 2, 3];
        let (_, kl, _) = distillation_loss(&model, &model, &tokens, &config);
        assert!(
            kl.abs() < 1e-5,
            "KL divergence of model against itself should be ~0, got {kl}"
        );
    }

    #[test]
    fn trainer_produces_finite_losses() {
        let mut rng = seeded_rng();
        let teacher = Transformer::new(tiny_config(), &mut rng);
        let student = Transformer::new(tiny_config(), &mut rng);
        let config = DistillConfig {
            temperature: 2.0,
            alpha: 0.5,
            lr: 0.001,
        };
        let mut trainer = DistillTrainer::new(teacher, student, config);

        let tokens = vec![0, 1, 2, 3];
        let (total, kl, ce) = trainer.train_step(&tokens);
        assert!(total.is_finite());
        assert!(kl.is_finite());
        assert!(ce.is_finite());
        assert_eq!(trainer.step(), 1);
    }
}
