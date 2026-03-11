//! Logit processing and sampling strategies.
//!
//! Provides temperature scaling, top-k filtering, top-p (nucleus) filtering,
//! repetition penalty, and categorical sampling over logit vectors.

use rand::Rng;

/// Apply temperature scaling to logits in-place.
///
/// Divides all logits by `temperature`. Higher temperature → more uniform
/// distribution; lower temperature → more peaked.
///
/// # Panics
///
/// Panics if `temperature <= 0.0` or logits is empty.
pub fn temperature(logits: &mut [f32], t: f32) {
    assert!(!logits.is_empty(), "logits must not be empty");
    assert!(t > 0.0, "temperature must be positive, got {t}");
    for x in logits.iter_mut() {
        *x /= t;
    }
}

/// Keep only the top-k logits; set all others to negative infinity.
///
/// If `k >= logits.len()`, this is a no-op.
///
/// # Panics
///
/// Panics if `k == 0` or logits is empty.
pub fn top_k_filter(logits: &mut [f32], k: usize) {
    assert!(!logits.is_empty(), "logits must not be empty");
    assert!(k > 0, "k must be positive");
    if k >= logits.len() {
        return;
    }
    // Find the k-th largest value.
    let mut sorted: Vec<f32> = logits.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let threshold = sorted[k - 1];
    // Count how many values are >= threshold (handle ties).
    let mut kept = 0;
    for x in logits.iter_mut() {
        if *x >= threshold && kept < k {
            kept += 1;
        } else if *x < threshold {
            *x = f32::NEG_INFINITY;
        }
    }
    // If we kept more than k due to ties at threshold, remove extras.
    if kept > k {
        let mut extra = kept - k;
        for x in logits.iter_mut().rev() {
            if extra == 0 {
                break;
            }
            if (*x - threshold).abs() < f32::EPSILON {
                *x = f32::NEG_INFINITY;
                extra -= 1;
            }
        }
    }
}

/// Keep only the smallest set of logits whose cumulative probability exceeds `p`
/// (nucleus sampling). All others are set to negative infinity.
///
/// # Panics
///
/// Panics if `p <= 0.0`, `p > 1.0`, or logits is empty.
pub fn top_p_filter(logits: &mut [f32], p: f32) {
    assert!(!logits.is_empty(), "logits must not be empty");
    assert!(p > 0.0 && p <= 1.0, "p must be in (0, 1], got {p}");

    // Convert to probabilities via softmax.
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = probs.iter().sum();
    let probs: Vec<f32> = probs.iter().map(|&x| x / sum).collect();

    // Sort indices by probability (descending).
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

    // Find the cutoff: keep the smallest prefix whose sum >= p.
    let mut cumulative = 0.0;
    let mut keep = vec![false; logits.len()];
    for &idx in &indices {
        keep[idx] = true;
        cumulative += probs[idx];
        if cumulative >= p {
            break;
        }
    }

    for (i, x) in logits.iter_mut().enumerate() {
        if !keep[i] {
            *x = f32::NEG_INFINITY;
        }
    }
}

/// Apply repetition penalty to tokens that have appeared in `past_tokens`.
///
/// For each token in `past_tokens`, if its logit is positive, divide by `penalty`;
/// if negative, multiply by `penalty`. This reduces the probability of repeated tokens.
///
/// # Panics
///
/// Panics if `penalty < 1.0` or logits is empty.
pub fn repetition_penalty(logits: &mut [f32], past_tokens: &[u32], penalty: f32) {
    assert!(!logits.is_empty(), "logits must not be empty");
    assert!(
        penalty >= 1.0,
        "repetition penalty must be >= 1.0, got {penalty}"
    );
    if (penalty - 1.0).abs() < f32::EPSILON {
        return;
    }
    for &tok in past_tokens {
        let idx = tok as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Convert logits to probabilities via softmax, then sample from the distribution.
///
/// Returns the index of the sampled token.
///
/// # Panics
///
/// Panics if logits is empty.
pub fn softmax_sample(logits: &[f32], rng: &mut impl Rng) -> usize {
    assert!(!logits.is_empty(), "logits must not be empty");
    let probs = softmax(logits);
    categorical_sample(&probs, rng)
}

/// Return the index of the maximum logit (greedy decoding).
///
/// # Panics
///
/// Panics if logits is empty.
pub fn argmax(logits: &[f32]) -> usize {
    assert!(!logits.is_empty(), "logits must not be empty");
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0
}

/// Compute softmax probabilities from logits.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&x| x / sum).collect()
}

/// Sample from a categorical distribution given probabilities.
fn categorical_sample(probs: &[f32], rng: &mut impl Rng) -> usize {
    let u: f32 = rng.gen();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if u < cumulative {
            return i;
        }
    }
    probs.len() - 1
}

/// A composable logit processor that chains multiple strategies.
///
/// Processing order: repetition penalty → temperature → top-k → top-p.
pub struct LogitProcessor {
    temp: f32,
    k: Option<usize>,
    p: Option<f32>,
    rep_penalty: Option<f32>,
}

impl LogitProcessor {
    pub fn new() -> Self {
        Self {
            temp: 1.0,
            k: None,
            p: None,
            rep_penalty: None,
        }
    }

    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temp = t;
        self
    }

    pub fn with_top_k(mut self, k: usize) -> Self {
        self.k = Some(k);
        self
    }

    pub fn with_top_p(mut self, p: f32) -> Self {
        self.p = Some(p);
        self
    }

    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.rep_penalty = Some(penalty);
        self
    }

    /// Apply all configured processing strategies to logits in-place.
    pub fn process(&self, logits: &mut [f32], past_tokens: &[u32]) {
        if let Some(penalty) = self.rep_penalty {
            repetition_penalty(logits, past_tokens, penalty);
        }
        if (self.temp - 1.0).abs() > f32::EPSILON {
            temperature(logits, self.temp);
        }
        if let Some(k) = self.k {
            top_k_filter(logits, k);
        }
        if let Some(p) = self.p {
            top_p_filter(logits, p);
        }
    }

    /// Process logits and sample a token index.
    pub fn sample(&self, logits: &mut [f32], past_tokens: &[u32], rng: &mut impl Rng) -> usize {
        self.process(logits, past_tokens);
        softmax_sample(logits, rng)
    }
}

impl Default for LogitProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn seeded_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_temperature_identity() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        temperature(&mut logits, 1.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_temperature_high_more_uniform() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs_t1 = softmax(&logits);

        let mut scaled = logits.clone();
        temperature(&mut scaled, 10.0);
        let probs_t10 = softmax(&scaled);

        // Higher temperature → more uniform → lower max probability.
        let max_t1 = probs_t1.iter().cloned().fold(0.0f32, f32::max);
        let max_t10 = probs_t10.iter().cloned().fold(0.0f32, f32::max);
        assert!(max_t10 < max_t1, "high temp should flatten distribution");
    }

    #[test]
    fn test_temperature_low_more_peaked() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs_t1 = softmax(&logits);

        let mut scaled = logits.clone();
        temperature(&mut scaled, 0.1);
        let probs_t01 = softmax(&scaled);

        let max_t1 = probs_t1.iter().cloned().fold(0.0f32, f32::max);
        let max_t01 = probs_t01.iter().cloned().fold(0.0f32, f32::max);
        assert!(max_t01 > max_t1, "low temp should sharpen distribution");
    }

    #[test]
    fn test_top_k_keeps_k() {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        top_k_filter(&mut logits, 2);
        let kept: Vec<usize> = logits
            .iter()
            .enumerate()
            .filter(|(_, &x)| x.is_finite())
            .map(|(i, _)| i)
            .collect();
        assert_eq!(kept.len(), 2);
        assert!(kept.contains(&1)); // 5.0
        assert!(kept.contains(&4)); // 4.0
    }

    #[test]
    fn test_top_k_noop_when_k_ge_len() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        top_k_filter(&mut logits, 5);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_top_p_keeps_nucleus() {
        let mut logits = vec![10.0, 1.0, 0.0, -1.0]; // first element dominates
        top_p_filter(&mut logits, 0.9);
        // The highest-probability token should be kept.
        assert!(logits[0].is_finite());
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "probs must sum to 1, got {sum}");
    }

    #[test]
    fn test_top_p_keeps_all_when_p_1() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        top_p_filter(&mut logits, 1.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_repetition_penalty_reduces_seen() {
        let mut logits = vec![2.0, 3.0, 1.0, 4.0];
        let original = logits.clone();
        repetition_penalty(&mut logits, &[1, 3], 2.0);
        // Positive logits of seen tokens should decrease.
        assert!(logits[1] < original[1]);
        assert!(logits[3] < original[3]);
        // Unseen tokens unchanged.
        assert_eq!(logits[0], original[0]);
        assert_eq!(logits[2], original[2]);
    }

    #[test]
    fn test_repetition_penalty_negative_logit() {
        let mut logits = vec![-2.0, 1.0];
        repetition_penalty(&mut logits, &[0], 2.0);
        // Negative logit is multiplied → more negative.
        assert_eq!(logits[0], -4.0);
    }

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0, 1.0, 2.0]), 0);
    }

    #[test]
    fn test_softmax_sample_returns_valid_index() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let mut rng = seeded_rng();
        for _ in 0..100 {
            let idx = softmax_sample(&logits, &mut rng);
            assert!(idx < logits.len());
        }
    }

    #[test]
    fn test_softmax_valid_distribution() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "must sum to 1, got {sum}");
        for &p in &probs {
            assert!(p >= 0.0, "probabilities must be non-negative");
        }
    }

    #[test]
    fn test_all_strategies_valid_distribution() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.5, -1.0, 2.5];
        let processor = LogitProcessor::new()
            .with_temperature(0.8)
            .with_top_k(5)
            .with_top_p(0.9)
            .with_repetition_penalty(1.2);
        processor.process(&mut logits, &[0, 2, 4]);
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "must sum to 1, got {sum}");
        for &p in &probs {
            assert!(p >= 0.0, "probabilities must be non-negative");
        }
    }

    #[test]
    fn test_processor_sample_valid() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        let processor = LogitProcessor::new().with_temperature(0.5).with_top_k(2);
        let mut rng = seeded_rng();
        let idx = processor.sample(&mut logits, &[], &mut rng);
        assert!(idx < 4);
    }

    #[test]
    #[should_panic(expected = "temperature must be positive")]
    fn test_temperature_zero_panics() {
        temperature(&mut [1.0], 0.0);
    }

    #[test]
    #[should_panic(expected = "k must be positive")]
    fn test_top_k_zero_panics() {
        top_k_filter(&mut [1.0], 0);
    }

    #[test]
    #[should_panic(expected = "p must be in (0, 1]")]
    fn test_top_p_zero_panics() {
        top_p_filter(&mut [1.0], 0.0);
    }

    #[test]
    #[should_panic(expected = "repetition penalty must be >= 1.0")]
    fn test_rep_penalty_below_one_panics() {
        repetition_penalty(&mut [1.0], &[0], 0.5);
    }
}
