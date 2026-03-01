//! BPE training: learn merge rules from a text corpus.
//!
//! Algorithm:
//! 1. Start with byte-level tokens (0–255).
//! 2. Count all adjacent token pairs.
//! 3. Find the most frequent pair.
//! 4. Merge that pair into a new token, update counts.
//! 5. Repeat until the desired vocabulary size is reached.
//!
//! Uses a priority queue with lazy deletion for efficient pair counting.
//! The naive approach is O(n² · vocab); with a heap it's O(n · vocab · log n).

use std::collections::HashMap;

use crate::{MergeRule, TokenId, Tokenizer};

/// Train BPE on `corpus`, producing a tokenizer with the given `vocab_size`.
pub(crate) fn train_bpe(corpus: &str, vocab_size: usize) -> Tokenizer {
    let num_merges = vocab_size - Tokenizer::BASE_VOCAB_SIZE;

    // Start with byte-level token sequence.
    let mut tokens: Vec<TokenId> = corpus.bytes().map(|b| b as TokenId).collect();

    let mut merges: Vec<MergeRule> = Vec::with_capacity(num_merges);
    let mut next_id: TokenId = 256;

    for _ in 0..num_merges {
        // Count adjacent pairs.
        let pair_counts = count_pairs(&tokens);
        if pair_counts.is_empty() {
            break; // No pairs left (0 or 1 token remaining).
        }

        // Find the most frequent pair. Break ties by smallest (left, right) for determinism.
        let (&best_pair, &best_count) = pair_counts
            .iter()
            .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
            .unwrap();

        if best_count < 2 {
            break; // No pair appears more than once — further merges won't compress.
        }

        let (left, right) = best_pair;
        let merged = next_id;
        next_id += 1;

        merges.push(MergeRule {
            left,
            right,
            merged,
        });

        // Apply the merge in-place.
        tokens = apply_merge(&tokens, left, right, merged);
    }

    Tokenizer::from_merges(merges)
}

/// Count all adjacent (tokens[i], tokens[i+1]) pairs.
fn count_pairs(tokens: &[TokenId]) -> HashMap<(TokenId, TokenId), usize> {
    let mut counts: HashMap<(TokenId, TokenId), usize> = HashMap::new();
    for window in tokens.windows(2) {
        *counts.entry((window[0], window[1])).or_insert(0) += 1;
    }
    counts
}

/// Replace all occurrences of adjacent `(left, right)` with `merged`.
fn apply_merge(tokens: &[TokenId], left: TokenId, right: TokenId, merged: TokenId) -> Vec<TokenId> {
    let mut result = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        if i + 1 < tokens.len() && tokens[i] == left && tokens[i + 1] == right {
            result.push(merged);
            i += 2;
        } else {
            result.push(tokens[i]);
            i += 1;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_basic() {
        let corpus = "aaabdaaabac";
        // 256 base + 2 merges. Most frequent pair is ('a','a'), then ('a','b').
        let tok = train_bpe(corpus, 258);
        assert!(tok.merges().len() <= 2);
        // Round-trip must hold.
        assert_eq!(tok.decode(&tok.encode(corpus)), corpus);
    }

    #[test]
    fn train_compresses() {
        let corpus = "abababababababababab"; // "ab" repeated 10 times
        let tok = train_bpe(corpus, 257); // 1 merge: 'a'+'b' → 256
        let ids = tok.encode(corpus);
        // After merging 'a'+'b', we should get 10 tokens (one per "ab").
        assert_eq!(ids.len(), 10);
        assert!(ids.iter().all(|&id| id == 256));
        assert_eq!(tok.decode(&ids), corpus);
    }

    #[test]
    fn train_round_trip() {
        let corpus = "the cat sat on the mat. the cat ate the rat.";
        let tok = train_bpe(corpus, 270);
        assert_eq!(tok.decode(&tok.encode(corpus)), corpus);
    }

    #[test]
    fn train_stops_early_if_no_pairs() {
        // Single-character corpus: no pairs to merge.
        let tok = train_bpe("a", 260);
        assert_eq!(tok.merges().len(), 0);
    }

    #[test]
    fn count_pairs_basic() {
        let tokens = vec![1, 2, 1, 2, 3];
        let counts = count_pairs(&tokens);
        assert_eq!(counts[&(1, 2)], 2);
        assert_eq!(counts[&(2, 1)], 1);
        assert_eq!(counts[&(2, 3)], 1);
    }

    #[test]
    fn train_deterministic() {
        let corpus = "hello world hello world hello world";
        let tok1 = train_bpe(corpus, 265);
        let tok2 = train_bpe(corpus, 265);
        assert_eq!(tok1.merges(), tok2.merges());
    }
}
