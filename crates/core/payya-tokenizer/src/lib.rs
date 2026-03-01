//! Byte-Pair Encoding (BPE) tokenizer.
//!
//! Implements BPE from first principles:
//! - **Training**: learn a vocabulary of merges from a text corpus.
//! - **Encoding**: tokenize text into token IDs using learned merges.
//! - **Decoding**: convert token IDs back to text.
//! - **GPT-2 compatibility**: load an existing GPT-2 vocabulary and merges file.
//!
//! # Example
//!
//! ```
//! use payya_tokenizer::Tokenizer;
//!
//! let corpus = "low lower newest widest";
//! let tokenizer = Tokenizer::train(corpus, 276);
//! let ids = tokenizer.encode("lower");
//! let text = tokenizer.decode(&ids);
//! assert_eq!(text, "lower");
//! ```

use std::collections::HashMap;

mod gpt2;
mod train;

/// A token ID. Tokens 0–255 are individual bytes; tokens >= 256 are learned merges.
pub type TokenId = u32;

/// A single merge rule: replace adjacent `(left, right)` with `merged`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MergeRule {
    pub left: TokenId,
    pub right: TokenId,
    pub merged: TokenId,
}

/// A trained BPE tokenizer.
///
/// The vocabulary always starts with 256 byte-level tokens (0–255).
/// Merge rules are applied in priority order (index 0 = highest priority).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Tokenizer {
    /// Merge rules in priority order. `merges[0]` is applied first.
    merges: Vec<MergeRule>,
    /// Maps a (left, right) pair to the merged token ID, for O(1) lookup during encoding.
    #[serde(skip)]
    merge_map: HashMap<(TokenId, TokenId), TokenId>,
    /// Maps token ID → byte sequence, for decoding.
    vocab: HashMap<TokenId, Vec<u8>>,
}

impl Tokenizer {
    /// The number of byte-level base tokens (0–255).
    pub const BASE_VOCAB_SIZE: usize = 256;

    /// Train a BPE tokenizer on the given `corpus` text.
    ///
    /// `vocab_size` is the desired total vocabulary size (must be > 256).
    /// The tokenizer starts with 256 byte tokens and learns `vocab_size - 256`
    /// merge rules from the corpus.
    ///
    /// # Panics
    ///
    /// Panics if `vocab_size <= 256` (no merges to learn) or if `corpus` is empty.
    pub fn train(corpus: &str, vocab_size: usize) -> Self {
        assert!(
            vocab_size > Self::BASE_VOCAB_SIZE,
            "vocab_size={} must be > {} (base byte tokens)",
            vocab_size,
            Self::BASE_VOCAB_SIZE
        );
        assert!(!corpus.is_empty(), "corpus must not be empty");
        train::train_bpe(corpus, vocab_size)
    }

    /// Build a tokenizer from an explicit list of merge rules.
    ///
    /// Merge rules are applied in the order given (index 0 = highest priority).
    /// The base vocabulary (bytes 0–255) is always included.
    ///
    /// # Panics
    ///
    /// Panics if any merge rule references a token ID that doesn't exist at the
    /// point it would be applied (i.e., the merge list is not topologically valid).
    pub fn from_merges(merges: Vec<MergeRule>) -> Self {
        let mut vocab: HashMap<TokenId, Vec<u8>> = HashMap::new();
        for b in 0u16..=255 {
            vocab.insert(b as TokenId, vec![b as u8]);
        }

        let mut merge_map = HashMap::new();
        for rule in &merges {
            assert!(
                vocab.contains_key(&rule.left),
                "merge rule references unknown left token {}",
                rule.left
            );
            assert!(
                vocab.contains_key(&rule.right),
                "merge rule references unknown right token {}",
                rule.right
            );
            let mut bytes = vocab[&rule.left].clone();
            bytes.extend_from_slice(&vocab[&rule.right]);
            vocab.insert(rule.merged, bytes);
            merge_map.insert((rule.left, rule.right), rule.merged);
        }

        Self {
            merges,
            merge_map,
            vocab,
        }
    }

    /// The total vocabulary size (256 base tokens + number of merges).
    pub fn vocab_size(&self) -> usize {
        Self::BASE_VOCAB_SIZE + self.merges.len()
    }

    /// Encode a string into a sequence of token IDs.
    ///
    /// The input is first converted to bytes (tokens 0–255), then merge rules
    /// are applied iteratively in priority order until no more merges apply.
    pub fn encode(&self, text: &str) -> Vec<TokenId> {
        if text.is_empty() {
            return Vec::new();
        }

        // Start with byte-level tokens.
        let mut tokens: Vec<TokenId> = text.bytes().map(|b| b as TokenId).collect();

        // Apply merges in priority order. Each merge pass scans the token list
        // and replaces all adjacent (left, right) pairs that match the current
        // merge rule. We iterate through all merge rules in order; for each rule,
        // we do a single linear scan.
        for rule in &self.merges {
            if tokens.len() < 2 {
                break;
            }
            tokens = merge_pass(&tokens, rule.left, rule.right, rule.merged);
        }

        tokens
    }

    /// Decode a sequence of token IDs back to a string.
    ///
    /// Each token ID is mapped to its byte sequence and the results are
    /// concatenated. The resulting bytes are interpreted as UTF-8 (with
    /// lossy replacement for invalid sequences).
    pub fn decode(&self, ids: &[TokenId]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            match self.vocab.get(&id) {
                Some(b) => bytes.extend_from_slice(b),
                None => panic!("unknown token ID {}", id),
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Look up the byte sequence for a token ID.
    pub fn token_bytes(&self, id: TokenId) -> Option<&[u8]> {
        self.vocab.get(&id).map(|v| v.as_slice())
    }

    /// Return the merge rules in priority order.
    pub fn merges(&self) -> &[MergeRule] {
        &self.merges
    }

    /// Serialize the tokenizer to a JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("tokenizer serialization should not fail")
    }

    /// Deserialize a tokenizer from a JSON string.
    ///
    /// # Panics
    ///
    /// Panics if the JSON is invalid or does not represent a valid tokenizer.
    pub fn from_json(json: &str) -> Self {
        let mut tok: Self = serde_json::from_str(json).expect("invalid tokenizer JSON");
        // Rebuild transient fields.
        tok.rebuild_caches();
        tok
    }

    /// Rebuild the merge_map and vocab caches from the merge rules.
    /// Called after deserialization.
    fn rebuild_caches(&mut self) {
        self.vocab.clear();
        for b in 0u16..=255 {
            self.vocab.insert(b as TokenId, vec![b as u8]);
        }
        self.merge_map.clear();
        for rule in &self.merges {
            let mut bytes = self.vocab[&rule.left].clone();
            bytes.extend_from_slice(&self.vocab[&rule.right]);
            self.vocab.insert(rule.merged, bytes);
            self.merge_map.insert((rule.left, rule.right), rule.merged);
        }
    }
}

/// Single pass over `tokens`, replacing all adjacent `(left, right)` with `merged`.
fn merge_pass(tokens: &[TokenId], left: TokenId, right: TokenId, merged: TokenId) -> Vec<TokenId> {
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
    fn byte_level_identity() {
        // With no merges, encoding produces byte-level tokens.
        let tok = Tokenizer::from_merges(vec![]);
        assert_eq!(tok.vocab_size(), 256);
        let ids = tok.encode("abc");
        assert_eq!(ids, vec![97, 98, 99]);
        assert_eq!(tok.decode(&ids), "abc");
    }

    #[test]
    fn empty_input() {
        let tok = Tokenizer::from_merges(vec![]);
        assert_eq!(tok.encode(""), Vec::<TokenId>::new());
        assert_eq!(tok.decode(&[]), "");
    }

    #[test]
    fn single_merge() {
        // Merge 'a' (97) + 'b' (98) → 256
        let tok = Tokenizer::from_merges(vec![MergeRule {
            left: 97,
            right: 98,
            merged: 256,
        }]);
        assert_eq!(tok.vocab_size(), 257);
        let ids = tok.encode("ab");
        assert_eq!(ids, vec![256]);
        assert_eq!(tok.decode(&ids), "ab");
    }

    #[test]
    fn chained_merges() {
        // Merge 'a'+'b' → 256, then 256+'c' → 257
        let tok = Tokenizer::from_merges(vec![
            MergeRule {
                left: 97,
                right: 98,
                merged: 256,
            },
            MergeRule {
                left: 256,
                right: 99,
                merged: 257,
            },
        ]);
        let ids = tok.encode("abc");
        assert_eq!(ids, vec![257]);
        assert_eq!(tok.decode(&ids), "abc");
    }

    #[test]
    fn repeated_pair() {
        // "abab" with merge 'a'+'b' → 256 should produce [256, 256]
        let tok = Tokenizer::from_merges(vec![MergeRule {
            left: 97,
            right: 98,
            merged: 256,
        }]);
        let ids = tok.encode("abab");
        assert_eq!(ids, vec![256, 256]);
        assert_eq!(tok.decode(&ids), "abab");
    }

    #[test]
    fn merge_pass_basic() {
        let tokens = vec![1, 2, 1, 2, 3];
        let result = merge_pass(&tokens, 1, 2, 99);
        assert_eq!(result, vec![99, 99, 3]);
    }

    #[test]
    fn round_trip_ascii() {
        let tok = Tokenizer::from_merges(vec![]);
        for s in &["hello world", "rust is great", "BPE tokenizer 123!@#"] {
            assert_eq!(tok.decode(&tok.encode(s)), *s);
        }
    }

    #[test]
    fn round_trip_utf8() {
        let tok = Tokenizer::from_merges(vec![]);
        for s in &["café", "日本語", "🦀 Rust", "über"] {
            assert_eq!(tok.decode(&tok.encode(s)), *s);
        }
    }

    #[test]
    fn json_round_trip() {
        let tok = Tokenizer::from_merges(vec![
            MergeRule {
                left: 97,
                right: 98,
                merged: 256,
            },
            MergeRule {
                left: 256,
                right: 99,
                merged: 257,
            },
        ]);
        let json = tok.to_json();
        let tok2 = Tokenizer::from_json(&json);
        assert_eq!(tok2.encode("abc"), vec![257]);
        assert_eq!(tok2.decode(&[257]), "abc");
    }

    #[test]
    #[should_panic(expected = "unknown token ID")]
    fn decode_unknown_token() {
        let tok = Tokenizer::from_merges(vec![]);
        tok.decode(&[9999]);
    }

    #[test]
    #[should_panic(expected = "merge rule references unknown left token")]
    fn invalid_merge_order() {
        // Token 257 doesn't exist yet, so this should panic.
        Tokenizer::from_merges(vec![MergeRule {
            left: 257,
            right: 98,
            merged: 258,
        }]);
    }
}
