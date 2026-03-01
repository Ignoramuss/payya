//! GPT-2 vocabulary and merges file loading.
//!
//! GPT-2's BPE uses a byte-to-unicode mapping where each byte is represented
//! by a printable Unicode character. The merges file lists pairs of these
//! unicode-encoded tokens, one per line.
//!
//! File formats:
//! - `vocab.json`: `{"token_string": token_id, ...}`
//! - `merges.txt`: header line, then one `"left right"` pair per line.

use std::collections::HashMap;

use crate::{MergeRule, TokenId, Tokenizer};

impl Tokenizer {
    /// Load a GPT-2-style tokenizer from a `vocab.json` string and a `merges.txt` string.
    ///
    /// `vocab_json` maps unicode-encoded token strings to integer IDs.
    /// `merges_txt` has a header line followed by merge rules like `"Ġ t" -> merge "Ġ" and "t"`.
    ///
    /// # Panics
    ///
    /// Panics if the vocab JSON is invalid, if a merge references unknown tokens,
    /// or if the files are inconsistent.
    pub fn from_gpt2(vocab_json: &str, merges_txt: &str) -> Self {
        let str_to_id: HashMap<String, TokenId> =
            serde_json::from_str(vocab_json).expect("invalid vocab.json");

        // Build the inverse: id → string.
        let id_to_str: HashMap<TokenId, String> =
            str_to_id.iter().map(|(s, &id)| (id, s.clone())).collect();

        // Build the byte-to-unicode and unicode-to-byte maps.
        let byte_to_unicode = gpt2_byte_to_unicode();
        let unicode_to_byte: HashMap<char, u8> =
            byte_to_unicode.iter().map(|(&b, &c)| (c, b)).collect();

        // Build vocab: token_id → byte sequence.
        let mut vocab: HashMap<TokenId, Vec<u8>> = HashMap::new();
        for (&id, token_str) in &id_to_str {
            let bytes: Vec<u8> = token_str
                .chars()
                .map(|c| {
                    *unicode_to_byte.get(&c).unwrap_or_else(|| {
                        panic!(
                            "unknown unicode char {:?} (U+{:04X}) in token {:?} (id={})",
                            c, c as u32, token_str, id
                        )
                    })
                })
                .collect();
            vocab.insert(id, bytes);
        }

        // Parse merges.
        let mut merges = Vec::new();
        let mut merge_map = HashMap::new();
        for line in merges_txt.lines().skip(1) {
            // Skip the "#version:" header line.
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.splitn(2, ' ').collect();
            assert!(
                parts.len() == 2,
                "invalid merge line (expected 'left right'): {:?}",
                line
            );
            let left_str = parts[0];
            let right_str = parts[1];

            let left_id = *str_to_id
                .get(left_str)
                .unwrap_or_else(|| panic!("merge references unknown left token {:?}", left_str));
            let right_id = *str_to_id
                .get(right_str)
                .unwrap_or_else(|| panic!("merge references unknown right token {:?}", right_str));

            let merged_str = format!("{}{}", left_str, right_str);
            let merged_id = *str_to_id.get(&merged_str).unwrap_or_else(|| {
                panic!(
                    "merge result {:?} not found in vocab (left={:?}, right={:?})",
                    merged_str, left_str, right_str
                )
            });

            merges.push(MergeRule {
                left: left_id,
                right: right_id,
                merged: merged_id,
            });
            merge_map.insert((left_id, right_id), merged_id);
        }

        Self {
            merges,
            merge_map,
            vocab,
        }
    }

    /// Encode text using GPT-2's pre-tokenization (byte-level BPE).
    ///
    /// GPT-2 first converts bytes to unicode characters, then applies BPE merges.
    /// This method handles that encoding transparently.
    pub fn encode_gpt2(&self, text: &str) -> Vec<TokenId> {
        if text.is_empty() {
            return Vec::new();
        }

        let byte_to_unicode = gpt2_byte_to_unicode();

        // Convert text bytes → GPT-2 unicode characters → find base token IDs.
        // Then apply merges.
        let unicode_chars: Vec<char> = text
            .bytes()
            .map(|b| *byte_to_unicode.get(&b).expect("byte not in GPT-2 map"))
            .collect();

        // Map each unicode char to its token ID (single-char tokens in the vocab).
        let mut tokens: Vec<TokenId> = Vec::with_capacity(unicode_chars.len());
        for ch in &unicode_chars {
            // Find the token ID for this single character.
            let id = self
                .vocab
                .iter()
                .find(|(_, bytes)| {
                    bytes.len() == 1 && gpt2_byte_to_unicode().get(&bytes[0]) == Some(ch)
                })
                .map(|(&id, _)| id)
                .unwrap_or_else(|| panic!("no vocab entry for GPT-2 char {:?}", ch));
            tokens.push(id);
        }

        // Apply merges in priority order.
        for rule in &self.merges {
            if tokens.len() < 2 {
                break;
            }
            tokens = crate::merge_pass(&tokens, rule.left, rule.right, rule.merged);
        }

        tokens
    }
}

/// GPT-2's byte-to-unicode mapping.
///
/// GPT-2 maps each byte to a unique Unicode character so that the BPE vocabulary
/// consists entirely of printable strings. Bytes that are already printable
/// (33–126, 161–172, 174–255) map to themselves as Unicode code points.
/// The remaining bytes (0–32, 127–160, 173) are mapped to code points starting
/// at U+0100 (256).
pub fn gpt2_byte_to_unicode() -> HashMap<u8, char> {
    let mut map = HashMap::new();
    let mut n: u32 = 0;
    for b in 0u16..=255 {
        let b = b as u8;
        let c = if is_gpt2_printable(b) {
            char::from(b)
        } else {
            let ch = char::from_u32(256 + n).unwrap();
            n += 1;
            ch
        };
        map.insert(b, c);
    }
    map
}

/// Returns true if the byte maps to itself in GPT-2's encoding.
fn is_gpt2_printable(b: u8) -> bool {
    matches!(b, 33..=126 | 161..=172 | 174..=255)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_to_unicode_is_bijective() {
        let map = gpt2_byte_to_unicode();
        assert_eq!(map.len(), 256);
        // All values should be unique.
        let mut values: Vec<char> = map.values().copied().collect();
        values.sort();
        values.dedup();
        assert_eq!(values.len(), 256);
    }

    #[test]
    fn printable_bytes_map_to_themselves() {
        let map = gpt2_byte_to_unicode();
        for b in 33u8..=126 {
            assert_eq!(map[&b], char::from(b));
        }
    }

    #[test]
    fn non_printable_bytes_map_to_high_codepoints() {
        let map = gpt2_byte_to_unicode();
        // Byte 0 (non-printable) should map to U+0100 or higher.
        let c = map[&0u8];
        assert!(c as u32 >= 256);
    }

    #[test]
    fn from_gpt2_minimal() {
        // Build a minimal GPT-2-style vocab and merges.
        let byte_to_unicode = gpt2_byte_to_unicode();

        // Create vocab for bytes 'h', 'i' and their merge "hi".
        let h_char = byte_to_unicode[&b'h'].to_string();
        let i_char = byte_to_unicode[&b'i'].to_string();
        let hi_str = format!("{}{}", h_char, i_char);

        let mut vocab_map: HashMap<String, TokenId> = HashMap::new();
        // Add all 256 byte-level tokens.
        for b in 0u8..=255 {
            let c = byte_to_unicode[&b].to_string();
            vocab_map.insert(c, b as TokenId);
        }
        // Add the merged token.
        vocab_map.insert(hi_str, 256);

        let vocab_json = serde_json::to_string(&vocab_map).unwrap();
        let merges_txt = format!("#version: 0.2\n{} {}", h_char, i_char);

        let tok = Tokenizer::from_gpt2(&vocab_json, &merges_txt);
        assert_eq!(tok.vocab_size(), 257);

        // "hi" should encode to a single merged token.
        let ids = tok.encode_gpt2("hi");
        assert_eq!(ids, vec![256]);
        assert_eq!(tok.decode(&ids), "hi");
    }
}
