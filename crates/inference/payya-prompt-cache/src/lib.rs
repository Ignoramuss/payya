//! Radix-tree prefix matching for KV cache reuse across requests.
//!
//! When multiple requests share a common prompt prefix (e.g., a system prompt),
//! recomputing the KV cache for that prefix is wasteful. This crate implements
//! a **radix tree** (compressed trie) over token sequences, where each node
//! stores a reference to cached KV data that can be reused.
//!
//! # Architecture
//!
//! ```text
//!  Request: [sys_tok, sys_tok, sys_tok, usr_tok, usr_tok]
//!                                       │
//!                    ┌──────────────────┘
//!                    ▼
//!         ┌──────────────────┐
//!         │    RadixTree     │
//!         │  root: Node      │
//!         └────────┬─────────┘
//!                  │
//!          ┌───────┴───────┐
//!          ▼               ▼
//!    [sys_tok×3]      [other_prefix]
//!    cache_id: 42     cache_id: 99
//!          │
//!    ┌─────┴─────┐
//!    ▼           ▼
//!  [usr_A]   [usr_B]
//!  id: 43    id: 44
//! ```
//!
//! The tree is queried with `lookup(tokens)` which returns the longest
//! matching prefix and its associated cache ID. After computing the
//! remaining tokens, the new prefix is inserted with `insert(tokens, cache_id)`.

use std::collections::HashMap;

// ── Cache entry ─────────────────────────────────────────────────────

/// Unique identifier for a cached KV entry (maps to external KV cache storage).
pub type CacheId = u64;

/// Token ID type (matches payya-tokenizer's TokenId).
pub type TokenId = u32;

// ── Radix tree node ─────────────────────────────────────────────────

/// A node in the radix tree. Edges are labeled with token subsequences.
#[derive(Debug)]
struct Node {
    /// Children keyed by the first token of the edge label.
    children: HashMap<TokenId, Edge>,
    /// If this node represents a complete cached prefix, its cache ID.
    cache_id: Option<CacheId>,
}

/// An edge in the radix tree: a sequence of tokens leading to a child node.
#[derive(Debug)]
struct Edge {
    /// The token subsequence labeling this edge.
    label: Vec<TokenId>,
    /// The child node at the end of this edge.
    child: Node,
}

impl Node {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            cache_id: None,
        }
    }
}

// ── Lookup result ───────────────────────────────────────────────────

/// Result of a prefix lookup in the radix tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixMatch {
    /// The cache ID for the longest matching prefix.
    pub cache_id: CacheId,
    /// Number of tokens matched (length of the prefix hit).
    pub matched_len: usize,
}

// ── Radix tree ──────────────────────────────────────────────────────

/// A radix tree for matching token sequence prefixes to cached KV data.
///
/// Supports insert, lookup, and eviction of cached prefixes.
#[derive(Debug)]
pub struct RadixTree {
    root: Node,
    /// Total number of cached entries (nodes with cache_id set).
    num_entries: usize,
    /// Next cache ID to assign (auto-increment).
    next_cache_id: CacheId,
}

impl RadixTree {
    /// Create an empty radix tree.
    pub fn new() -> Self {
        Self {
            root: Node::new(),
            num_entries: 0,
            next_cache_id: 0,
        }
    }

    /// Look up the longest cached prefix of `tokens`.
    ///
    /// Returns `Some(PrefixMatch)` if any prefix matches, `None` if no prefix
    /// is cached.
    pub fn lookup(&self, tokens: &[TokenId]) -> Option<PrefixMatch> {
        let mut best: Option<PrefixMatch> = None;
        let mut node = &self.root;
        let mut pos = 0;

        if let Some(cache_id) = node.cache_id {
            best = Some(PrefixMatch {
                cache_id,
                matched_len: 0,
            });
        }

        while pos < tokens.len() {
            let first_tok = tokens[pos];
            match node.children.get(&first_tok) {
                None => break,
                Some(edge) => {
                    // Check if the remaining tokens match the edge label.
                    let remaining = &tokens[pos..];
                    let match_len = common_prefix_len(&edge.label, remaining);
                    pos += match_len;

                    if match_len < edge.label.len() {
                        // Partial match within edge — can't go deeper.
                        break;
                    }

                    // Full edge match — move to child.
                    node = &edge.child;
                    if let Some(cache_id) = node.cache_id {
                        best = Some(PrefixMatch {
                            cache_id,
                            matched_len: pos,
                        });
                    }
                }
            }
        }

        best
    }

    /// Insert a token sequence into the tree with an auto-assigned cache ID.
    ///
    /// Returns the assigned cache ID. If the exact sequence already exists,
    /// returns the existing cache ID.
    pub fn insert(&mut self, tokens: &[TokenId]) -> CacheId {
        if tokens.is_empty() {
            if self.root.cache_id.is_none() {
                let id = self.next_cache_id;
                self.next_cache_id += 1;
                self.root.cache_id = Some(id);
                self.num_entries += 1;
            }
            return self.root.cache_id.unwrap();
        }

        let id = self.next_cache_id;
        let inserted = Self::insert_recursive(&mut self.root.children, tokens, id);
        if inserted {
            self.next_cache_id += 1;
            self.num_entries += 1;
            id
        } else {
            // Already existed — find the existing ID.
            self.lookup(tokens).unwrap().cache_id
        }
    }

    /// Internal recursive insert. Returns true if a new entry was created.
    fn insert_recursive(
        // We need to work with a mutable reference to children to avoid
        // borrow-checker issues with self.root.
        children: *mut HashMap<TokenId, Edge>,
        tokens: &[TokenId],
        cache_id: CacheId,
    ) -> bool {
        // SAFETY: We have exclusive access via &mut self in the caller.
        let children = unsafe { &mut *children };

        let first_tok = tokens[0];

        if let Some(edge) = children.get_mut(&first_tok) {
            let match_len = common_prefix_len(&edge.label, tokens);

            if match_len == edge.label.len() && match_len == tokens.len() {
                // Exact match with this edge — set cache ID on child node.
                if edge.child.cache_id.is_some() {
                    return false; // already exists
                }
                edge.child.cache_id = Some(cache_id);
                return true;
            }

            if match_len == edge.label.len() {
                // Full edge consumed, continue into child.
                let remaining = &tokens[match_len..];
                let child_children = &mut edge.child.children as *mut HashMap<TokenId, Edge>;
                return Self::insert_recursive(child_children, remaining, cache_id);
            }

            // Partial match — need to split the edge.
            let common = edge.label[..match_len].to_vec();
            let old_suffix = edge.label[match_len..].to_vec();

            // Take ownership of the old edge's child.
            let old_child_children = std::mem::take(&mut edge.child.children);
            let old_cache_id = edge.child.cache_id.take();

            // Create the split node (at the divergence point).
            let mut split_node = Node::new();

            // Re-attach the old suffix as a child of the split node.
            let mut old_child = Node::new();
            old_child.children = old_child_children;
            old_child.cache_id = old_cache_id;
            split_node.children.insert(
                old_suffix[0],
                Edge {
                    label: old_suffix,
                    child: old_child,
                },
            );

            if match_len == tokens.len() {
                // The new key ends exactly at the split point.
                split_node.cache_id = Some(cache_id);
            } else {
                // Insert the new suffix as another child of the split node.
                let new_suffix = tokens[match_len..].to_vec();
                let mut new_child = Node::new();
                new_child.cache_id = Some(cache_id);
                split_node.children.insert(
                    new_suffix[0],
                    Edge {
                        label: new_suffix,
                        child: new_child,
                    },
                );
            }

            // Replace the edge with the shortened common prefix.
            edge.label = common;
            edge.child = split_node;

            return true;
        }

        // No matching edge — create a new one.
        let mut new_child = Node::new();
        new_child.cache_id = Some(cache_id);
        children.insert(
            first_tok,
            Edge {
                label: tokens.to_vec(),
                child: new_child,
            },
        );
        true
    }

    /// Remove a cached prefix by its cache ID. Returns true if found and removed.
    pub fn remove(&mut self, cache_id: CacheId) -> bool {
        let removed = Self::remove_recursive(&mut self.root, cache_id);
        if removed {
            self.num_entries -= 1;
        }
        removed
    }

    fn remove_recursive(node: &mut Node, cache_id: CacheId) -> bool {
        if node.cache_id == Some(cache_id) {
            node.cache_id = None;
            return true;
        }
        for edge in node.children.values_mut() {
            if Self::remove_recursive(&mut edge.child, cache_id) {
                return true;
            }
        }
        false
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.num_entries
    }

    /// Whether the tree has any cached entries.
    pub fn is_empty(&self) -> bool {
        self.num_entries == 0
    }
}

impl Default for RadixTree {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the length of the common prefix between two slices.
fn common_prefix_len(a: &[TokenId], b: &[TokenId]) -> usize {
    a.iter().zip(b.iter()).take_while(|(&x, &y)| x == y).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_tree_returns_none() {
        let tree = RadixTree::new();
        assert!(tree.lookup(&[1, 2, 3]).is_none());
        assert!(tree.is_empty());
    }

    #[test]
    fn insert_and_exact_lookup() {
        let mut tree = RadixTree::new();
        let id = tree.insert(&[1, 2, 3]);
        let result = tree.lookup(&[1, 2, 3]).unwrap();
        assert_eq!(result.cache_id, id);
        assert_eq!(result.matched_len, 3);
    }

    #[test]
    fn prefix_match() {
        let mut tree = RadixTree::new();
        let id = tree.insert(&[10, 20, 30]);
        // Query with longer sequence — should match the prefix.
        let result = tree.lookup(&[10, 20, 30, 40, 50]).unwrap();
        assert_eq!(result.cache_id, id);
        assert_eq!(result.matched_len, 3);
    }

    #[test]
    fn no_match_for_different_prefix() {
        let mut tree = RadixTree::new();
        tree.insert(&[1, 2, 3]);
        assert!(tree.lookup(&[4, 5, 6]).is_none());
    }

    #[test]
    fn longest_prefix_wins() {
        let mut tree = RadixTree::new();
        let _short = tree.insert(&[1, 2]);
        let long = tree.insert(&[1, 2, 3, 4]);

        let result = tree.lookup(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(result.cache_id, long);
        assert_eq!(result.matched_len, 4);

        // Shorter query should match shorter prefix.
        let result = tree.lookup(&[1, 2, 9]).unwrap();
        assert_eq!(result.cache_id, _short);
        assert_eq!(result.matched_len, 2);
    }

    #[test]
    fn duplicate_insert_returns_same_id() {
        let mut tree = RadixTree::new();
        let id1 = tree.insert(&[1, 2, 3]);
        let id2 = tree.insert(&[1, 2, 3]);
        assert_eq!(id1, id2);
        assert_eq!(tree.len(), 1);
    }

    #[test]
    fn diverging_paths() {
        let mut tree = RadixTree::new();
        let id_a = tree.insert(&[1, 2, 3]);
        let id_b = tree.insert(&[1, 2, 4]);

        let a = tree.lookup(&[1, 2, 3]).unwrap();
        assert_eq!(a.cache_id, id_a);
        assert_eq!(a.matched_len, 3);

        let b = tree.lookup(&[1, 2, 4]).unwrap();
        assert_eq!(b.cache_id, id_b);
        assert_eq!(b.matched_len, 3);
    }

    #[test]
    fn remove_entry() {
        let mut tree = RadixTree::new();
        let id = tree.insert(&[1, 2, 3]);
        assert_eq!(tree.len(), 1);
        assert!(tree.remove(id));
        assert_eq!(tree.len(), 0);
        assert!(tree.lookup(&[1, 2, 3]).is_none());
    }

    #[test]
    fn remove_preserves_siblings() {
        let mut tree = RadixTree::new();
        let id_a = tree.insert(&[1, 2, 3]);
        let id_b = tree.insert(&[1, 2, 4]);
        tree.remove(id_a);

        assert!(tree.lookup(&[1, 2, 3]).is_none());
        let b = tree.lookup(&[1, 2, 4]).unwrap();
        assert_eq!(b.cache_id, id_b);
    }

    #[test]
    fn many_prefixes() {
        let mut tree = RadixTree::new();
        let mut ids = Vec::new();
        for i in 0..100u32 {
            let tokens: Vec<TokenId> = vec![0, 1, i];
            ids.push(tree.insert(&tokens));
        }
        assert_eq!(tree.len(), 100);

        for (i, &id) in ids.iter().enumerate() {
            let tokens: Vec<TokenId> = vec![0, 1, i as u32];
            let result = tree.lookup(&tokens).unwrap();
            assert_eq!(result.cache_id, id);
        }
    }

    #[test]
    fn partial_edge_match() {
        let mut tree = RadixTree::new();
        let id = tree.insert(&[1, 2, 3, 4, 5]);
        // Query [1, 2] — partial match of edge [1,2,3,4,5], no node with cache_id.
        assert!(tree.lookup(&[1, 2]).is_none());
        // Query [1, 2, 3, 4, 5, 6] — full match.
        let result = tree.lookup(&[1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(result.cache_id, id);
    }

    #[test]
    fn split_creates_intermediate() {
        let mut tree = RadixTree::new();
        tree.insert(&[1, 2, 3, 4, 5]);
        let id_short = tree.insert(&[1, 2, 3]);
        // Now [1,2,3] should be a valid prefix.
        let result = tree.lookup(&[1, 2, 3]).unwrap();
        assert_eq!(result.cache_id, id_short);
        assert_eq!(result.matched_len, 3);
    }

    #[test]
    fn prefix_hit_reduces_recomputation() {
        // Simulate: two requests share a system prompt prefix.
        let mut tree = RadixTree::new();
        let system_tokens: Vec<TokenId> = vec![100, 101, 102, 103, 104];

        // First request caches the system prompt.
        let cache_id = tree.insert(&system_tokens);

        // Second request with same system prompt + user tokens.
        let mut request2 = system_tokens.clone();
        request2.extend_from_slice(&[200, 201, 202]);

        let hit = tree.lookup(&request2).unwrap();
        assert_eq!(hit.cache_id, cache_id);
        assert_eq!(hit.matched_len, 5);

        // Only 3 new tokens need to be computed.
        let tokens_to_compute = request2.len() - hit.matched_len;
        assert_eq!(tokens_to_compute, 3);
    }
}
