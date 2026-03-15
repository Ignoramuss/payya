//! Paged KV cache with block tables (vLLM-style).
//!
//! Instead of allocating one contiguous buffer per sequence, the cache is
//! divided into fixed-size **blocks**. Each sequence maintains a **block table**
//! — a list of block indices that map logical positions to physical storage.
//! This avoids internal fragmentation and enables copy-on-write sharing.
//!
//! # Architecture
//!
//! ```text
//!  ┌────────────────────────────────────────────────────────┐
//!  │                   BlockAllocator                       │
//!  │  free_blocks: [4, 5, 6, ...]  total: 1024             │
//!  └────────────────────────────────────────────────────────┘
//!                          │
//!              ┌───────────┴──────────┐
//!              ▼                      ▼
//!  ┌───────────────────┐  ┌───────────────────┐
//!  │  Sequence 0       │  │  Sequence 1       │
//!  │  block_table:     │  │  block_table:     │
//!  │    [0, 1, 2]      │  │    [3, 7, 8]      │
//!  │  len: 40          │  │  len: 35          │
//!  └───────────────────┘  └───────────────────┘
//!              │                      │
//!              ▼                      ▼
//!  ┌──────────────────────────────────────────────────────┐
//!  │              Physical Block Storage                   │
//!  │  block 0: [k0..k15, v0..v15]  (seq 0, pos 0..15)    │
//!  │  block 1: [k16..k31, v16..v31] (seq 0, pos 16..31)  │
//!  │  block 2: [k32..k39, -, -, ...]  (seq 0, pos 32..39)│
//!  │  block 3: [k0..k15, v0..v15]  (seq 1, pos 0..15)    │
//!  │  ...                                                  │
//!  └──────────────────────────────────────────────────────┘
//! ```

// ── Block allocator ─────────────────────────────────────────────────

/// Manages a pool of fixed-size blocks. Blocks are identified by `usize` index.
#[derive(Debug)]
pub struct BlockAllocator {
    total_blocks: usize,
    free_list: Vec<usize>,
}

impl BlockAllocator {
    /// Create a new allocator with the given number of blocks.
    pub fn new(total_blocks: usize) -> Self {
        assert!(total_blocks > 0, "must have at least 1 block");
        let free_list: Vec<usize> = (0..total_blocks).rev().collect();
        Self {
            total_blocks,
            free_list,
        }
    }

    /// Allocate a single block. Returns `None` if OOM.
    pub fn alloc(&mut self) -> Option<usize> {
        self.free_list.pop()
    }

    /// Free a previously allocated block.
    ///
    /// # Panics
    ///
    /// Panics if `block_id` is out of range.
    pub fn free(&mut self, block_id: usize) {
        assert!(
            block_id < self.total_blocks,
            "block_id={block_id} out of range (total={})",
            self.total_blocks
        );
        self.free_list.push(block_id);
    }

    /// Number of free blocks remaining.
    pub fn num_free(&self) -> usize {
        self.free_list.len()
    }

    /// Total number of blocks.
    pub fn total_blocks(&self) -> usize {
        self.total_blocks
    }
}

// ── Sequence state ──────────────────────────────────────────────────

/// Per-sequence metadata: block table + token count.
#[derive(Debug, Clone)]
pub struct SequenceState {
    /// Ordered list of physical block indices holding this sequence's KV data.
    block_table: Vec<usize>,
    /// Number of tokens currently cached.
    len: usize,
}

impl SequenceState {
    fn new() -> Self {
        Self {
            block_table: Vec::new(),
            len: 0,
        }
    }

    /// Return the block table (physical block indices in logical order).
    pub fn block_table(&self) -> &[usize] {
        &self.block_table
    }

    /// Number of tokens cached for this sequence.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this sequence has any cached tokens.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// ── Paged KV cache ─────────────────────────────────────────────────

/// A paged KV cache for multi-layer, multi-head attention.
///
/// Storage layout per block:
///   block stores `block_size` positions.
///   For each position: `n_layers * n_heads * d_head` floats for K,
///   then the same for V.
///
/// Total floats per block = `block_size * n_layers * n_heads * d_head * 2`.
#[derive(Debug)]
pub struct PagedKvCache {
    /// Number of tokens per block.
    block_size: usize,
    /// Number of transformer layers.
    n_layers: usize,
    /// Number of attention heads.
    n_heads: usize,
    /// Dimension per head.
    d_head: usize,
    /// Block allocator.
    allocator: BlockAllocator,
    /// Physical storage: `total_blocks * floats_per_block` contiguous floats.
    storage: Vec<f32>,
    /// Per-sequence state, keyed by sequence id.
    sequences: Vec<Option<SequenceState>>,
    /// Floats per block (cached for speed).
    floats_per_block: usize,
}

impl PagedKvCache {
    /// Create a new paged KV cache.
    ///
    /// - `block_size`: tokens per block (e.g. 16).
    /// - `total_blocks`: number of physical blocks.
    /// - `n_layers`: number of transformer layers.
    /// - `n_heads`: number of attention heads.
    /// - `d_head`: dimension per head.
    /// - `max_sequences`: maximum number of concurrent sequences.
    pub fn new(
        block_size: usize,
        total_blocks: usize,
        n_layers: usize,
        n_heads: usize,
        d_head: usize,
        max_sequences: usize,
    ) -> Self {
        assert!(block_size > 0, "block_size must be > 0");
        assert!(n_layers > 0, "n_layers must be > 0");
        assert!(n_heads > 0, "n_heads must be > 0");
        assert!(d_head > 0, "d_head must be > 0");
        assert!(max_sequences > 0, "max_sequences must be > 0");

        // Each block stores K and V for all layers/heads.
        // Layout: [K data for pos 0..block_size][V data for pos 0..block_size]
        // K data per pos: n_layers * n_heads * d_head
        let kv_per_pos = n_layers * n_heads * d_head;
        let floats_per_block = block_size * kv_per_pos * 2; // *2 for K and V

        let storage = vec![0.0f32; total_blocks * floats_per_block];
        let sequences = vec![None; max_sequences];

        Self {
            block_size,
            n_layers,
            n_heads,
            d_head,
            allocator: BlockAllocator::new(total_blocks),
            storage,
            sequences,
            floats_per_block,
        }
    }

    /// Register a new sequence. Returns the sequence ID.
    ///
    /// # Panics
    ///
    /// Panics if no sequence slots are available.
    pub fn add_sequence(&mut self) -> usize {
        for (i, slot) in self.sequences.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some(SequenceState::new());
                return i;
            }
        }
        panic!("no free sequence slots (max={})", self.sequences.len());
    }

    /// Remove a sequence and free all its blocks.
    ///
    /// # Panics
    ///
    /// Panics if `seq_id` is invalid.
    pub fn remove_sequence(&mut self, seq_id: usize) {
        let state = self.sequences[seq_id]
            .take()
            .unwrap_or_else(|| panic!("sequence {seq_id} does not exist"));
        for &block_id in &state.block_table {
            self.allocator.free(block_id);
        }
    }

    /// Get the sequence state (block table + length).
    pub fn sequence(&self, seq_id: usize) -> &SequenceState {
        self.sequences[seq_id]
            .as_ref()
            .unwrap_or_else(|| panic!("sequence {seq_id} does not exist"))
    }

    /// Append KV data for new tokens to a sequence.
    ///
    /// `k_data` and `v_data` are flat arrays of shape
    /// `(num_new_tokens, n_layers, n_heads, d_head)`.
    ///
    /// Allocates new blocks as needed. Returns `Err` if out of memory.
    pub fn append(
        &mut self,
        seq_id: usize,
        k_data: &[f32],
        v_data: &[f32],
    ) -> Result<(), KvCacheError> {
        let kv_per_pos = self.n_layers * self.n_heads * self.d_head;
        assert_eq!(
            k_data.len() % kv_per_pos,
            0,
            "k_data length {} not divisible by kv_per_pos={}",
            k_data.len(),
            kv_per_pos
        );
        assert_eq!(
            k_data.len(),
            v_data.len(),
            "k_data.len()={} != v_data.len()={}",
            k_data.len(),
            v_data.len()
        );

        let num_new = k_data.len() / kv_per_pos;
        let state = self.sequences[seq_id]
            .as_mut()
            .unwrap_or_else(|| panic!("sequence {seq_id} does not exist"));

        for tok_idx in 0..num_new {
            let pos = state.len;
            let block_idx_in_seq = pos / self.block_size;
            let pos_in_block = pos % self.block_size;

            // Allocate a new block if we've moved past the current one.
            if block_idx_in_seq >= state.block_table.len() {
                let block_id = self.allocator.alloc().ok_or(KvCacheError::OutOfMemory)?;
                state.block_table.push(block_id);
            }

            let phys_block = state.block_table[block_idx_in_seq];
            let block_base = phys_block * self.floats_per_block;

            // K offset: block_base + pos_in_block * kv_per_pos
            let k_offset = block_base + pos_in_block * kv_per_pos;
            let src_offset = tok_idx * kv_per_pos;
            self.storage[k_offset..k_offset + kv_per_pos]
                .copy_from_slice(&k_data[src_offset..src_offset + kv_per_pos]);

            // V offset: after all K data in block.
            let v_base = block_base + self.block_size * kv_per_pos;
            let v_offset = v_base + pos_in_block * kv_per_pos;
            self.storage[v_offset..v_offset + kv_per_pos]
                .copy_from_slice(&v_data[src_offset..src_offset + kv_per_pos]);

            state.len += 1;
        }

        Ok(())
    }

    /// Read cached K values for a sequence, for a specific layer and head.
    ///
    /// Returns a vector of shape `(seq_len, d_head)`.
    pub fn read_k(&self, seq_id: usize, layer: usize, head: usize) -> Vec<f32> {
        assert!(
            layer < self.n_layers,
            "layer={layer} >= n_layers={}",
            self.n_layers
        );
        assert!(
            head < self.n_heads,
            "head={head} >= n_heads={}",
            self.n_heads
        );

        let state = self.sequence(seq_id);
        let kv_per_pos = self.n_layers * self.n_heads * self.d_head;
        let head_offset = (layer * self.n_heads + head) * self.d_head;

        let mut result = Vec::with_capacity(state.len * self.d_head);

        for pos in 0..state.len {
            let block_idx = pos / self.block_size;
            let pos_in_block = pos % self.block_size;
            let phys_block = state.block_table[block_idx];
            let block_base = phys_block * self.floats_per_block;
            let k_offset = block_base + pos_in_block * kv_per_pos + head_offset;
            result.extend_from_slice(&self.storage[k_offset..k_offset + self.d_head]);
        }

        result
    }

    /// Read cached V values for a sequence, for a specific layer and head.
    ///
    /// Returns a vector of shape `(seq_len, d_head)`.
    pub fn read_v(&self, seq_id: usize, layer: usize, head: usize) -> Vec<f32> {
        assert!(
            layer < self.n_layers,
            "layer={layer} >= n_layers={}",
            self.n_layers
        );
        assert!(
            head < self.n_heads,
            "head={head} >= n_heads={}",
            self.n_heads
        );

        let state = self.sequence(seq_id);
        let kv_per_pos = self.n_layers * self.n_heads * self.d_head;
        let head_offset = (layer * self.n_heads + head) * self.d_head;

        let mut result = Vec::with_capacity(state.len * self.d_head);

        for pos in 0..state.len {
            let block_idx = pos / self.block_size;
            let pos_in_block = pos % self.block_size;
            let phys_block = state.block_table[block_idx];
            let block_base = phys_block * self.floats_per_block;
            let v_base = block_base + self.block_size * kv_per_pos;
            let v_offset = v_base + pos_in_block * kv_per_pos + head_offset;
            result.extend_from_slice(&self.storage[v_offset..v_offset + self.d_head]);
        }

        result
    }

    /// Number of free blocks remaining.
    pub fn num_free_blocks(&self) -> usize {
        self.allocator.num_free()
    }

    /// Block size (tokens per block).
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Total number of active sequences.
    pub fn num_active_sequences(&self) -> usize {
        self.sequences.iter().filter(|s| s.is_some()).count()
    }

    /// Token count for all cached KV data across all sequences.
    pub fn total_cached_tokens(&self) -> usize {
        self.sequences
            .iter()
            .filter_map(|s| s.as_ref())
            .map(|s| s.len)
            .sum()
    }
}

/// Errors that can occur during KV cache operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvCacheError {
    /// No free blocks available to allocate.
    OutOfMemory,
}

impl std::fmt::Display for KvCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KvCacheError::OutOfMemory => write!(f, "KV cache out of memory: no free blocks"),
        }
    }
}

impl std::error::Error for KvCacheError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocator_basic() {
        let mut alloc = BlockAllocator::new(4);
        assert_eq!(alloc.num_free(), 4);
        let b0 = alloc.alloc().unwrap();
        let b1 = alloc.alloc().unwrap();
        assert_ne!(b0, b1);
        assert_eq!(alloc.num_free(), 2);
        alloc.free(b0);
        assert_eq!(alloc.num_free(), 3);
    }

    #[test]
    fn allocator_exhaustion() {
        let mut alloc = BlockAllocator::new(2);
        alloc.alloc().unwrap();
        alloc.alloc().unwrap();
        assert!(alloc.alloc().is_none());
    }

    #[test]
    fn cache_add_remove_sequence() {
        let mut cache = PagedKvCache::new(16, 64, 2, 4, 8, 4);
        let s0 = cache.add_sequence();
        let s1 = cache.add_sequence();
        assert_eq!(cache.num_active_sequences(), 2);
        cache.remove_sequence(s0);
        assert_eq!(cache.num_active_sequences(), 1);
        cache.remove_sequence(s1);
        assert_eq!(cache.num_active_sequences(), 0);
    }

    #[test]
    fn cache_append_and_read() {
        let n_layers = 2;
        let n_heads = 2;
        let d_head = 4;
        let block_size = 4;
        let mut cache = PagedKvCache::new(block_size, 16, n_layers, n_heads, d_head, 2);

        let seq = cache.add_sequence();
        let kv_per_pos = n_layers * n_heads * d_head; // 2*2*4 = 16

        // Append 3 tokens.
        let k_data: Vec<f32> = (0..3 * kv_per_pos).map(|i| i as f32).collect();
        let v_data: Vec<f32> = (0..3 * kv_per_pos).map(|i| 100.0 + i as f32).collect();
        cache.append(seq, &k_data, &v_data).unwrap();

        assert_eq!(cache.sequence(seq).len(), 3);

        // Read K for layer 0, head 0.
        let k00 = cache.read_k(seq, 0, 0);
        assert_eq!(k00.len(), 3 * d_head);
        // First token, layer 0, head 0 = positions 0..4 of k_data.
        assert_eq!(&k00[0..d_head], &[0.0, 1.0, 2.0, 3.0]);

        // Read V for layer 0, head 0.
        let v00 = cache.read_v(seq, 0, 0);
        assert_eq!(v00.len(), 3 * d_head);
        assert_eq!(&v00[0..d_head], &[100.0, 101.0, 102.0, 103.0]);
    }

    #[test]
    fn cache_spans_multiple_blocks() {
        let n_layers = 1;
        let n_heads = 1;
        let d_head = 2;
        let block_size = 2;
        let mut cache = PagedKvCache::new(block_size, 8, n_layers, n_heads, d_head, 1);

        let seq = cache.add_sequence();
        let kv_per_pos = n_layers * n_heads * d_head; // 2

        // Append 5 tokens — needs 3 blocks (block_size=2).
        let k_data: Vec<f32> = (0..5 * kv_per_pos).map(|i| i as f32).collect();
        let v_data: Vec<f32> = (0..5 * kv_per_pos).map(|i| 50.0 + i as f32).collect();
        cache.append(seq, &k_data, &v_data).unwrap();

        assert_eq!(cache.sequence(seq).len(), 5);
        assert_eq!(cache.sequence(seq).block_table().len(), 3);

        let k = cache.read_k(seq, 0, 0);
        assert_eq!(k.len(), 5 * d_head);
        // Token 4 (last): k_data offset = 4*2 = 8, values [8.0, 9.0].
        assert_eq!(&k[4 * d_head..5 * d_head], &[8.0, 9.0]);
    }

    #[test]
    fn cache_oom_returns_error() {
        let mut cache = PagedKvCache::new(4, 1, 1, 1, 2, 1);
        let seq = cache.add_sequence();
        let kv = vec![1.0f32; 4 * 2]; // 4 tokens, fills 1 block
        cache.append(seq, &kv, &kv).unwrap();

        // 5th token needs a 2nd block but only 1 exists.
        let one = vec![1.0f32; 2];
        assert_eq!(
            cache.append(seq, &one, &one),
            Err(KvCacheError::OutOfMemory)
        );
    }

    #[test]
    fn cache_free_blocks_reclaimed() {
        let mut cache = PagedKvCache::new(2, 2, 1, 1, 2, 2);
        let free_before = cache.num_free_blocks();
        let s0 = cache.add_sequence();
        let kv = vec![1.0f32; 4]; // 2 tokens = 1 block
        cache.append(s0, &kv, &kv).unwrap();
        assert_eq!(cache.num_free_blocks(), free_before - 1);
        cache.remove_sequence(s0);
        assert_eq!(cache.num_free_blocks(), free_before);
    }

    #[test]
    fn cache_memory_bounded() {
        // Verify that removing and re-adding sequences doesn't leak blocks.
        let mut cache = PagedKvCache::new(4, 8, 1, 1, 2, 4);
        for _ in 0..100 {
            let seq = cache.add_sequence();
            let kv = vec![1.0f32; 4 * 2]; // 4 tokens
            cache.append(seq, &kv, &kv).unwrap();
            cache.remove_sequence(seq);
        }
        assert_eq!(cache.num_free_blocks(), 8);
    }

    #[test]
    fn cache_incremental_append() {
        let mut cache = PagedKvCache::new(4, 8, 1, 1, 2, 1);
        let seq = cache.add_sequence();
        let kv_per_pos = 2;

        // Append one token at a time.
        for i in 0..6 {
            let k = vec![i as f32; kv_per_pos];
            let v = vec![(i as f32) + 100.0; kv_per_pos];
            cache.append(seq, &k, &v).unwrap();
        }
        assert_eq!(cache.sequence(seq).len(), 6);

        let k = cache.read_k(seq, 0, 0);
        assert_eq!(k.len(), 6 * 2);
        // Token 3: k value should be [3.0, 3.0].
        assert_eq!(&k[6..8], &[3.0, 3.0]);
    }

    #[test]
    fn concurrent_sequences() {
        let mut cache = PagedKvCache::new(4, 16, 1, 2, 4, 4);
        let kv_per_pos = 1 * 2 * 4; // 8

        let s0 = cache.add_sequence();
        let s1 = cache.add_sequence();

        let k0: Vec<f32> = (0..3 * kv_per_pos).map(|i| i as f32).collect();
        let v0: Vec<f32> = (0..3 * kv_per_pos).map(|i| 100.0 + i as f32).collect();
        cache.append(s0, &k0, &v0).unwrap();

        let k1: Vec<f32> = (0..5 * kv_per_pos).map(|i| -1.0 * i as f32).collect();
        let v1: Vec<f32> = (0..5 * kv_per_pos).map(|i| -100.0 - i as f32).collect();
        cache.append(s1, &k1, &v1).unwrap();

        assert_eq!(cache.sequence(s0).len(), 3);
        assert_eq!(cache.sequence(s1).len(), 5);
        assert_eq!(cache.total_cached_tokens(), 8);

        // Verify data isolation.
        let k0_read = cache.read_k(s0, 0, 0);
        let k1_read = cache.read_k(s1, 0, 0);
        assert_eq!(&k0_read[0..4], &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(&k1_read[0..4], &[0.0, -1.0, -2.0, -3.0]);
    }
}
