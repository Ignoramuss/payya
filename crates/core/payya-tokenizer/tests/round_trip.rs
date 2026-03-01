//! Integration tests: encode→decode round-trip on diverse inputs.

use payya_tokenizer::Tokenizer;

/// Train a tokenizer and verify round-trip on the training corpus itself.
#[test]
fn round_trip_training_corpus() {
    let corpus = "the quick brown fox jumps over the lazy dog. \
                  the quick brown fox jumps over the lazy dog. \
                  the quick brown fox jumps over the lazy dog.";
    let tok = Tokenizer::train(corpus, 280);
    assert_eq!(tok.decode(&tok.encode(corpus)), corpus);
}

/// Round-trip on text different from the training corpus.
#[test]
fn round_trip_unseen_text() {
    let corpus = "abcabc defdef ghighi abcabc defdef";
    let tok = Tokenizer::train(corpus, 265);

    // Text not seen during training still round-trips because all bytes are in the vocab.
    let unseen = "xyz 123 !@# abc def";
    assert_eq!(tok.decode(&tok.encode(unseen)), unseen);
}

/// Round-trip on various Unicode texts.
#[test]
fn round_trip_unicode() {
    let corpus = "hello world hello world hello world";
    let tok = Tokenizer::train(corpus, 260);

    let cases = [
        "café au lait",
        "日本語テスト",
        "🦀🦀🦀 Rust",
        "Ñoño über straße",
        "mixed ASCII and ünïcödé",
        "", // empty string
    ];

    for text in &cases {
        assert_eq!(
            tok.decode(&tok.encode(text)),
            *text,
            "round-trip failed for {:?}",
            text
        );
    }
}

/// Training produces compression: encoded output is shorter than byte-level.
#[test]
fn training_compresses() {
    let corpus = "aaaa bbbb cccc aaaa bbbb cccc aaaa bbbb cccc";
    let tok = Tokenizer::train(corpus, 270);
    let ids = tok.encode(corpus);
    // Encoded should be shorter than byte count due to merges.
    assert!(
        ids.len() < corpus.len(),
        "expected compression: {} tokens < {} bytes",
        ids.len(),
        corpus.len()
    );
    // But still round-trips correctly.
    assert_eq!(tok.decode(&ids), corpus);
}

/// JSON serialization round-trip preserves encode/decode behavior.
#[test]
fn json_persistence_round_trip() {
    let corpus = "the cat sat on the mat the cat sat on the mat";
    let tok = Tokenizer::train(corpus, 270);

    let json = tok.to_json();
    let tok2 = Tokenizer::from_json(&json);

    let test_texts = ["the cat", "sat on", "hello world", ""];
    for text in &test_texts {
        assert_eq!(
            tok.encode(text),
            tok2.encode(text),
            "encode mismatch for {:?}",
            text
        );
        let ids = tok.encode(text);
        assert_eq!(
            tok.decode(&ids),
            tok2.decode(&ids),
            "decode mismatch for {:?}",
            text
        );
    }
}

/// Larger vocabulary still produces correct results.
#[test]
fn larger_vocab() {
    let corpus = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. \
                  Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. \
                  Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
        .repeat(10);
    let tok = Tokenizer::train(&corpus, 350);

    // Should have learned many merges.
    assert!(
        tok.merges().len() > 50,
        "expected >50 merges, got {}",
        tok.merges().len()
    );

    // Round-trip.
    assert_eq!(tok.decode(&tok.encode(&corpus)), corpus);

    // Compression.
    let ids = tok.encode(&corpus);
    assert!(ids.len() < corpus.len());
}
