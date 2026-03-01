//! CLI example: train a BPE tokenizer and encode/decode text.
//!
//! Usage:
//!   echo "hello world" | cargo run -p payya-tokenizer --example tokenize
//!   cargo run -p payya-tokenizer --example tokenize -- --train "corpus text here" --vocab-size 280
//!   cargo run -p payya-tokenizer --example tokenize -- --encode "text to encode"

use std::io::Read;

use payya_tokenizer::Tokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Default: read stdin, train a small tokenizer, encode the input.
    if args.len() == 1 {
        let mut input = String::new();
        std::io::stdin()
            .read_to_string(&mut input)
            .expect("failed to read stdin");
        let input = input.trim_end();

        if input.is_empty() {
            eprintln!("Usage: echo \"text\" | cargo run -p payya-tokenizer --example tokenize");
            std::process::exit(1);
        }

        let tok = Tokenizer::train(input, 280);
        let ids = tok.encode(input);
        let decoded = tok.decode(&ids);

        println!("Input:      {:?}", input);
        println!("Vocab size: {}", tok.vocab_size());
        println!("Merges:     {}", tok.merges().len());
        println!("Token IDs:  {:?}", ids);
        println!("Tokens:     {}", ids.len());
        println!("Decoded:    {:?}", decoded);
        println!(
            "Compression: {:.1}x ({} bytes → {} tokens)",
            input.len() as f64 / ids.len() as f64,
            input.len(),
            ids.len()
        );
        return;
    }

    // Parse simple flags.
    let mut i = 1;
    let mut train_corpus: Option<String> = None;
    let mut vocab_size: usize = 280;
    let mut encode_text: Option<String> = None;

    while i < args.len() {
        match args[i].as_str() {
            "--train" => {
                i += 1;
                train_corpus = Some(args[i].clone());
            }
            "--vocab-size" => {
                i += 1;
                vocab_size = args[i].parse().expect("invalid --vocab-size");
            }
            "--encode" => {
                i += 1;
                encode_text = Some(args[i].clone());
            }
            other => {
                eprintln!("Unknown flag: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let corpus = train_corpus.unwrap_or_else(|| {
        let mut s = String::new();
        std::io::stdin()
            .read_to_string(&mut s)
            .expect("failed to read stdin");
        s
    });

    let tok = Tokenizer::train(&corpus, vocab_size);
    println!(
        "Trained tokenizer: {} merges, vocab size {}",
        tok.merges().len(),
        tok.vocab_size()
    );

    if let Some(text) = encode_text {
        let ids = tok.encode(&text);
        let decoded = tok.decode(&ids);
        println!("Input:   {:?}", text);
        println!("Tokens:  {:?}", ids);
        println!("Decoded: {:?}", decoded);
    } else {
        // Encode the training corpus itself.
        let ids = tok.encode(&corpus);
        println!(
            "Corpus: {} bytes → {} tokens ({:.1}x compression)",
            corpus.len(),
            ids.len(),
            corpus.len() as f64 / ids.len() as f64
        );
    }
}
