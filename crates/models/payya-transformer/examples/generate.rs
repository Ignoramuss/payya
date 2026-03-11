//! Demo: train a tiny transformer on a text corpus and generate text.
//!
//! Usage:
//!   cargo run -p payya-transformer --example generate
//!   cargo run -p payya-transformer --example generate -- --generate "The meaning of"

use payya_logit_processor::LogitProcessor;
use payya_tokenizer::Tokenizer;
use payya_transformer::{PosEncoding, Transformer, TransformerConfig};
use rand::SeedableRng;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse --generate flag.
    let generate_prompt = args
        .windows(2)
        .find(|w| w[0] == "--generate")
        .map(|w| w[1].clone());

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Training corpus: diverse enough that the tokenizer doesn't collapse it.
    let corpus = "The cat sat on the mat and the dog ran in the park. \
                  A bird flew over the tall tree near the river bank. \
                  The sun was warm and the sky was clear and blue today. \
                  Small fish swim in the deep cold lake beside the hill.";

    // Train a BPE tokenizer with a modest vocabulary.
    let tokenizer = Tokenizer::train(corpus, 280);
    let tokens = tokenizer.encode(corpus);
    let vocab_size = tokenizer.vocab_size();

    println!(
        "Corpus: {} chars, {} tokens, vocab_size={}",
        corpus.len(),
        tokens.len(),
        vocab_size
    );

    // Configure a small transformer.
    let config = TransformerConfig {
        vocab_size,
        d_model: 64,
        n_heads: 4,
        n_layers: 2,
        d_ff: 128,
        max_seq_len: 128,
        pos_encoding: PosEncoding::Sinusoidal,
    };
    let mut model = Transformer::new(config, &mut rng);

    // Train for several epochs on sliding windows.
    let token_ids: Vec<usize> = tokens.iter().map(|&t| t as usize).collect();
    let window = 32.min(token_ids.len());
    let n_steps = 100;
    let stride = if token_ids.len() > window {
        token_ids.len() - window
    } else {
        1
    };

    println!("Training for {n_steps} steps (window={window})...");
    for step in 0..n_steps {
        let start = if stride > 1 { (step * 3) % stride } else { 0 };
        let end = (start + window).min(token_ids.len());
        let chunk = &token_ids[start..end];
        if chunk.len() < 2 {
            continue;
        }
        let loss = model.train_step(chunk, 0.005);
        if step % 10 == 0 {
            println!("  step {step:3}: loss={loss:.4}");
        }
    }

    // Generate text.
    let prompt_text = generate_prompt.as_deref().unwrap_or("The");
    let prompt_tokens: Vec<usize> = tokenizer
        .encode(prompt_text)
        .iter()
        .map(|&t| t as usize)
        .collect();

    println!(
        "\nPrompt: \"{prompt_text}\" ({} tokens)",
        prompt_tokens.len()
    );

    let processor = LogitProcessor::new().with_temperature(0.8).with_top_k(20);

    let generated = model.generate(&prompt_tokens, 30, &processor, &mut rng);
    let generated_ids: Vec<u32> = generated.iter().map(|&t| t as u32).collect();
    let text = tokenizer.decode(&generated_ids);

    println!("Generated: \"{text}\"");
}
