//! Train a 2-layer MLP on XOR using payya-autograd.
//!
//! This demonstrates the autograd engine's ability to train a simple
//! neural network from scratch. The MLP has:
//! - Input: 2 features (the XOR inputs)
//! - Hidden: 8 neurons with ReLU activation
//! - Output: 1 neuron with sigmoid activation
//!
//! Run: `cargo run -p payya-autograd --example xor_mlp`

use payya_autograd::Graph;
use rand::Rng;

fn main() {
    // XOR dataset: 4 samples, 2 features each
    let inputs: Vec<[f32; 2]> = vec![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let targets: Vec<f32> = vec![0.0, 1.0, 1.0, 0.0];

    let mut rng = rand::thread_rng();

    // Network dimensions
    let input_dim = 2;
    let hidden_dim = 8;
    let output_dim = 1;

    // Initialize weights with Xavier/Glorot initialization
    let scale1 = (2.0 / (input_dim + hidden_dim) as f32).sqrt();
    let scale2 = (2.0 / (hidden_dim + output_dim) as f32).sqrt();

    let mut w1: Vec<f32> = (0..input_dim * hidden_dim)
        .map(|_| rng.gen_range(-scale1..scale1))
        .collect();
    let mut b1: Vec<f32> = vec![0.0; hidden_dim];
    let mut w2: Vec<f32> = (0..hidden_dim * output_dim)
        .map(|_| rng.gen_range(-scale2..scale2))
        .collect();
    let mut b2: Vec<f32> = vec![0.0; output_dim];

    let lr = 0.5;
    let epochs = 1000;
    let batch_size = inputs.len();

    println!("Training 2-layer MLP on XOR...");
    println!(
        "Architecture: {} -> {} (ReLU) -> {} (Sigmoid)",
        input_dim, hidden_dim, output_dim
    );
    println!("Learning rate: {lr}, Epochs: {epochs}");
    println!("---");

    for epoch in 0..epochs {
        // Build the full batch as a (4, 2) matrix
        let x_data: Vec<f32> = inputs.iter().flat_map(|x| x.iter().copied()).collect();
        let t_data: Vec<f32> = targets.clone();

        // Build computation graph
        let mut g = Graph::new();

        let x = g.tensor(&x_data, &[batch_size, input_dim]);
        let t = g.tensor(&t_data, &[batch_size, output_dim]);

        let w1_id = g.param(&w1, &[input_dim, hidden_dim]);
        let b1_id = g.param(&b1, &[hidden_dim]);
        let w2_id = g.param(&w2, &[hidden_dim, output_dim]);
        let b2_id = g.param(&b2, &[output_dim]);

        // Forward: h = relu(x @ W1 + b1)
        let h = g.matmul(x, w1_id);
        let h = g.add(h, b1_id);
        let h = g.relu(h);

        // Forward: y = sigmoid(h @ W2 + b2)
        let logits = g.matmul(h, w2_id);
        let logits = g.add(logits, b2_id);
        let y = g.sigmoid(logits);

        // Loss: MSE = sum((y - t)^2) (we don't divide by n; lr absorbs scaling)
        let diff = g.sub(y, t);
        let sq = g.mul(diff, diff);
        let loss = g.sum(sq);

        let loss_val = g.data(loss)[0] / batch_size as f32;
        let predictions: Vec<f32> = g.data(y).to_vec();

        // Backward
        g.backward(loss);

        // SGD update
        let g_w1 = g.grad(w1_id).to_vec();
        let g_b1 = g.grad(b1_id).to_vec();
        let g_w2 = g.grad(w2_id).to_vec();
        let g_b2 = g.grad(b2_id).to_vec();

        for (w, gw) in w1.iter_mut().zip(g_w1.iter()) {
            *w -= lr * gw / batch_size as f32;
        }
        for (b, gb) in b1.iter_mut().zip(g_b1.iter()) {
            *b -= lr * gb / batch_size as f32;
        }
        for (w, gw) in w2.iter_mut().zip(g_w2.iter()) {
            *w -= lr * gw / batch_size as f32;
        }
        for (b, gb) in b2.iter_mut().zip(g_b2.iter()) {
            *b -= lr * gb / batch_size as f32;
        }

        if epoch % 100 == 0 || epoch == epochs - 1 {
            let correct = predictions
                .iter()
                .zip(targets.iter())
                .filter(|(&p, &t)| (p > 0.5) == (t > 0.5))
                .count();
            let accuracy = correct as f32 / batch_size as f32 * 100.0;

            println!(
                "Epoch {epoch:4}: loss={loss_val:.6}, accuracy={accuracy:.0}%, \
                 predictions=[{:.3}, {:.3}, {:.3}, {:.3}]",
                predictions[0], predictions[1], predictions[2], predictions[3]
            );
        }
    }

    // Final evaluation
    println!("\n--- Final Results ---");
    let x_data: Vec<f32> = inputs.iter().flat_map(|x| x.iter().copied()).collect();
    let mut g = Graph::new();
    let x = g.tensor(&x_data, &[batch_size, input_dim]);
    let w1_id = g.tensor(&w1, &[input_dim, hidden_dim]);
    let b1_id = g.tensor(&b1, &[hidden_dim]);
    let w2_id = g.tensor(&w2, &[hidden_dim, output_dim]);
    let b2_id = g.tensor(&b2, &[output_dim]);

    let h = g.matmul(x, w1_id);
    let h = g.add(h, b1_id);
    let h = g.relu(h);
    let logits = g.matmul(h, w2_id);
    let logits = g.add(logits, b2_id);
    let y = g.sigmoid(logits);

    let predictions = g.data(y);
    let mut all_correct = true;
    for (i, (input, &target)) in inputs.iter().zip(targets.iter()).enumerate() {
        let pred = predictions[i];
        let label = if pred > 0.5 { 1 } else { 0 };
        let expected = target as u8;
        let status = if label == expected { "OK" } else { "WRONG" };
        if label != expected {
            all_correct = false;
        }
        println!(
            "  {:?} -> {:.4} (predicted={label}, expected={expected}) [{status}]",
            input, pred
        );
    }

    if all_correct {
        println!("\nXOR learned successfully! All predictions correct.");
    } else {
        println!("\nSome predictions wrong — try increasing epochs or adjusting lr.");
    }
}
