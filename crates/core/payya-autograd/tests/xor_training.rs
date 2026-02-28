//! Integration test: train a 2-layer MLP on XOR to >95% accuracy.
//!
//! This validates the full training loop: forward pass, loss computation,
//! backward pass (gradient computation), and SGD parameter updates.

use payya_autograd::Graph;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[test]
fn train_xor_mlp_to_convergence() {
    let inputs: Vec<[f32; 2]> = vec![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let targets: Vec<f32> = vec![0.0, 1.0, 1.0, 0.0];

    // Deterministic seed for reproducibility
    let mut rng = StdRng::seed_from_u64(42);

    let input_dim = 2;
    let hidden_dim = 8;
    let output_dim = 1;
    let batch_size = inputs.len();

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
    let epochs = 500;

    let x_data: Vec<f32> = inputs.iter().flat_map(|x| x.iter().copied()).collect();
    let t_data: Vec<f32> = targets.clone();

    let mut final_accuracy = 0.0;
    let mut final_loss = f32::MAX;

    for _epoch in 0..epochs {
        let mut g = Graph::new();

        let x = g.tensor(&x_data, &[batch_size, input_dim]);
        let t = g.tensor(&t_data, &[batch_size, output_dim]);
        let w1_id = g.param(&w1, &[input_dim, hidden_dim]);
        let b1_id = g.param(&b1, &[hidden_dim]);
        let w2_id = g.param(&w2, &[hidden_dim, output_dim]);
        let b2_id = g.param(&b2, &[output_dim]);

        // Forward
        let h = g.matmul(x, w1_id);
        let h = g.add(h, b1_id);
        let h = g.relu(h);
        let logits = g.matmul(h, w2_id);
        let logits = g.add(logits, b2_id);
        let y = g.sigmoid(logits);

        // Loss
        let diff = g.sub(y, t);
        let sq = g.mul(diff, diff);
        let loss = g.sum(sq);

        final_loss = g.data(loss)[0] / batch_size as f32;
        let predictions: Vec<f32> = g.data(y).to_vec();

        let correct = predictions
            .iter()
            .zip(targets.iter())
            .filter(|(&p, &t)| (p > 0.5) == (t > 0.5))
            .count();
        final_accuracy = correct as f32 / batch_size as f32 * 100.0;

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
    }

    assert!(
        final_accuracy >= 95.0,
        "MLP should achieve >95% accuracy on XOR, got {final_accuracy}%"
    );
    assert!(
        final_loss < 0.1,
        "Loss should converge below 0.1, got {final_loss}"
    );
}
