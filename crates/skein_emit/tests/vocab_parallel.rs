//! Vocab-parallel embedding + lm_head — hand-computed reference.
//!
//! Each TP rank holds a vocab slice. `vocab_parallel_embed` zeros tokens
//! outside its slice; summing the per-rank outputs (the runtime AllReduce)
//! must reconstruct the full embedding. The lm_head is the dual: each rank
//! computes logits for its vocab slice and an AllGather concatenates them
//! into the full logit vector.

use luminal::op::Runtime;
use luminal::prelude::*;
use skein_emit::op_wiring::vocab_parallel_embed;

/// Run one rank's vocab-parallel embedding and return the flat output.
fn run_rank(
    table_rows: &[f32],
    vocab_start: usize,
    vocab_local: usize,
    tokens: &[f32],
) -> Vec<f32> {
    let hidden = 2usize;
    let seq = tokens.len();
    let mut cx = Graph::new();
    // `vocab_parallel_embed` casts the token tensor internally, so feeding the
    // ids as f32 exercises the same path as the Int runtime input.
    let tok = cx.named_tensor("tok", (1usize, seq));
    let table = cx.named_tensor("table", (vocab_local, hidden));
    let out = vocab_parallel_embed(tok, table, vocab_start, vocab_local, 1, seq, hidden).output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);
    rt.set_data(tok.id, tokens.to_vec());
    rt.set_data(table.id, table_rows.to_vec());
    rt.execute(&cx.dyn_map);
    rt.get_f32(out.id).clone()
}

#[test]
fn vocab_parallel_embed_sums_to_full_embedding() {
    // Full vocab=4, hidden=2: row v = [2v+1, 2v+2].
    //   rank 0 owns ids 0,1 -> [[1,2],[3,4]]
    //   rank 1 owns ids 2,3 -> [[5,6],[7,8]]
    let tokens = [0.0_f32, 2.0, 1.0, 3.0];
    let r0 = run_rank(&[1.0, 2.0, 3.0, 4.0], 0, 2, &tokens);
    let r1 = run_rank(&[5.0, 6.0, 7.0, 8.0], 2, 2, &tokens);

    // Each rank zeros the tokens it does not own.
    assert_eq!(r0, vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]);
    assert_eq!(r1, vec![0.0, 0.0, 5.0, 6.0, 0.0, 0.0, 7.0, 8.0]);

    // AllReduce(sum) reconstructs the true embeddings for [0,2,1,3].
    let summed: Vec<f32> = r0.iter().zip(&r1).map(|(a, b)| a + b).collect();
    assert_eq!(summed, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
}

#[test]
fn lm_head_vocab_shards_allgather_to_full_logits() {
    // lm_head is a plain matmul against a vocab-sharded weight: rank r holds
    // weight rows [r*Vloc, (r+1)*Vloc) and produces logits[.., r*Vloc..]. An
    // AllGather concatenates the shards into the full logit vector. We verify
    // that concatenation equals the unsharded matmul.
    let hidden = 2usize;
    let full_vocab = 4usize;
    let normed = [1.0_f32, 1.0]; // one token, hidden=2
    let full_w = [
        1.0_f32, 0.0, // row 0
        0.0, 1.0, // row 1
        1.0, 1.0, // row 2
        2.0, 2.0, // row 3
    ];

    let shard_logits = |rows: &[f32], vloc: usize| -> Vec<f32> {
        let mut cx = Graph::new();
        let x = cx.named_tensor("x", (1usize, hidden));
        let w = cx.named_tensor("w", (vloc, hidden));
        let y = x.matmul(w.permute((1, 0))).output();
        cx.build_search_space::<NativeRuntime>();
        let mut rt = cx.search(NativeRuntime::default(), 1);
        rt.set_data(x.id, normed.to_vec());
        rt.set_data(w.id, rows.to_vec());
        rt.execute(&cx.dyn_map);
        rt.get_f32(y.id).clone()
    };

    let r0 = shard_logits(&full_w[0..4], 2); // rows 0,1
    let r1 = shard_logits(&full_w[4..8], 2); // rows 2,3
    let gathered: Vec<f32> = r0.iter().chain(&r1).copied().collect();

    // Reference: full matmul.
    let reference: Vec<f32> = (0..full_vocab)
        .map(|v| normed[0] * full_w[v * 2] + normed[1] * full_w[v * 2 + 1])
        .collect();
    assert_eq!(gathered, reference);
}
