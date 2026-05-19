//! Test 7 — weight slicing byte ranges, with a synthetic safetensors fixture.
//!
//! Generated data inside this test is the *only* place in skein_emit's tests
//! where synthetic numbers appear. Their purpose is to exercise the slicing
//! arithmetic — not to model real weights — and the scope is limited to the
//! 4×4 fixture below.

mod common;

use std::collections::HashMap;

use skein_emit::{ShardStrategy, weights};
use skein_ir::types::Dtype;

#[test]
fn outer_axis_slice_byte_ranges_cover_full_tensor() {
    // 4×4 i8 source tensor. Outer-axis split into two halves: rows 0–1
    // and rows 2–3 → 8 bytes each, contiguous.
    let source_shape = vec![4usize, 4usize];
    let half_a = ShardStrategy::OuterAxisSlice { start: 0, end: 2 };
    let half_b = ShardStrategy::OuterAxisSlice { start: 2, end: 4 };

    let a_ranges = half_a.source_byte_ranges(&source_shape, Dtype::Int8);
    let b_ranges = half_b.source_byte_ranges(&source_shape, Dtype::Int8);
    assert_eq!(a_ranges, vec![0..8]);
    assert_eq!(b_ranges, vec![8..16]);

    // No overlap, no gaps: union is exactly 0..16.
    let mut covered = [false; 16];
    for r in a_ranges.iter().chain(b_ranges.iter()) {
        for byte in r.clone() {
            assert!(!covered[byte as usize], "byte {byte} double-covered");
            covered[byte as usize] = true;
        }
    }
    assert!(covered.iter().all(|c| *c), "some bytes left uncovered");
}

#[test]
fn inner_axis_slice_byte_ranges_are_row_strided() {
    let source_shape = vec![4usize, 4usize];
    let left = ShardStrategy::InnerAxisSlice { start: 0, end: 2 };
    let right = ShardStrategy::InnerAxisSlice { start: 2, end: 4 };

    let left_ranges = left.source_byte_ranges(&source_shape, Dtype::Int8);
    let right_ranges = right.source_byte_ranges(&source_shape, Dtype::Int8);
    // 4 rows, each contributes a 2-byte range stride 4 apart.
    assert_eq!(left_ranges, vec![0..2, 4..6, 8..10, 12..14]);
    assert_eq!(right_ranges, vec![2..4, 6..8, 10..12, 14..16]);

    // Coverage: each byte exactly once.
    let mut covered = [false; 16];
    for r in left_ranges.iter().chain(right_ranges.iter()) {
        for byte in r.clone() {
            assert!(!covered[byte as usize]);
            covered[byte as usize] = true;
        }
    }
    assert!(covered.iter().all(|c| *c));
}

#[test]
fn write_weight_shard_end_to_end_with_synthetic_fixture() {
    // Build a 4×4 i8 tensor with values 0..16 and serialize via safetensors.
    let source_dir = tempfile_dir("skein_emit_weight_test");
    let source_path = source_dir.join("weights.safetensors");
    {
        let data: Vec<u8> = (0..16u8).collect();
        let view = safetensors::tensor::TensorView::new(safetensors::Dtype::I8, vec![4, 4], &data)
            .unwrap();
        let bytes =
            safetensors::serialize(std::iter::once(("test.weight".to_string(), view)), &None)
                .unwrap();
        std::fs::write(&source_path, &bytes).unwrap();
    }

    // Hand-build a `WeightShard` that takes the outer half (rows 0–1) of
    // the source. The full lower_per_device path is exercised by other
    // tests; here we focus on the write side.
    let shard = weights::WeightShard {
        device_idx: 0,
        slices: vec![weights::WeightSlice {
            source_key: "test.weight".into(),
            source_shape: vec![4, 4],
            source_dtype: Dtype::Int8,
            dest_shape: vec![2, 4],
            strategy: ShardStrategy::OuterAxisSlice { start: 0, end: 2 },
        }],
        total_bytes: 8,
    };

    let dest_path = source_dir.join("device_0.safetensors");
    weights::write_weight_shard(&shard, &source_dir, &dest_path).expect("write shard");

    // Re-read the destination and verify shape + bytes.
    let written = std::fs::read(&dest_path).unwrap();
    let st = safetensors::SafeTensors::deserialize(&written).unwrap();
    let view = st.tensor("test.weight").unwrap();
    assert_eq!(view.shape(), &[2, 4]);
    assert_eq!(view.data(), &[0u8, 1, 2, 3, 4, 5, 6, 7]);

    // Repeat for device 1 (rows 2–3).
    let shard1 = weights::WeightShard {
        device_idx: 1,
        slices: vec![weights::WeightSlice {
            source_key: "test.weight".into(),
            source_shape: vec![4, 4],
            source_dtype: Dtype::Int8,
            dest_shape: vec![2, 4],
            strategy: ShardStrategy::OuterAxisSlice { start: 2, end: 4 },
        }],
        total_bytes: 8,
    };
    let dest_path1 = source_dir.join("device_1.safetensors");
    weights::write_weight_shard(&shard1, &source_dir, &dest_path1).expect("write shard 1");
    let written1 = std::fs::read(&dest_path1).unwrap();
    let st1 = safetensors::SafeTensors::deserialize(&written1).unwrap();
    let view1 = st1.tensor("test.weight").unwrap();
    assert_eq!(view1.shape(), &[2, 4]);
    assert_eq!(view1.data(), &[8u8, 9, 10, 11, 12, 13, 14, 15]);

    // Two devices' bytes concatenate to the full source.
    let _ = HashMap::<String, ()>::new(); // silence unused import
}

fn tempfile_dir(prefix: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "{prefix}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}
