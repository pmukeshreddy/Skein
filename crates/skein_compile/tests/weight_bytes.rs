//! Byte-level weight decoding — hand-computed bit patterns.

use skein_compile::{WeightDtype, decode_weight_bytes};

#[test]
fn decode_bf16_bytes() {
    // bf16 is the top 16 bits of f32: 1.0 = 0x3F80, 2.0 = 0x4000, -1.0 = 0xBF80.
    let bytes = [0x80, 0x3F, 0x00, 0x40, 0x80, 0xBF];
    let got = decode_weight_bytes("w", &bytes, WeightDtype::Bf16).unwrap();
    assert_eq!(got, vec![1.0, 2.0, -1.0]);
}

#[test]
fn decode_f16_bytes() {
    // f16: 1.0 = 0x3C00, 0.5 = 0x3800, 2.0 = 0x4000.
    let bytes = [0x00, 0x3C, 0x00, 0x38, 0x00, 0x40];
    let got = decode_weight_bytes("w", &bytes, WeightDtype::F16).unwrap();
    assert_eq!(got, vec![1.0, 0.5, 2.0]);
}

#[test]
fn decode_f32_bytes_round_trips() {
    let values = [1.5_f32, -3.25, 1024.0, 0.0];
    let mut bytes = Vec::new();
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let got = decode_weight_bytes("w", &bytes, WeightDtype::F32).unwrap();
    assert_eq!(got, values);
}

#[test]
fn decode_rejects_misaligned_byte_length() {
    // Three bytes is not a whole number of bf16 (2-byte) elements.
    let err = decode_weight_bytes("w", &[0x00, 0x3C, 0x00], WeightDtype::Bf16);
    assert!(err.is_err(), "odd byte length must error");
}
