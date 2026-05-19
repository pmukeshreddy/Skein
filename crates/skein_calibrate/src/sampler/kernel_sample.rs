//! Kernel runtime sampler backed by real Luminal compile + execute.

use std::time::Instant;

use luminal::prelude::*;
use skein_compile::ComputeRuntime;
use skein_cost::OpKind;

use crate::corpus::{CalibrationCorpus, KernelSample};
use crate::error::CalibrationError;
use crate::hardware::HardwareSpec;
use crate::measurement::KernelMeasurement;

pub fn sample<R: ComputeRuntime>(
    corpus: &CalibrationCorpus,
    hardware: &HardwareSpec,
) -> Result<Vec<KernelMeasurement>, CalibrationError> {
    let mut measurements = Vec::new();
    for sample in &corpus.kernel_samples {
        for _ in 0..sample.repeats {
            let mut graph = build_kernel_graph(sample)?;
            let mut runtime = R::build_and_search(&mut graph.graph, 10)?;
            for input in &graph.inputs {
                runtime.set_data_f32(input.node, deterministic_data(input.len));
            }

            let start = Instant::now();
            runtime.execute(&graph.graph);
            let measured_us = start.elapsed().as_secs_f64() * 1_000_000.0;
            let theoretical_peak_us = compute_theoretical_peak_us(sample, hardware)?;
            measurements.push(KernelMeasurement {
                op_kind: sample.op_kind,
                dtype: sample.dtype,
                shape: sample.shape.clone(),
                measured_us,
                theoretical_peak_us,
            });
        }
    }
    Ok(measurements)
}

struct KernelGraph {
    graph: Graph,
    inputs: Vec<KernelInput>,
}

struct KernelInput {
    node: NodeIndex,
    len: usize,
}

fn build_kernel_graph(sample: &KernelSample) -> Result<KernelGraph, CalibrationError> {
    match sample.op_kind {
        OpKind::Gemm => build_gemm_graph(sample),
        OpKind::Attention => build_attention_graph(sample),
        OpKind::Elementwise => build_elementwise_graph(sample),
    }
}

fn build_gemm_graph(sample: &KernelSample) -> Result<KernelGraph, CalibrationError> {
    if sample.shape.len() != 2 {
        return Err(CalibrationError::UnsupportedSample {
            reason: format!(
                "gemm sample shape must be [out, in], got {:?}",
                sample.shape
            ),
        });
    }
    let out = sample.shape[0] as usize;
    let input = sample.shape[1] as usize;
    let mut graph = Graph::new();
    let x = graph.tensor((1usize, input));
    let w = graph.tensor((out, input));
    let _y = x.matmul(w.permute((1, 0))).output();
    Ok(KernelGraph {
        graph,
        inputs: vec![
            KernelInput {
                node: x.id,
                len: input,
            },
            KernelInput {
                node: w.id,
                len: out * input,
            },
        ],
    })
}

fn build_attention_graph(sample: &KernelSample) -> Result<KernelGraph, CalibrationError> {
    if sample.shape.len() != 4 {
        return Err(CalibrationError::UnsupportedSample {
            reason: format!(
                "attention sample shape must be [batch, heads, seq, head_dim], got {:?}",
                sample.shape
            ),
        });
    }
    let batch = sample.shape[0] as usize;
    let heads = sample.shape[1] as usize;
    let seq = sample.shape[2] as usize;
    let head_dim = sample.shape[3] as usize;
    let mut graph = Graph::new();
    let q = graph.tensor((batch, heads, seq, head_dim));
    let k = graph.tensor((batch, heads, head_dim, seq));
    let v = graph.tensor((batch, heads, seq, head_dim));
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let _out = (q.matmul(k) * scale).softmax(3).matmul(v).output();
    let len = batch * heads * seq * head_dim;
    Ok(KernelGraph {
        graph,
        inputs: vec![
            KernelInput { node: q.id, len },
            KernelInput { node: k.id, len },
            KernelInput { node: v.id, len },
        ],
    })
}

fn build_elementwise_graph(sample: &KernelSample) -> Result<KernelGraph, CalibrationError> {
    if sample.shape.is_empty() {
        return Err(CalibrationError::UnsupportedSample {
            reason: "elementwise sample shape is empty".to_string(),
        });
    }
    let shape: Vec<usize> = sample.shape.iter().map(|d| *d as usize).collect();
    let len = shape.iter().product::<usize>();
    let mut graph = Graph::new();
    let x = tensor_for_shape(&mut graph, &shape)?;
    let _out = x.std_norm(x.shape.last_axis(), 1.0e-5).output();
    Ok(KernelGraph {
        graph,
        inputs: vec![KernelInput { node: x.id, len }],
    })
}

fn tensor_for_shape(graph: &mut Graph, shape: &[usize]) -> Result<GraphTensor, CalibrationError> {
    Ok(match shape {
        [a] => graph.tensor((*a,)),
        [a, b] => graph.tensor((*a, *b)),
        [a, b, c] => graph.tensor((*a, *b, *c)),
        [a, b, c, d] => graph.tensor((*a, *b, *c, *d)),
        _ => {
            return Err(CalibrationError::UnsupportedSample {
                reason: format!("unsupported elementwise rank {}", shape.len()),
            });
        }
    })
}

fn deterministic_data(len: usize) -> Vec<f32> {
    (0..len).map(|i| ((i % 31) as f32 - 15.0) / 31.0).collect()
}

fn compute_theoretical_peak_us(
    sample: &KernelSample,
    hardware: &HardwareSpec,
) -> Result<f64, CalibrationError> {
    let peak_tflops =
        hardware
            .peak_tflops(sample.dtype)
            .ok_or_else(|| CalibrationError::UnsupportedSample {
                reason: format!(
                    "hardware {} has no peak_tflops entry for {:?}",
                    hardware.kind, sample.dtype
                ),
            })?;
    let flops = estimate_flops(sample)?;
    Ok(flops / (peak_tflops * 1.0e12) * 1.0e6)
}

fn estimate_flops(sample: &KernelSample) -> Result<f64, CalibrationError> {
    match sample.op_kind {
        OpKind::Gemm => {
            if sample.shape.len() != 2 {
                return Err(CalibrationError::UnsupportedSample {
                    reason: format!(
                        "gemm sample shape must be [out, in], got {:?}",
                        sample.shape
                    ),
                });
            }
            Ok(2.0 * sample.shape[0] as f64 * sample.shape[1] as f64)
        }
        OpKind::Attention => {
            if sample.shape.len() != 4 {
                return Err(CalibrationError::UnsupportedSample {
                    reason: format!(
                        "attention sample shape must be [batch, heads, seq, head_dim], got {:?}",
                        sample.shape
                    ),
                });
            }
            let batch = sample.shape[0] as f64;
            let heads = sample.shape[1] as f64;
            let seq = sample.shape[2] as f64;
            let head_dim = sample.shape[3] as f64;
            Ok(4.0 * batch * heads * seq * seq * head_dim)
        }
        OpKind::Elementwise => {
            let n = sample.shape.iter().map(|d| *d as f64).product::<f64>();
            Ok(8.0 * n)
        }
    }
}
