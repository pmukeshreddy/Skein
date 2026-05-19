//! Real Mac sampler coverage.

mod common;

use skein_calibrate::corpus::CalibrationCorpus;
use skein_calibrate::hardware::HardwareSpec;
use skein_calibrate::sampler::sample_kernel_runtimes;
use skein_compile::NativeComputeRuntime;

const TINY_CORPUS: &str = r#"
workload_trace_path = "/tmp/anywhere.jsonl"
n_drift_prompts     = 1
sampling_strategy   = "first_n"

[[kernel_samples]]
op_kind = "gemm"
dtype = "bf16"
shape = [8, 8]
repeats = 1

[[kernel_samples]]
op_kind = "attention"
dtype = "bf16"
shape = [1, 1, 4, 8]
repeats = 1

[[kernel_samples]]
op_kind = "elementwise"
dtype = "bf16"
shape = [1, 1, 16]
repeats = 1
"#;

#[test]
fn sample_kernel_runtimes_native_produces_real_data() {
    let corpus = CalibrationCorpus::from_toml_str(TINY_CORPUS).unwrap();
    let constants = common::load_cost_constants();
    let hardware = HardwareSpec::new("h100_sxm5").with_cost_constants(&constants);
    let measurements = sample_kernel_runtimes::<NativeComputeRuntime>(&corpus, &hardware).unwrap();
    assert_eq!(measurements.len(), 3);
    for m in &measurements {
        assert!(m.measured_us > 0.0, "{m:?}");
        let eff = m.efficiency();
        assert!(eff > 0.0 && eff <= 1.0, "{m:?} efficiency={eff}");
    }
}
