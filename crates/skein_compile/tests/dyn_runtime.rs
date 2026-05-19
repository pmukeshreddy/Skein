//! DynRuntimeWrapper tests.

use std::collections::HashMap;

use luminal::prelude::*;
use skein_compile::{DynRuntime, DynRuntimeWrapper, NativeComputeRuntime, compile_with_luminal};

#[test]
fn dyn_runtime_wrapper_sets_executes_and_gets_by_name() {
    let mut cx = Graph::new();
    let input = cx.named_tensor("input", (2usize, 2usize));
    let weights = cx.named_tensor("weights", (2usize, 2usize));
    let out = input.matmul(weights).output();

    let mut runtimes =
        compile_with_luminal::<NativeComputeRuntime>(std::slice::from_mut(&mut cx), 1)
            .expect("compile tiny named graph");
    let runtime = runtimes.pop().expect("one runtime");

    let mut names = HashMap::new();
    names.insert("input".to_string(), input.id);
    names.insert("weights".to_string(), weights.id);
    names.insert("out".to_string(), out.id);

    let mut dyn_rt = DynRuntimeWrapper::new(runtime, cx, names);
    dyn_rt
        .set_tensor_by_name("input", vec![1.0, 2.0, 3.0, 4.0])
        .expect("set input");
    dyn_rt
        .set_tensor_by_name("weights", vec![1.0, 0.0, 0.0, 1.0])
        .expect("set weights");
    dyn_rt.execute_segment().expect("execute");

    let got = dyn_rt.get_tensor_by_name("out").expect("get output");
    assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0]);
    assert!(dyn_rt.get_tensor_by_name("missing").is_err());
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_dyn_runtime_wrapper_typechecks() {
    fn _accept(
        runtime: skein_compile::CudaComputeRuntime,
        graph: luminal::prelude::Graph,
        names: HashMap<String, luminal::prelude::NodeIndex>,
    ) -> Box<dyn DynRuntime> {
        Box::new(DynRuntimeWrapper::new(runtime, graph, names))
    }
    let _f: fn(_, _, _) -> Box<dyn DynRuntime> = _accept;
}
