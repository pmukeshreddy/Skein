use std::collections::HashMap;

use skein_compile::{DynRuntime, DynRuntimeError};
use skein_cost::collectives::CollectiveKind;
use skein_runtime::{
    CollectiveBackend, DispatchOutcome, EagerDispatcher, InProcessCollective, KernelDispatcher,
    StepBatch,
};

#[test]
fn in_process_collective_ring_allreduce() {
    let mut runtimes = vec![
        MapRuntime::new("x", vec![1.0, 2.0, 3.0]),
        MapRuntime::new("x", vec![2.0, 3.0, 4.0]),
        MapRuntime::new("x", vec![3.0, 4.0, 5.0]),
        MapRuntime::new("x", vec![4.0, 5.0, 6.0]),
    ];
    let mut refs = runtimes
        .iter_mut()
        .map(|r| r as &mut dyn DynRuntime)
        .collect::<Vec<_>>();
    InProcessCollective::new(4)
        .execute(
            CollectiveKind::RingAllReduce,
            &[0, 1, 2, 3],
            "x",
            refs.as_mut_slice(),
        )
        .unwrap();
    for rt in runtimes {
        assert_eq!(rt.get_tensor_by_name("x").unwrap(), vec![10.0, 14.0, 18.0]);
    }
}

#[test]
fn in_process_collective_allgather() {
    let mut runtimes = vec![
        MapRuntime::new("x", vec![1.0, 2.0, 3.0]),
        MapRuntime::new("x", vec![4.0, 5.0, 6.0]),
    ];
    let mut refs = runtimes
        .iter_mut()
        .map(|r| r as &mut dyn DynRuntime)
        .collect::<Vec<_>>();
    InProcessCollective::new(2)
        .execute(CollectiveKind::AllGather, &[0, 1], "x", refs.as_mut_slice())
        .unwrap();
    for rt in runtimes {
        assert_eq!(
            rt.get_tensor_by_name("x").unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }
}

#[test]
fn in_process_collective_alltoall() {
    let mut runtimes = [
        MapRuntime::new("x", vec![1.0, 2.0, 3.0, 4.0]),
        MapRuntime::new("x", vec![5.0, 6.0, 7.0, 8.0]),
    ];
    let mut refs = runtimes
        .iter_mut()
        .map(|r| r as &mut dyn DynRuntime)
        .collect::<Vec<_>>();
    InProcessCollective::new(2)
        .execute(CollectiveKind::AllToAll, &[0, 1], "x", refs.as_mut_slice())
        .unwrap();
    assert_eq!(
        runtimes[0].get_tensor_by_name("x").unwrap(),
        vec![1.0, 2.0, 5.0, 6.0]
    );
    assert_eq!(
        runtimes[1].get_tensor_by_name("x").unwrap(),
        vec![3.0, 4.0, 7.0, 8.0]
    );
}

#[test]
fn eager_dispatcher_calls_execute_once() {
    let mut runtime = CountingRuntime::default();
    let mut dispatcher = EagerDispatcher;
    let batch = StepBatch {
        prefill_requests: Vec::new(),
        decode_requests: Vec::new(),
        total_kv_pages: 0,
        uniform_decode_size: None,
    };
    let outcome = dispatcher.dispatch(&batch, &mut runtime).unwrap();
    assert_eq!(outcome, DispatchOutcome::Eager);
    assert_eq!(runtime.execs, 1);
}

struct MapRuntime {
    tensors: HashMap<String, Vec<f32>>,
}

impl MapRuntime {
    fn new(name: &str, data: Vec<f32>) -> Self {
        let mut tensors = HashMap::new();
        tensors.insert(name.to_string(), data);
        Self { tensors }
    }
}

impl DynRuntime for MapRuntime {
    fn execute_segment(&mut self) -> Result<(), DynRuntimeError> {
        Ok(())
    }

    fn get_tensor_by_name(&self, name: &str) -> Result<Vec<f32>, DynRuntimeError> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| DynRuntimeError::UnknownTensor(name.to_string()))
    }

    fn set_tensor_by_name(&mut self, name: &str, data: Vec<f32>) -> Result<(), DynRuntimeError> {
        self.tensors.insert(name.to_string(), data);
        Ok(())
    }

    fn set_tensor_i32_by_name(
        &mut self,
        name: &str,
        data: Vec<i32>,
    ) -> Result<(), DynRuntimeError> {
        self.set_tensor_by_name(name, data.into_iter().map(|v| v as f32).collect())
    }
}

#[derive(Default)]
struct CountingRuntime {
    execs: usize,
}

impl DynRuntime for CountingRuntime {
    fn execute_segment(&mut self) -> Result<(), DynRuntimeError> {
        self.execs += 1;
        Ok(())
    }

    fn get_tensor_by_name(&self, name: &str) -> Result<Vec<f32>, DynRuntimeError> {
        Err(DynRuntimeError::UnknownTensor(name.to_string()))
    }

    fn set_tensor_by_name(&mut self, _name: &str, _data: Vec<f32>) -> Result<(), DynRuntimeError> {
        Ok(())
    }

    fn set_tensor_i32_by_name(
        &mut self,
        _name: &str,
        _data: Vec<i32>,
    ) -> Result<(), DynRuntimeError> {
        Ok(())
    }
}
