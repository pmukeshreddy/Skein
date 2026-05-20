//! CPU reference implementation of [`RankCollective`] using OS threads.
//!
//! Each rank runs on its own thread and meets the others at a
//! [`std::sync::Barrier`]; the leader performs the reduction over every rank's
//! contributed buffer and publishes the result, which all ranks then read.
//! This is *real* concurrent rank-parallel execution — it exercises the same
//! contribute → rendezvous → reduce → read sequence NCCL performs across
//! GPUs — so it serves as the GPU-free validation of the multi-rank
//! architecture. Construct one group and hand each spawned rank-thread its
//! [`BarrierCollective`] handle.

use std::sync::{Arc, Barrier, Mutex};

use super::{CollectiveError, RankCollective};

/// Shared state for a barrier-synchronized collective group.
struct Group {
    world_size: usize,
    barrier: Barrier,
    /// Each rank's contributed buffer for the in-flight op.
    slots: Mutex<Vec<Vec<f32>>>,
    /// The op's result, published by the leader, read by every rank.
    result: Mutex<Vec<f32>>,
}

/// One rank's handle on a [`Group`]. Cheap to clone-by-`Arc`; hand one to each
/// rank thread.
#[derive(Clone)]
pub struct BarrierCollective {
    group: Arc<Group>,
    rank: usize,
}

impl BarrierCollective {
    /// Build a group of `world_size` rank handles. Spawn one thread per handle;
    /// each thread must issue the *same sequence* of collective calls (as the
    /// shared decode schedule guarantees) so the barrier stays aligned.
    pub fn group(world_size: usize) -> Result<Vec<BarrierCollective>, CollectiveError> {
        if world_size == 0 {
            return Err(CollectiveError::BadLayout {
                rank: 0,
                world_size,
            });
        }
        let group = Arc::new(Group {
            world_size,
            barrier: Barrier::new(world_size),
            slots: Mutex::new(vec![Vec::new(); world_size]),
            result: Mutex::new(Vec::new()),
        });
        Ok((0..world_size)
            .map(|rank| BarrierCollective {
                group: group.clone(),
                rank,
            })
            .collect())
    }

    fn contribute(&self, buf: &[f32]) -> Result<(), CollectiveError> {
        let mut slots = self.group.slots.lock().map_err(|_| CollectiveError::Poisoned)?;
        slots[self.rank] = buf.to_vec();
        Ok(())
    }

    fn read_result(&self, buf: &mut [f32]) -> Result<(), CollectiveError> {
        let result = self.group.result.lock().map_err(|_| CollectiveError::Poisoned)?;
        if result.len() != buf.len() {
            return Err(CollectiveError::LengthMismatch {
                rank: self.rank,
                got: buf.len(),
                expected: result.len(),
            });
        }
        buf.copy_from_slice(&result);
        Ok(())
    }
}

impl RankCollective for BarrierCollective {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.group.world_size
    }

    fn all_reduce_sum(&self, buf: &mut [f32]) -> Result<(), CollectiveError> {
        self.contribute(buf)?;
        // All ranks have contributed.
        let leader = self.group.barrier.wait().is_leader();
        if leader {
            let slots = self.group.slots.lock().map_err(|_| CollectiveError::Poisoned)?;
            let len = slots[self.rank].len();
            let mut acc = vec![0.0f32; len];
            for (r, s) in slots.iter().enumerate() {
                if s.len() != len {
                    return Err(CollectiveError::LengthMismatch {
                        rank: r,
                        got: s.len(),
                        expected: len,
                    });
                }
                for (a, v) in acc.iter_mut().zip(s) {
                    *a += *v;
                }
            }
            *self.group.result.lock().map_err(|_| CollectiveError::Poisoned)? = acc;
        }
        // Result is published.
        self.group.barrier.wait();
        self.read_result(buf)?;
        // Everyone has read before the next op overwrites the slots.
        self.group.barrier.wait();
        Ok(())
    }

    fn all_gather(&self, buf: &[f32]) -> Result<Vec<f32>, CollectiveError> {
        self.contribute(buf)?;
        let leader = self.group.barrier.wait().is_leader();
        if leader {
            let slots = self.group.slots.lock().map_err(|_| CollectiveError::Poisoned)?;
            let mut gathered = Vec::with_capacity(slots.iter().map(|s| s.len()).sum());
            for s in slots.iter() {
                gathered.extend_from_slice(s);
            }
            *self.group.result.lock().map_err(|_| CollectiveError::Poisoned)? = gathered;
        }
        self.group.barrier.wait();
        let out = self
            .group
            .result
            .lock()
            .map_err(|_| CollectiveError::Poisoned)?
            .clone();
        self.group.barrier.wait();
        Ok(out)
    }

    fn broadcast(&self, buf: &mut [f32], root: usize) -> Result<(), CollectiveError> {
        if self.rank == root {
            *self.group.result.lock().map_err(|_| CollectiveError::Poisoned)? = buf.to_vec();
        }
        self.group.barrier.wait();
        self.read_result(buf)?;
        self.group.barrier.wait();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    /// Run `world_size` rank threads, each invoking `body(rank, handle)`, and
    /// collect their return values in rank order.
    fn run_ranks<T, F>(world_size: usize, body: F) -> Vec<T>
    where
        T: Send + 'static,
        F: Fn(usize, BarrierCollective) -> T + Send + Sync + 'static,
    {
        let handles = BarrierCollective::group(world_size).expect("group");
        let body = Arc::new(body);
        let mut joins = Vec::new();
        for h in handles {
            let body = body.clone();
            let rank = h.rank();
            joins.push(thread::spawn(move || body(rank, h)));
        }
        let mut out: Vec<Option<T>> = (0..world_size).map(|_| None).collect();
        for (i, j) in joins.into_iter().enumerate() {
            out[i] = Some(j.join().expect("rank thread"));
        }
        out.into_iter().map(|o| o.unwrap()).collect()
    }

    #[test]
    fn all_reduce_sum_across_threads() {
        // Rank r contributes [r+1, r+1, r+1]; every rank must end with the
        // elementwise sum 1+2+3+4 = 10.
        let results = run_ranks(4, |rank, h| {
            let mut buf = vec![(rank as f32) + 1.0; 3];
            h.all_reduce_sum(&mut buf).expect("all_reduce");
            buf
        });
        for (rank, buf) in results.iter().enumerate() {
            assert_eq!(buf, &vec![10.0f32; 3], "rank {rank} got {buf:?}");
        }
    }

    #[test]
    fn all_gather_concatenates_in_rank_order() {
        let results = run_ranks(3, |rank, h| {
            let buf = vec![rank as f32];
            h.all_gather(&buf).expect("all_gather")
        });
        for (rank, g) in results.iter().enumerate() {
            assert_eq!(g, &vec![0.0, 1.0, 2.0], "rank {rank} gathered {g:?}");
        }
    }

    #[test]
    fn broadcast_from_root() {
        let results = run_ranks(4, |rank, h| {
            let mut buf = if rank == 2 {
                vec![7.0, 8.0]
            } else {
                vec![0.0, 0.0]
            };
            h.broadcast(&mut buf, 2).expect("broadcast");
            buf
        });
        for (rank, buf) in results.iter().enumerate() {
            assert_eq!(buf, &vec![7.0, 8.0], "rank {rank} got {buf:?}");
        }
    }

    #[test]
    fn repeated_ops_stay_barrier_aligned() {
        // Two collectives back-to-back must not race on the shared slots.
        let results = run_ranks(3, |rank, h| {
            let mut a = vec![(rank + 1) as f32; 2];
            h.all_reduce_sum(&mut a).expect("ar1");
            let mut b = vec![1.0f32; 2];
            h.all_reduce_sum(&mut b).expect("ar2");
            (a, b)
        });
        for (rank, (a, b)) in results.iter().enumerate() {
            assert_eq!(a, &vec![6.0f32; 2], "rank {rank} a={a:?}"); // 1+2+3
            assert_eq!(b, &vec![3.0f32; 2], "rank {rank} b={b:?}"); // 1+1+1
        }
    }
}
