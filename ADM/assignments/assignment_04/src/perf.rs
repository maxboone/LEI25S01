use std::env;
use std::process::exit;

use perf_event::events::Hardware;
use perf_event::{Builder, Counter};

pub struct BranchMissTracker {
    counter: Option<Counter>,
}

impl Default for BranchMissTracker {
    fn default() -> Self {
        BranchMissTracker::new()
    }
}

impl BranchMissTracker {
    pub fn new() -> Self {
        if env::var("PERF").is_ok() {
            match Builder::new(Hardware::BRANCH_MISSES).build() {
                Ok(counter) => Self {
                    counter: Some(counter),
                },
                Err(err) => {
                    eprintln!("ERR - could not initialize branch miss tracker {:?}", err);
                    exit(1);
                }
            }
        } else {
            Self { counter: None }
        }
    }

    pub fn enable(&mut self) {
        if let Some(ref mut counter) = self.counter {
            if let Err(err) = counter.enable() {
                eprintln!("ERR - could not start branch miss counter {:?}", err);
                exit(1);
            }
        }
    }

    pub fn disable(&mut self) {
        if let Some(ref mut counter) = self.counter {
            if let Err(err) = counter.disable() {
                eprintln!("ERR - could not stop branch miss counter {:?}", err);
                exit(1);
            }
        }
    }

    pub fn read(&mut self) -> Option<u64> {
        self.counter.as_mut().map(|counter| match counter.read() {
            Ok(count) => count,
            Err(err) => {
                eprintln!("ERR - could not collect branch misses {:?}", err);
                exit(1);
            }
        })
    }
}
