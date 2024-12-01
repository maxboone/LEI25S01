use assignment_04::sorting;
use std::time::Instant;

pub mod args;
pub mod load;
pub mod perf;

fn main() {
    let mut tracker = perf::BranchMissTracker::new();
    let (n_elements, path) = args::args();
    let mut elements = load::load(n_elements, path);

    tracker.enable();
    let now = Instant::now();
    sorting::sort(elements.as_mut());
    let elapsed = now.elapsed();
    tracker.disable();

    eprintln!("INFO - elapsed: {:?}", elapsed);
    if let Some(count) = tracker.read() {
        eprintln!("INFO - branch misses: {:?}", count);
    }

    for i in elements {
        println!("{}", i);
    }
}
