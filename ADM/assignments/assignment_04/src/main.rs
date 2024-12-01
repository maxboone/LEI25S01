use assignment_04::sorting;
use perf_event::events::Hardware;
use perf_event::Builder;
use std::process::exit;
use std::time::Instant;

pub mod args;
pub mod load;

fn main() {
    let mut counter = match Builder::new(Hardware::BRANCH_MISSES).build() {
        Ok(counter) => counter,
        Err(err) => {
            eprintln!("ERR - could not initialize branch miss tracker {:?}", err);
            exit(1);
        }
    };

    let (n_elements, path) = args::args();
    let mut elements = load::load(n_elements, path);

    if let Err(err) = counter.enable() {
        eprintln!("ERR - could not start branch miss counter {:?}", err);
        exit(1);
    }

    let now = Instant::now();
    sorting::sort(elements.as_mut());
    let elapsed = now.elapsed();

    if let Err(err) = counter.disable() {
        eprintln!("ERR - could not stop branch miss counter {:?}", err);
        exit(1);
    }

    eprintln!("INFO - elapsed: {:?}", elapsed);
    let count = match counter.read() {
        Ok(count) => count,
        Err(err) => {
            eprintln!("ERR - could not collect branch misses {:?}", err);
            exit(1);
        }
    };
    eprintln!("INFO - branch misses: {:?}", count);

    for i in elements {
        println!("{}", i);
    }
}
