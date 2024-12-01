use assignment_04::sorting;
use std::time::Instant;

pub mod args;
pub mod load;

fn main() {
    let (n_elements, path) = args::args();
    let mut elements = load::load(n_elements, path);

    let now = Instant::now();
    sorting::sort(elements.as_mut());
    let elapsed = now.elapsed();
    eprintln!("INFO - elapsed: {:?}", elapsed);
    for i in elements {
        println!("{}", i);
    }
}
