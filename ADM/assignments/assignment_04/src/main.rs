use assignment_04::sorting;

pub mod args;
pub mod load;

fn main() {
    let (n_elements, path) = args::args();
    let mut elements = load::load(n_elements, path);

    sorting::sort(elements.as_mut());
    for i in elements {
        println!("{}", i)
    }
}
