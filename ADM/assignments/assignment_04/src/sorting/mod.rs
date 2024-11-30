pub mod branchless;
pub mod branchsort;

pub fn sort(list: &mut [u32]) {
    let len = list.len();

    if cfg!(feature = "branching") {
        branchsort::sort(list, 0, len - 1);
    }

    if cfg!(feature = "branchless") {
        branchless::sort(list, 0, len - 1);
    }
}
