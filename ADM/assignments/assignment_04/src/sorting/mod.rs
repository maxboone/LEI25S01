pub mod branching;
pub mod condition;

pub fn sort(list: &mut [u32]) {
    let len = list.len();

    if cfg!(feature = "branching") {
        eprintln!("sorting using branching implementation");
        branching::sort(list, 0, len - 1);
        return;
    }

    if cfg!(feature = "condition") {
        eprintln!("sorting using condition-assign implementation");
        condition::sort(list, 0, len - 1);
        return;
    }

    eprintln!("array not sorted, binary was not compiled with --features branching or --features condition");
}
