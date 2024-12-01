use std::process::exit;

pub mod branching;
pub mod condition;

pub fn sort(list: &mut [u32]) {
    let len = list.len();

    if len < 2 {
        return;
    }

    if cfg!(feature = "branching") {
        eprintln!("INFO - sorting using branching implementation");
        branching::sort(list);
        return;
    }

    if cfg!(feature = "condition") {
        eprintln!("INFO - sorting using condition-assign implementation");
        condition::sort(list);
        return;
    }

    eprintln!("ERR - array not sorted, binary was not compiled with --features branching or --features condition");
    exit(1);
}
