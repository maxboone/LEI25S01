use std::{
    env,
    path::{Path, PathBuf},
    process::exit,
};

pub fn args() -> (u32, PathBuf) {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!(
            "ERR - Please provide two arguments, provided: {:?}, exiting.",
            args
        );
        exit(1);
    }

    let n_elements = args[1].parse::<u32>().unwrap();
    let path = Path::new(&args[2]).to_path_buf();
    eprintln!(
        "INFO - Loading {:} elements from file {:?}",
        n_elements, path
    );
    (n_elements, path)
}
