use std::fs::File;
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};
use std::process::exit;

pub fn load(n: u32, path: PathBuf) -> Vec<u32> {
    let mut elements = Vec::<u32>::new();
    let mut count = 0;

    if let Ok(lines) = read_lines(path.clone()) {
        for line in lines {
            match line {
                Ok(line) => {
                    count += 1;

                    if count > n {
                        return elements;
                    }

                    match line.parse::<u32>() {
                        Ok(number) => elements.push(number),
                        Err(err) => {
                            eprintln!("ERR - failed to parse number from line: {}", err);
                            exit(1);
                        }
                    };
                }
                Err(err) => {
                    eprintln!("ERR - failed to retrieve line: {:?}", err);
                    exit(1);
                }
            }
        }
    }
    eprintln!(
        "INFO - loaded {} elements from {:?}",
        elements.len(),
        path.clone()
    );
    elements
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}
