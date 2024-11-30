#!/bin/bash

cargo build --release --features branching --target-dir target/branching
cargo build --release --features condition --target-dir target/condition
cp target/branching/release/assignment_04 bin/branching
cp target/condition/release/assignment_04 bin/condition
