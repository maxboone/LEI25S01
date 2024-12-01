#!/bin/bash

cargo build --features branching --target-dir target/branching
cargo build --features condition --target-dir target/condition
cp target/branching/debug/assignment_04 bin/branching
cp target/condition/debug/assignment_04 bin/condition
