# Assignment 4

Branching and condition-assigned sorting implementation of Quick Sort in rust.

## Building

Branching and condition

Get stable rust version: [rustup](https://rustup.rs/).

Build applications:

```bash
cargo build --release --features branching --target-dir target/branching
cargo build --release --features condition --target-dir target/condition
cp target/branching/release/assignment_04 bin/branching
cp target/condition/release/assignment_04 bin/condition
```

## Build with container runtime (docker)

This will execute the build inside a docker container, meaning the rust
toolchain is not required locally.

```bash
cat << EOF | docker run \
  --rm \
  --user "$(id -u)":"$(id -g)" \
  -i \
  -v "$PWD":/usr/src/assignment_04 \
  -w /usr/src/assignment_04 \
  rust:1.82 /bin/bash
cargo build --release --features branching --target-dir target/branching
cargo build --release --features condition --target-dir target/condition
cp target/branching/release/assignment_04 bin/branching
cp target/condition/release/assignment_04 bin/condition
EOF
```
