# Assignment 4

Branching and condition-assigned sorting implementation of Quick Sort in rust.
There is a single dependency on perf events to measure the branch misses in the
code. This is only used when the `PERF` environment flag is used and points to
the correct `PERF` install.

# Building

Get stable (1.82) rust version: [rustup](https://rustup.rs/).

Build applications:

```bash
cargo build --release --features branching --target-dir target/branching
cargo build --release --features condition --target-dir target/condition
cp target/branching/release/assignment_04 bin/branching
cp target/condition/release/assignment_04 bin/condition
```

## Build inside a container runtime (docker)

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

# Running

Generate a file with random integers to sort:

```bash
head -c $((100000 * 4)) /dev/urandom | od -An -tu4 | tr -s ' ' '\n' | tail -n+1 > ./data/test.csv
```

Then run the binaries, pointing to your local linux-perf install:

```bash
PERF=/usr/lib/linux-tools-6.8.0-49/perf ./bin/branching 100000 ./data/test.csv > ./data/test.csv.branching
PERF=/usr/lib/linux-tools-6.8.0-49/perf ./bin/condition 100000 ./data/test.csv > ./data/test.csv.condition
```

Test if the output is correct:

```bash
diff ./data/test.csv.branching <(cat ./data/test.csv | sort -n)
diff ./data/test.csv.condition <(cat ./data/test.csv | sort -n)
```
