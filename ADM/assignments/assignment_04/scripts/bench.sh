#!/bin/bash

for f in l_discount.csv l_extendedprice.csv l_orderkey.csv l_tax.csv data.csv; do
	for i in $(seq 1000 5000 100000); do
		# extra run to load file into memory
		stderr_output=$(./bin/branching $i ./data/$f 2>&1 >/dev/null)
		stderr_output=$(./bin/branching $i ./data/$f 2>&1 >/dev/null)
		echo $stderr_output
		# extra run to load file into memory
		stderr_output=$(./bin/condition $i ./data/$f 2>&1 >/dev/null)
		stderr_output=$(./bin/condition $i ./data/$f 2>&1 >/dev/null)
		echo $stderr_output
	done
done
