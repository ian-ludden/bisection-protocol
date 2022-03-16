#!/bin/bash

K=4
for fname in grid_instances/4x6*
do
	echo $fname >> out/opt_recursive_bisection.out
	echo "K=4" >> out/opt_recursive_bisection.out
	echo "D first" >> out/opt_recursive_bisection.out
	python bisection_optimality_grid_graphs_recursive.py enumerations/enum_[4,6]_[6]_4_rc.txt 6 4 $fname "D" >> out/opt_recursive_bisection.out
	echo "R first" >> out/opt_recursive_bisection.out
	python bisection_optimality_grid_graphs_recursive.py enumerations/enum_[4,6]_[6]_4_rc.txt 6 4 $fname "R" >> out/opt_recursive_bisection.out
done

for fname in grid_instances/6x6*
do
	echo $fname >> out/opt_recursive_bisection.out
	echo "K=4" >> out/opt_recursive_bisection.out
	echo "D first" >> out/opt_recursive_bisection.out
	python bisection_optimality_grid_graphs_recursive.py enumerations/enum_[6,6]_[9]_4_rc.txt 6 6 $fname "D" >> out/opt_recursive_bisection.out
	echo "R first" >> out/opt_recursive_bisection.out
	python bisection_optimality_grid_graphs_recursive.py enumerations/enum_[6,6]_[9]_4_rc.txt 6 6 $fname "R" >> out/opt_recursive_bisection.out
done

K=6
for fname in grid_instances/4x6*
do
	echo $fname >> out/opt_recursive_bisection.out
	echo "K=6" >> out/opt_recursive_bisection.out
	echo "D first" >> out/opt_recursive_bisection.out
	python bisection_optimality_grid_graphs_recursive.py enumerations/enum_[4,6]_[4]_6_rc.txt 6 4 $fname "D" >> out/opt_recursive_bisection.out
	echo "R first" >> out/opt_recursive_bisection.out
	python bisection_optimality_grid_graphs_recursive.py enumerations/enum_[4,6]_[4]_6_rc.txt 6 4 $fname "R" >> out/opt_recursive_bisection.out
done

for fname in grid_instances/6x6*
do
	echo $fname >> out/opt_recursive_bisection.out
	echo "K=6" >> out/opt_recursive_bisection.out
	echo "D first" >> out/opt_recursive_bisection.out
	python bisection_optimality_grid_graphs_recursive.py enumerations/enum_[6,6]_[6]_6_rc.txt 6 6 $fname "D" >> out/opt_recursive_bisection.out
	echo "R first" >> out/opt_recursive_bisection.out
	python bisection_optimality_grid_graphs_recursive.py enumerations/enum_[6,6]_[6]_6_rc.txt 6 6 $fname "R" >> out/opt_recursive_bisection.out
done
