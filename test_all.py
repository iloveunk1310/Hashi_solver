

from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Iterable, Callable
from solve_cnf import Hashi_solver
from generate_test_case import HashiGenerator
import heapq

import os
from traditional_algo import *
# HASHI TEST 

def test_hashi_with_solvers(test_num, grid, output_dir="output", time_limit_s=10.0):
    """Test Hashi solver with all CNF solvers."""
    print(f"\n{'='*70}")
    print(f"TEST CASE #{test_num} - Grid Size: {len(grid)}x{len(grid[0])}")
    print(f"{'='*70}")
    
    print("Input Grid:")
    for row in grid:
        print(" , ".join(str(x) for x in row))
    #write grid to file
    with open(f"input/test_case_{str(len(grid))}_{test_num}.txt", "w") as f:
        for row in grid:
            f.write(" , ".join(str(x) for x in row) + "\n")
    # Create solver instance
    solver = Hashi_solver(hashi_matrix=grid, size=len(grid))
    
    try:
        # Generate CNF
        solver.generate_cnf(
            dimacs_path=f"{output_dir}/dimacs/test_case_{str(len(grid))}_{test_num}.dimacs",
            varmap_path=f"{output_dir}/varmap/test_case_{str(len(grid))}_{test_num}_varmap.txt",
            lazy=True
        )
        print("\n✓ CNF generated successfully")
        
        # Read the CNF from DIMACS file
        dimacs_file = f"{output_dir}/dimacs/test_case_{test_num}.dimacs"
        cnf = read_dimacs(dimacs_file)
        
        print(f"CNF Stats: {cnf['num_vars']} variables, {len(cnf['clauses'])} clauses")
        
        results = {}
        solvers_to_test = ["brute_force", "backtracking", "astar_zero", "astar_undecided"]
        
        for solver_name in solvers_to_test:
            print(f"\n  Testing {solver_name}...", end=" ")
            if solver_name == "brute_force":
                res = brute_force_solve(cnf, time_limit_s=time_limit_s)
            elif solver_name == "backtracking":
                res = backtracking_solve(cnf, use_unit=True, time_limit_s=time_limit_s)
            elif solver_name == "astar_zero":
                res = a_star_solve(cnf, heuristic="zero", use_unit=True, time_limit_s=time_limit_s)
            elif solver_name == "astar_undecided":
                res = a_star_solve(cnf, heuristic="undecided", use_unit=True, time_limit_s=time_limit_s)
            
            results[solver_name] = res
            
            if res["sat"]:
                verified = verify_solution(cnf, res["assignment"])
                print(f"✓ SAT (verified={verified}, nodes={res['nodes']}, time={res['time']:.3f}s)")
            elif res["timeout"]:
                print(f"⏱ TIMEOUT (time={res['time']:.3f}s)")
            else:
                print(f"✗ UNSAT (nodes={res['nodes']}, time={res['time']:.3f}s)")
        
        # Try to solve with Hashi's internal solver
        print(f"\n  Testing Hashi native solver...", end=" ")
        solution = solver.solve_lazy()
        if solution["status"] == "SAT" :
            print(f"✓ Solution found")
            grid_sol = solver.structure_solver(name=f"output/test_case_{str(len(grid))}_{test_num}_solution.txt")
            print("\nSolution Grid:")
            for row in grid_sol:
                print(" , ".join(str(x) for x in row))
            return True, results
        else:
            print(f"✗ No solution found")
            with open(f"output/test_case_{str(len(grid))}_{test_num}_solution.txt", "w") as f:
                f.write("No solution found for this test case.\n")
            return False, results
            
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, {}

def read_dimacs(filename: str) -> Dict:
    """Read DIMACS CNF format."""
    clauses = []
    num_vars = 0
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            if line.startswith('p'):
                parts = line.split()
                num_vars = int(parts[2])
                continue
            if line and not line.startswith('p'):
                clause = [int(x) for x in line.split() if x != '0']
                if clause:
                    clauses.append(clause)
    return {"num_vars": num_vars, "clauses": clauses}

if __name__ == "__main__":
    """Run comprehensive tests."""
    os.makedirs("output/dimacs", exist_ok=True)
    os.makedirs("output/varmap", exist_ok=True)
    
    test_sizes = [
        (7, 7),
        (7, 7),
        (9, 9),
        (9, 9)
    ]
    
    print("="*70)
    print("COMPREHENSIVE CNF SOLVER TEST - HASHI PUZZLES")
    print("="*70)
    
    successful_tests = 0
    failed_tests = 0
    all_results = {}
    
    for i, (height, width) in enumerate(test_sizes, start=1):
        print(f"\n[TEST {i}/10] Generating Hashi puzzle ({height}x{width})...")
        
        gen = HashiGenerator(width, height)
        grid = gen.generate()
        
        success, results = test_hashi_with_solvers(i, grid, time_limit_s=10.0 * grid.__len__())
        all_results[i] = results
        
        if success:
            successful_tests += 1
        else:
            failed_tests += 1
    
    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total tests: {len(test_sizes)}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success rate: {successful_tests/len(test_sizes)*100:.1f}%")
    print(f"{'='*70}\n")
    
    # Solver comparison
    print("SOLVER PERFORMANCE SUMMARY:")
    print(f"{'='*70}")
    solver_stats = {name: {"success": 0, "time": 0, "nodes": 0} 
                    for name in ["brute_force", "backtracking", "astar_zero", "astar_undecided"]}
    
    for test_id, results in all_results.items():
        for solver_name, res in results.items():
            if res["sat"]:
                solver_stats[solver_name]["success"] += 1
                solver_stats[solver_name]["time"] += res["time"]
                solver_stats[solver_name]["nodes"] += res["nodes"]
    
    for solver_name, stats in solver_stats.items():
        if stats["success"] > 0:
            avg_time = stats["time"] / stats["success"]
            avg_nodes = stats["nodes"] / stats["success"]
            print(f"{solver_name:20} | Success: {stats['success']}/10 | Avg Time: {avg_time:.3f}s | Avg Nodes: {avg_nodes:.0f}")
        else:
            print(f"{solver_name:20} | Success: 0/10")
    
