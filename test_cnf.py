import sys
from solve_cnf import Hashi_solver, read_file
from generate_test_case import HashiGenerator
import os

def test_hashi_solver(test_num, grid, output_dir="output"):
    """
    Test the Hashi solver with a given grid
    """
    print(f"\n{'='*60}")
    print(f"TEST CASE #{test_num}")
    print(f"{'='*60}")
    
    # Print the grid
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
        # Convert to CNF
        solver.generate_cnf(
            dimacs_path=f"{output_dir}/dimacs/test_case_{str(len(grid))}_{test_num}.dimacs",
            varmap_path=f"{output_dir}/varmap/test_case_{str(len(grid))}_{test_num}_varmap.txt",
            lazy=True
        )
        print("\n✓ CNF conversion successful")
        
        # Solve using SAT solver
        solution = solver.solve_lazy()
        if solution["status"] == "SAT":
            print("✓ Solution found")
            print("\nSolution Grid:")
            grid = solver.structure_solver(name=f"output/test_case_{str(len(grid))}_{test_num}_solution.txt")
            solver.draw_hashi()
            for row in grid:
                print(" , ".join(str(x) for x in row))
            return True
        else:
            print("✗ No solution found")
            with open(f"output/test_case_{str(len(grid))}_{test_num}_solution.txt", "w") as f:
                f.write("No solution found for this test case.\n")
            return False
            
    except Exception as e:
        print(f"✗ Error during solving: {str(e)}")
        return False

if __name__ == "__main__":
    """
    Generate 10 test cases and test the Hashi solver with each
    """
    # Create output directories
    os.makedirs("output/dimacs", exist_ok=True)
    os.makedirs("output/varmap", exist_ok=True)
    
    # Test case sizes: 10 different cases with varying dimensions
    test_sizes = [
        (7, 7),
        (7, 7),
        (9, 9),
        (9, 9),
        (11, 11),
        (11, 11),
        (13, 13),
        (13, 13),
        (17, 17),
        (20, 20)
    ]
    
    print("="*60)
    print("HASHI SOLVER - 10 TEST CASES")
    print("="*60)
    
    successful_tests = 0
    failed_tests = 0
    
    for i, (height, width) in enumerate(test_sizes, start=1):
        print(f"\nGenerating test case #{i} (Grid: {height}x{width})")
        
        # Generate test case
        gen = HashiGenerator(width, height)
        grid = gen.generate()
        
        # Test the solver
        if test_hashi_solver(i, grid):
            successful_tests += 1
        else:
            failed_tests += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {len(test_sizes)}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success rate: {successful_tests/len(test_sizes)*100:.1f}%")
    print(f"{'='*60}\n")

