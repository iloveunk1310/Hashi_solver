# ============ TYPE DEFINITIONS ============
from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Iterable, Callable
from dataclasses import dataclass
import heapq
import itertools
import time
import math
import statistics
import pandas as pd

Assignment = List[Optional[bool]]

# ============ HELPER FUNCTIONS ============

def make_empty_assignment(n: int) -> Assignment:
    """Create an empty assignment list of length n+1 (index 1..n)."""
    return [None] * (n + 1)

def eval_literal(lit: int, assignment: Assignment) -> Optional[bool]:
    """Evaluate a single literal under assignment."""
    v = abs(lit)
    val = assignment[v]
    if val is None:
        return None
    return val if lit > 0 else (not val)

def clause_status(clause: Iterable[int], assignment: Assignment) -> str:
    """
    Return one of {"SAT","FALSIFIED","UNDECIDED"} for a clause.
    SAT        : at least one literal True
    FALSIFIED  : all literals False
    UNDECIDED  : no True, but some None
    """
    undecided = False
    for lit in clause:
        v = eval_literal(lit, assignment)
        if v is True:
            return "SAT"
        if v is None:
            undecided = True
    return "UNDECIDED" if undecided else "FALSIFIED"

def cnf_status(cnf: Dict, assignment: Assignment) -> Dict[str, int | bool]:
    """Summarize CNF status under partial assignment."""
    sat = undec = fals = 0
    for clause in cnf["clauses"]:
        st = clause_status(clause, assignment)
        if st == "SAT":
            sat += 1
        elif st == "UNDECIDED":
            undec += 1
        else:
            fals += 1
    return {
        "all_satisfied": (undec == 0 and fals == 0),
        "contradiction": (fals > 0),
        "sat_clauses": sat,
        "undecided_clauses": undec,
        "falsified_clauses": fals,
    }

def unit_propagate(cnf: Dict, assignment: Assignment) -> Tuple[Assignment, bool, int]:
    """Perform unit propagation to fixpoint."""
    n = cnf["num_vars"]
    a = assignment[:]
    implied = 0
    while True:
        changed = False
        for clause in cnf["clauses"]:
            st = clause_status(clause, a)
            if st == "SAT":
                continue
            if st == "FALSIFIED":
                return a, True, implied

            undecided_lits = []
            all_false = True
            for lit in clause:
                val = eval_literal(lit, a)
                if val is True:
                    all_false = False
                    break
                if val is None:
                    undecided_lits.append(lit)

            if st != "SAT":
                if len(undecided_lits) == 0:
                    return a, True, implied
                if len(undecided_lits) == 1 and all_false:
                    only = undecided_lits[0]
                    var = abs(only)
                    val = (only > 0)
                    if a[var] is not None and a[var] != val:
                        return a, True, implied
                    if a[var] is None:
                        a[var] = val
                        implied += 1
                        changed = True
                        break
        if not changed:
            break
    return a, False, implied

def choose_var(cnf: Dict, assignment: Assignment, strategy: str = "most_frequent") -> int:
    """Pick an unassigned variable."""
    n = cnf["num_vars"]
    unassigned = {i for i in range(1, n + 1) if assignment[i] is None}
    if not unassigned:
        return 0

    if strategy != "most_frequent":
        return min(unassigned)

    freq: Dict[int, int] = {}
    for clause in cnf["clauses"]:
        st = clause_status(clause, assignment)
        if st == "UNDECIDED":
            for lit in clause:
                v = abs(lit)
                if v in unassigned:
                    freq[v] = freq.get(v, 0) + 1

    if not freq:
        return min(unassigned)
    maxf = max(freq.values())
    candidates = [v for v, f in freq.items() if f == maxf]
    return min(candidates)

def verify_solution(cnf: Dict, assignment: Optional[Assignment]) -> bool:
    """Check if assignment satisfies all clauses."""
    if assignment is None:
        return False
    st = cnf_status(cnf, assignment)
    return bool(st["all_satisfied"] and not st["contradiction"])

# ============ SOLVERS ============

def brute_force_solve(cnf: Dict, time_limit_s: Optional[float] = None) -> Dict:
    """Brute force: try all 2^n assignments."""
    start = time.perf_counter()
    n = cnf["num_vars"]
    nodes = 0
    checks = 0
    timeout = False

    def now() -> float:
        return time.perf_counter() - start

    for bits in itertools.product([False, True], repeat=n):
        if time_limit_s is not None and now() > time_limit_s:
            timeout = True
            break
        nodes += 1
        a = [None] + list(bits)
        all_sat = True
        for clause in cnf["clauses"]:
            checks += 1
            satisfied = False
            for lit in clause:
                v = eval_literal(lit, a)
                if v is True:
                    satisfied = True
                    break
            if not satisfied:
                all_sat = False
                break
        if all_sat:
            return {
                "sat": True,
                "assignment": a,
                "nodes": nodes,
                "checks": checks,
                "time": now(),
                "timeout": False,
                "status": "SAT",
            }

    elapsed = time.perf_counter() - start
    status = "UNKNOWN" if timeout else "UNSAT"

    return {
        "sat": False,
        "assignment": None,
        "nodes": nodes,
        "checks": checks,
        "time": elapsed,
        "timeout": timeout,
        "status": status,
    }

def backtracking_solve(cnf: Dict, use_unit: bool = True, time_limit_s: Optional[float] = None) -> Dict:
    """DFS with optional unit propagation."""
    start = time.perf_counter()
    nodes = 0
    timeout = False
    n = cnf["num_vars"]

    def now() -> float:
        return time.perf_counter() - start

    def dfs(a: Assignment) -> Tuple[bool, Optional[Assignment]]:
        nonlocal nodes, timeout
        if time_limit_s is not None and now() > time_limit_s:
            timeout = True
            return False, None

        nodes += 1

        if use_unit:
            a, clash, _ = unit_propagate(cnf, a)
            if clash:
                return False, None

        st = cnf_status(cnf, a)
        if st["contradiction"]:
            return False, None
        if st["all_satisfied"]:
            return True, a

        v = choose_var(cnf, a)
        if v == 0:
            return False, None

        for val in (True, False):
            a2 = a[:]
            a2[v] = val
            ok, sol = dfs(a2)
            if timeout:
                return False, None
            if ok:
                return True, sol
        return False, None

    empty = make_empty_assignment(n)
    sat, sol = dfs(empty)
    elapsed = time.perf_counter() - start

    status = "SAT" if sat else ("UNKNOWN" if timeout else "UNSAT")

    return {
        "sat": sat,
        "assignment": sol if sat else None,
        "nodes": nodes,
        "time": elapsed,
        "timeout": timeout,
        "status": status,
    }

def h_zero(cnf: Dict, assignment: Assignment) -> int:
    """Heuristic: h = 0 (UCS)."""
    return 0

def h_undecided_clauses(cnf: Dict, assignment: Assignment) -> int:
    """Heuristic: number of undecided clauses."""
    st = cnf_status(cnf, assignment)
    return int(st["undecided_clauses"])

def a_star_solve(cnf: Dict, heuristic: str = "zero", use_unit: bool = True, 
                time_limit_s: Optional[float] = None) -> Dict:
    """A* search with choice of heuristic."""
    start = time.perf_counter()
    timeout = False
    nodes = 0
    n = cnf["num_vars"]

    H: Callable[[Dict, Assignment], int]
    H = h_zero if heuristic == "zero" else h_undecided_clauses

    def now() -> float:
        return time.perf_counter() - start

    a0 = make_empty_assignment(n)
    if use_unit:
        a0, clash, _ = unit_propagate(cnf, a0)
        if clash:
            return {
                "sat": False,
                "assignment": None,
                "nodes": 0,
                "time": now(),
                "timeout": False,
                "status": "UNSAT",
            }

    st0 = cnf_status(cnf, a0)
    if st0["all_satisfied"]:
        return {
            "sat": True,
            "assignment": a0,
            "nodes": 0,
            "time": now(),
            "timeout": False,
            "status": "SAT",
        }

    g0 = sum(1 for i in range(1, n + 1) if a0[i] is not None)
    h0 = H(cnf, a0)
    f0 = g0 + h0

    counter = 0
    open_heap: List[Tuple[int, int, int, Tuple[Optional[bool], ...], Assignment]] = []
    key0 = tuple(a0[1:])
    heapq.heappush(open_heap, (f0, g0, counter, key0, a0))
    counter += 1

    closed: set[Tuple[Optional[bool], ...]] = set()

    while open_heap:
        if time_limit_s is not None and now() > time_limit_s:
            timeout = True
            break

        f, g, _, key, a = heapq.heappop(open_heap)
        if key in closed:
            continue
        closed.add(key)
        nodes += 1

        st = cnf_status(cnf, a)
        if st["all_satisfied"]:
            return {
                "sat": True,
                "assignment": a,
                "nodes": nodes,
                "time": now(),
                "timeout": False,
                "status": "SAT",
            }

        v = choose_var(cnf, a)
        if v == 0:
            continue

        for val in (True, False):
            a2 = a[:]
            a2[v] = val
            if use_unit:
                a2, clash, _ = unit_propagate(cnf, a2)
                if clash:
                    continue

            st2 = cnf_status(cnf, a2)
            if st2["contradiction"]:
                continue

            key2 = tuple(a2[1:])
            if key2 in closed:
                continue

            g2 = sum(1 for i in range(1, n + 1) if a2[i] is not None)
            h2 = H(cnf, a2)
            f2 = g2 + h2
            heapq.heappush(open_heap, (f2, g2, counter, key2, a2))
            counter += 1

    elapsed = time.perf_counter() - start
    status = "UNKNOWN" if timeout else "UNSAT"

    return {
        "sat": False,
        "assignment": None,
        "nodes": nodes,
        "time": elapsed,
        "timeout": timeout,
        "status": status,
    }


import random

# Sinh instance SAT ngẫu nhiên
# Mỗi clause có k literal
def random_k_sat(n_vars: int, n_clauses: int, k: int = 3, seed: Optional[int] = None) -> Dict:
    """
    Generate random k-SAT:
    - mỗi clause có k biến khác nhau
    - không tautology (và ít clause trùng)
    """
    rng = random.Random(seed)
    clauses: List[List[int]] = []
    for _ in range(n_clauses):
        vars_in_clause = rng.sample(range(1, n_vars + 1), k)
        clause = []
        for v in vars_in_clause:
            sign = rng.choice([1, -1])
            clause.append(sign * v)
        clauses.append(clause)
    return {"num_vars": n_vars, "clauses": clauses}

# Chạy solver theo tên, gom lại kết quả thành dict
def run_solver(solver_name: str, cnf: Dict, time_limit_s: Optional[float]) -> Dict:
    if solver_name == "brute_force":
        r = brute_force_solve(cnf, time_limit_s=time_limit_s)
    elif solver_name == "backtracking":
        r = backtracking_solve(cnf, use_unit=True, time_limit_s=time_limit_s)
    elif solver_name == "astar_zero":
        r = a_star_solve(cnf, heuristic="zero", use_unit=True, time_limit_s=time_limit_s)
    elif solver_name == "astar_undecided":
        r = a_star_solve(cnf, heuristic="undecided", use_unit=True, time_limit_s=time_limit_s)
    else:
        raise ValueError(solver_name)
    r["solver"] = solver_name
    return r


# Chạy nhiều solver trên nhiều CNF, repeat nhiều lần
# Kết quả thu thành Dataframe
def benchmark_solvers(
    cnf_list: List[Tuple[str, Dict]],
    repeats: int = 5,
    time_limit_s: Optional[float] = 5.0,
    solvers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    columns: instance_id, n_vars, n_clauses, solver,
             sat, status, time_sec, nodes, timeout,
             verified, time_median_over_repeats
    """
    if solvers is None:
        solvers = ["brute_force", "backtracking", "astar_zero", "astar_undecided"]

    rows = []
    for inst_id, cnf in cnf_list:
        n_vars = cnf["num_vars"]
        n_clauses = len(cnf["clauses"])
        for solver in solvers:
            times = []
            last = None
            for _ in range(repeats):
                res = run_solver(solver, cnf, time_limit_s=time_limit_s)
                last = res
                times.append(res["time"])
            assert last is not None
            verified = False
            if last["sat"] and not last["timeout"]:
                verified = verify_solution(cnf, last["assignment"])
            rows.append({
                "instance_id": inst_id,
                "n_vars": n_vars,
                "n_clauses": n_clauses,
                "solver": solver,
                "sat": bool(last["sat"]),
                "status": last.get("status", ""),
                "time_sec": float(last["time"]),
                "nodes": int(last["nodes"]),
                "timeout": bool(last["timeout"]),
                "verified": bool(verified),
                "time_median_over_repeats": float(statistics.median(times)),
            })
    df = pd.DataFrame(rows)
    return df
if __name__ == "__main__":

    def pretty_assignment(a: Optional[Assignment]) -> str:
        if a is None:
            return "None"
        return " ".join(
            f"x{i}={( 'T' if a[i] else 'F')}" if a[i] is not None else f"x{i}=."
            for i in range(1, len(a))
        )

    # Test instances
    CNF_SAT = {
        "num_vars": 2,
        "clauses": [
            [1],
            [-1, 2],
        ]
    }
    CNF_UNSAT = {
        "num_vars": 1,
        "clauses": [
            [1],
            [-1],
        ]
    }
    CNF_SAT_DONTCARE = {
        "num_vars": 3,
        "clauses": [
            [1],
        ]
    }


    tests = [
        ("SAT_basic", CNF_SAT, True),
        ("UNSAT_basic", CNF_UNSAT, False),
        ("SAT_dontcare", CNF_SAT_DONTCARE, True),
    ]

    for name, cnf, expect_sat in tests:
        bf = brute_force_solve(cnf, time_limit_s=2.0)
        a0 = a_star_solve(cnf, heuristic="zero", use_unit=True, time_limit_s=2.0)
        a1 = a_star_solve(cnf, heuristic="undecided", use_unit=True, time_limit_s=2.0)

        assert bf["sat"] == expect_sat, f"brute-force failed on {name}"
        assert a0["sat"] == expect_sat, f"A* h=0 failed on {name}"
        assert a1["sat"] == expect_sat, f"A* h=undecided failed on {name}"

        print("Basic sanity tests passed.")

    BIG_BENCH = [
    ("R20_80",  random_k_sat(20, 80)),
    ("R25_100", random_k_sat(25, 100)),
    ("R30_120", random_k_sat(30, 120)),
    ]
    print(random_k_sat(20, 80))
    df_big = benchmark_solvers(BIG_BENCH, repeats=3, time_limit_s=5.0)
    print(df_big)

