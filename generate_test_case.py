import random

class HashiGenerator:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(width)] for _ in range(height)]

    def generate(self):
        nodes = []
        for r in range(0, self.height, 2): # Cách 1 dòng có đảo
            for c in range(0, self.width, 2): # Cách 1 cột có đảo
                if random.random() < 0.9:
                    nodes.append((r, c))

        h_bridges = {}
        v_bridges = {}

        # Duyệt qua các node để nối ngang
        for r, c in nodes:
            next_c = c + 2
            if (r, next_c) in nodes:
                val = random.choices([0, 1, 2], weights=[0.3, 0.35, 0.35])[0]
                if val > 0:
                    h_bridges[(r, c)] = val

        # Duyệt qua các node để nối dọc
        for r, c in nodes:
            # Tìm hàng xóm bên dưới
            next_r = r + 2
            if (next_r, c) in nodes:
                # Random nối dọc
                val = random.choices([0, 1, 2], weights=[0.3, 0.35, 0.35])[0]
                if val > 0:
                    v_bridges[(r, c)] = val

        final_grid = [[0 for _ in range(self.width)] for _ in range(self.height)]

        valid_islands_count = 0

        for r, c in nodes:
            count = 0
            # + Cầu nối sang trái (từ node bên trái nối sang)
            if (r, c-2) in h_bridges: count += h_bridges[(r, c-2)]
            # + Cầu nối sang phải (tự mình nối sang phải)
            if (r, c) in h_bridges: count += h_bridges[(r, c)]
            # + Cầu nối lên trên
            if (r-2, c) in v_bridges: count += v_bridges[(r-2, c)]
            # + Cầu nối xuống dưới
            if (r, c) in v_bridges: count += v_bridges[(r, c)]

            if count > 0:
                final_grid[r][c] = count
                valid_islands_count += 1

        if valid_islands_count > 0:
            start_node = None
            for r in range(self.height):
                for c in range(self.width):
                    if final_grid[r][c] > 0:
                        start_node = (r, c)
                        break
                if start_node: break

            # BFS đếm số đảo đi tới được
            visited = set()
            queue = [start_node]
            visited.add(start_node)
            count_visited = 0

            while queue:
                ur, uc = queue.pop(0)
                count_visited += 1

                # Trái
                if (ur, uc-2) in h_bridges and (ur, uc-2) not in visited:
                    visited.add((ur, uc-2)); queue.append((ur, uc-2))
                # Phải
                if (ur, uc) in h_bridges and (ur, uc+2) not in visited:
                    visited.add((ur, uc+2)); queue.append((ur, uc+2))
                # Trên
                if (ur-2, uc) in v_bridges and (ur-2, uc) not in visited:
                    visited.add((ur-2, uc)); queue.append((ur-2, uc))
                # Dưới
                if (ur, uc) in v_bridges and (ur+2, uc) not in visited:
                    visited.add((ur+2, uc)); queue.append((ur+2, uc))

            # Nếu số đảo đi được < tổng số đảo -> Không liên thông -> Generate lại
            if count_visited < valid_islands_count:
                if random.random() < 0.5: #tạo nhiễu
                    return self.generate()

        return final_grid

    def print_grid(self, grid):
        for row in grid:
            print(" , ".join(str(x) for x in row))
        print("\n")

sizes = [
    (7,7),
    (9,9),
    (11,11)
]

for i, (h, w) in enumerate(sizes):
    print(f"TEST CASE #{i+1}")
    gen = HashiGenerator(w, h)
    grid = gen.generate()
    gen.print_grid(grid)