import heapq
import numpy as np

class AStarPlanner:
    def __init__(self,grid_map):
        self.grid = grid_map
    def plan(self, start, goal):   
        h, w = self.grid.shape

        def heuristic(a, b):
            return np.linalg.norm(np.array(a) - np.array(b))

        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}

        neighbors = [
            (1,0), (-1,0), (0,1), (0,-1),
            (1,1), (1,-1), (-1,1), (-1,-1)
        ]

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dx, dy in neighbors:
                nx, ny = current[0] + dx, current[1] + dy

                if not (0 <= nx < h and 0 <= ny < w):
                    continue

                if self.grid[nx, ny] == 1:
                    continue

                move_cost = 1.0 if dx == 0 or dy == 0 else 1.414
                tentative_g = g_score[current] + move_cost

                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = tentative_g
                    f = tentative_g + heuristic((nx, ny), goal)

                    heapq.heappush(open_set, (f, (nx, ny)))
                    came_from[(nx, ny)] = current

        return None