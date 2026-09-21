import numpy as np
import heapq

class Node:
    """
    Đại diện cho một ô (node) trên bản đồ lưới (Grid Map).
    """
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0 # G: Chi phí từ điểm bắt đầu đến node hiện tại
        self.h = 0 # H: Chi phí ước lượng (Heuristic) từ node hiện tại đến đích
        self.f = 0 # F: Tổng chi phí (F = G + H)

    def __eq__(self, other):
        return self.position == other.position

    # Dùng cho hàng đợi ưu tiên (Priority Queue)
    def __lt__(self, other):
        return self.f < other.f

class AStarPlanner:
    def __init__(self, grid_map):
        """
        Khởi tạo bộ tìm đường với bản đồ 2D.
        - grid_map: Mảng numpy 2D (0: đường trống, 1: vật cản).
        """
        self.grid_map = grid_map
        self.max_x = len(grid_map)
        self.max_y = len(grid_map[0])
        
        # 8 hướng di chuyển (Lên, Xuống, Trái, Phải, và 4 đường chéo)
        self.movements = [
            (0, -1), (0, 1), (-1, 0), (1, 0),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]

    def _heuristic(self, current_pos, goal_pos):
        """
        Tính khoảng cách Euclid (đường chim bay) làm Heuristic.
        """
        return np.linalg.norm(np.array(current_pos) - np.array(goal_pos))

    def _is_valid_position(self, pos):
        """
        Kiểm tra xem tọa độ có nằm trong bản đồ và không phải vật cản không.
        """
        x, y = pos
        if 0 <= x < self.max_x and 0 <= y < self.max_y:
            if self.grid_map[x][y] == 0: # 0 là đường trống
                return True
        return False

    def plan(self, start, goal):
        """
        Tìm đường đi ngắn nhất từ start đến goal bằng A*.
        - start, goal: Tuple tọa độ (x, y).
        - Trả về: Danh sách các tọa độ [(x1,y1), (x2,y2), ...] hoặc rỗng nếu không tìm thấy.
        """
        # 1. Khởi tạo
        start_node = Node(None, start)
        goal_node = Node(None, goal)

        open_list = [] # Danh sách các node cần kiểm tra (ưu tiên chi phí F thấp nhất)
        closed_set = set() # Tập hợp các node đã kiểm tra (tránh vòng lặp vô hạn)

        # Thêm điểm xuất phát vào open_list
        heapq.heappush(open_list, start_node)

        # 2. Vòng lặp tìm kiếm chính
        while open_list:
            # Lấy node có chi phí F thấp nhất ra để xét
            current_node = heapq.heappop(open_list)
            closed_set.add(current_node.position)

            # 3. Đã tới đích chưa?
            if current_node == goal_node:
                path = []
                current = current_node
                # Truy vết (Backtrack) từ đích về điểm xuất phát
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                return path[::-1] # Đảo ngược danh sách để có đường đi từ Start -> Goal

            # 4. Tạo các node lân cận (Neighbors)
            for move in self.movements:
                node_position = (current_node.position[0] + move[0], 
                                 current_node.position[1] + move[1])

                # Bỏ qua nếu vị trí không hợp lệ (ngoài bản đồ hoặc là vật cản)
                if not self._is_valid_position(node_position):
                    continue

                # Bỏ qua nếu node đã được kiểm tra xong
                if node_position in closed_set:
                    continue

                # Tính toán chi phí G, H, F cho node lân cận
                neighbor = Node(current_node, node_position)
                
                # Chi phí di chuyển thẳng = 1, di chuyển chéo = sqrt(2) ≈ 1.414
                cost = 1 if move[0] == 0 or move[1] == 0 else 1.414
                neighbor.g = current_node.g + cost
                neighbor.h = self._heuristic(neighbor.position, goal_node.position)
                neighbor.f = neighbor.g + neighbor.h

                # Kiểm tra xem neighbor đã có trong open_list với chi phí G thấp hơn chưa
                # (Nếu có đường đi tốt hơn tới neighbor này rồi thì bỏ qua)
                in_open_list = False
                for open_node in open_list:
                    if neighbor == open_node and neighbor.g >= open_node.g:
                        in_open_list = True
                        break
                
                if not in_open_list:
                    heapq.heappush(open_list, neighbor)

        # Trả về rỗng nếu không tìm thấy đường đi
        print("A* Warning: Không tìm thấy đường đi tới đích!")
        return []

# Ví dụ cách sử dụng nhanh:
if __name__ == '__main__':
    # 0 = Đường trống, 1 = Vật cản
    mock_grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    
    planner = AStarPlanner(mock_grid)
    path = planner.plan(start=(0, 0), goal=(4, 4))
    print("Đường đi:", path)