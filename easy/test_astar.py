#!/usr/bin/env python3
"""
Test script to verify A* implementation without pygame
"""
import json
import random
import sys

# Load team configuration
def load_config():
    try:
        with open('team_config.json', 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print("Error: team_config.json not found!")
        sys.exit(1)

# Simplified world class for testing
class TestWorld:
    def __init__(self, config):
        self.config = config
        self.seed = config['seed']
        random.seed(self.seed)
        
        # Grid constants
        self.GRID_ROWS = 5
        self.GRID_COLS = 10
        
        # Initialize grid
        self.grid = [[{'type': 'empty', 'weight': random.randint(1,15)}
              for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        
        # Agent starts at bottom-left diagonal
        self.agent_start = (0, self.GRID_ROWS - 1)
        self.agent_pos = list(self.agent_start)
        
        # Generate world
        self._generate_world()
        
    def _generate_world(self):
        """Generate random world elements based on config"""
        num_traffic_lights = self.config['grid_config']['traffic_lights']
        num_cows = self.config['grid_config']['cows']
        num_pits = self.config['grid_config']['pits']

        # Generate available positions (exclude agent start)
        available_positions = [(x, y) for x in range(self.GRID_COLS) for y in range(self.GRID_ROWS)
                               if (x, y) != tuple(self.agent_start)]

        random.shuffle(available_positions)

        # Place traffic lights
        for i in range(num_traffic_lights):
            if available_positions:
                pos = available_positions.pop()
                self.grid[pos[1]][pos[0]]['type'] = 'traffic_light'

        # Place cows
        for i in range(num_cows):
            if available_positions:
                pos = available_positions.pop()
                self.grid[pos[1]][pos[0]]['type'] = 'cow'

        # Place pits
        for i in range(num_pits):
            if available_positions:
                pos = available_positions.pop()
                self.grid[pos[1]][pos[0]]['type'] = 'pit'

        # Place goal
        if available_positions:
            goal_pos = available_positions.pop()
            self.grid[goal_pos[1]][goal_pos[0]]['type'] = 'goal'
            self.goal_pos = goal_pos
            
    def _get_neighbors(self, x, y):
        """Get valid adjacent neighbors (no diagonals)"""
        neighbors = []
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS:
                neighbors.append((nx, ny))

        return neighbors
    
    def get_cell_cost(self, x, y):
        """Get the cost of moving to a specific cell"""
        cell_type = self.grid[y][x]['type']
        
        if cell_type == 'pit':
            return float('inf')  # Avoid pits completely
        elif cell_type == 'traffic_light':
            return 20  # Higher cost for traffic lights
        elif cell_type == 'cow':
            return 10  # Cost for cows
        else:
            return self.grid[y][x]['weight']  # Random cost for normal cells

    def heuristic(self, pos, goal):
        """Manhattan distance heuristic"""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def find_path_astar(self):
        """A* pathfinding algorithm implementation"""
        import heapq
        
        start = tuple(self.agent_pos)
        goal = self.goal_pos
        
        print(f"Starting A* from {start} to {goal}")
        
        # Priority queue: (f_score, g_score, position)
        open_set = []
        heapq.heappush(open_set, (0, 0, start))
        
        # Track visited nodes
        closed_set = set()
        
        # Track scores
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        # Track path
        came_from = {}
        
        while open_set:
            # Get node with lowest f_score
            current_f, current_g, current = heapq.heappop(open_set)
            
            # Skip if already processed
            if current in closed_set:
                continue
                
            # Add to closed set
            closed_set.add(current)
            
            print(f"Processing: {current}, g={current_g}, h={self.heuristic(current, goal)}, f={current_f}")
            
            # Check if we reached the goal
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                
                print(f"Path found: {path}")
                return path
            
            # Explore neighbors
            neighbors = self._get_neighbors(current[0], current[1])
            
            for neighbor_x, neighbor_y in neighbors:
                neighbor = (neighbor_x, neighbor_y)
                
                # Skip if already processed
                if neighbor in closed_set:
                    continue
                
                # Calculate cost to move to this neighbor
                move_cost = self.get_cell_cost(neighbor_x, neighbor_y)
                
                # Skip if pit (infinite cost)
                if move_cost == float('inf'):
                    continue
                
                # Calculate tentative g_score
                tentative_g = g_score[current] + move_cost
                
                # If we haven't seen this neighbor or found a better path
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # Update scores and path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h_score = self.heuristic(neighbor, goal)
                    f_score[neighbor] = tentative_g + h_score
                    
                    # Add to open set
                    heapq.heappush(open_set, (f_score[neighbor], tentative_g, neighbor))
                    
                    print(f"  Neighbor {neighbor}: g={tentative_g}, h={h_score}, f={f_score[neighbor]}")
        
        # No path found
        print("No path found!")
        return None
    
    def print_grid(self):
        """Print grid for debugging"""
        print("\nGrid layout:")
        for y in range(self.GRID_ROWS):
            row = []
            for x in range(self.GRID_COLS):
                if (x, y) == tuple(self.agent_pos):
                    row.append("A")
                elif (x, y) == self.goal_pos:
                    row.append("G")
                elif self.grid[y][x]['type'] == 'pit':
                    row.append("P")
                elif self.grid[y][x]['type'] == 'cow':
                    row.append("C")
                elif self.grid[y][x]['type'] == 'traffic_light':
                    row.append("T")
                else:
                    row.append(".")
            print(" ".join(row))
        print()

def main():
    config = load_config()
    world = TestWorld(config)
    
    print("=== Testing A* Implementation ===")
    print(f"Team ID: {config['team_id']}")
    print(f"Agent Start: {world.agent_start}")
    print(f"Goal Position: {world.goal_pos}")
    
    world.print_grid()
    
    # Test A* algorithm
    path = world.find_path_astar()
    
    if path:
        print(f"\n✅ SUCCESS! Path found: {path}")
        print(f"Path length: {len(path)} steps")
        
        # Calculate total cost
        total_cost = 0
        for i in range(1, len(path)):
            x, y = path[i]
            cost = world.get_cell_cost(x, y)
            total_cost += cost
            print(f"Step {i}: {path[i]} -> cost: {cost}")
        print(f"Total path cost: {total_cost}")
    else:
        print("❌ FAILED! No path found")

if __name__ == "__main__":
    main()