#!/usr/bin/env python3
"""
Test with different configurations to verify robustness
"""
from wumpus_world import BangaloreWumpusWorld
import json

def test_configuration(seed, description):
    print(f"\n=== Testing {description} (seed={seed}) ===")
    
    # Create custom config
    config = {
        "team_id": "test_team",
        "seed": seed,
        "grid_config": {
            "traffic_lights": 4,
            "cows": 3,
            "pits": 6
        }
    }
    
    world = BangaloreWumpusWorld(config)
    
    print(f"Agent Start: {world.agent_start}")
    print(f"Goal Position: {world.goal_pos}")
    
    # Print grid for visualization
    print("\nGrid layout:")
    for y in range(5):
        row = []
        for x in range(10):
            if (x, y) == tuple(world.agent_pos):
                row.append("A")
            elif (x, y) == world.goal_pos:
                row.append("G")
            elif world.grid[y][x]['type'] == 'pit':
                row.append("P")
            elif world.grid[y][x]['type'] == 'cow':
                row.append("C")
            elif world.grid[y][x]['type'] == 'traffic_light':
                row.append("T")
            else:
                row.append(".")
        print(" ".join(row))
    
    # Test A* pathfinding
    print("\nExecuting A* pathfinding...")
    path = world.find_path_astar()
    
    if path:
        print(f"✅ SUCCESS! Path length: {len(path)}")
        print(f"Path: {path}")
        
        # Calculate total cost
        total_cost = 0
        for i in range(1, len(path)):
            x, y = path[i]
            cost = world.get_cell_cost(x, y)
            total_cost += cost
        print(f"Total path cost: {total_cost}")
        
        return True
    else:
        print("❌ No path found")
        print(f"Message: {world.message}")
        return False

def main():
    print("=== A* Algorithm Robustness Test ===")
    
    # Test multiple scenarios
    test_cases = [
        (9, "Original Configuration"),
        (42, "Alternative Layout 1"),
        (123, "Alternative Layout 2"),
        (999, "Challenge Layout"),
    ]
    
    success_count = 0
    for seed, description in test_cases:
        success = test_configuration(seed, description)
        if success:
            success_count += 1
    
    print(f"\n=== Final Results ===")
    print(f"✅ Successful tests: {success_count}/{len(test_cases)}")
    print(f"Success rate: {success_count/len(test_cases)*100:.1f}%")

if __name__ == "__main__":
    main()