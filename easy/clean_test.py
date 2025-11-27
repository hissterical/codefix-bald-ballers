#!/usr/bin/env python3
"""
Clean verification of cost functions
"""
from wumpus_world import BangaloreWumpusWorld, load_config

def test_cost_function():
    """Test cost function directly"""
    config = load_config()
    world = BangaloreWumpusWorld(config)
    
    # Create a clean test world
    test_world = BangaloreWumpusWorld(config)
    
    print("🧪 Testing Cost Function")
    print("-" * 30)
    
    # Test each cost type by directly setting grid cells
    
    # Test pit
    test_world.grid[1][1]['type'] = 'pit'
    pit_cost = test_world.get_cell_cost(1, 1)
    print(f"Pit cost: {pit_cost} (should be inf)")
    
    # Test traffic light  
    test_world.grid[1][2]['type'] = 'traffic_light'
    traffic_cost = test_world.get_cell_cost(1, 2)
    print(f"Traffic light cost: {traffic_cost} (should be 20)")
    
    # Test cow
    test_world.grid[1][3]['type'] = 'cow'
    cow_cost = test_world.get_cell_cost(1, 3)
    print(f"Cow cost: {cow_cost} (should be 10)")
    
    # Test normal cell
    test_world.grid[1][4]['type'] = 'empty'
    test_world.grid[1][4]['weight'] = 7
    normal_cost = test_world.get_cell_cost(1, 4)
    print(f"Normal cell cost: {normal_cost} (should be 7)")
    
    # Verify all are correct
    results = []
    results.append(("Pit avoidance", pit_cost == float('inf')))
    results.append(("Traffic light cost", traffic_cost == 20))
    results.append(("Cow cost", cow_cost == 10))
    results.append(("Normal cell weight", normal_cost == 7))
    
    print("\n📊 Results:")
    for test_name, passed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {test_name}")
    
    return all(result[1] for result in results)

def test_world_navigation():
    """Test actual world navigation"""
    print("\n🌍 Testing World Navigation")
    print("-" * 30)
    
    config = load_config()
    world = BangaloreWumpusWorld(config)
    
    print(f"Agent: {world.agent_pos} -> Goal: {world.goal_pos}")
    
    # Find path
    path = world.find_path_astar()
    
    if path:
        print(f"✅ Path found: {path}")
        
        # Calculate and verify cost
        total_cost = 0
        for i in range(1, len(path)):
            x, y = path[i]
            step_cost = world.get_cell_cost(x, y)
            total_cost += step_cost
            cell_type = world.grid[y][x]['type']
            print(f"  Step {i}: {path[i]} (type: {cell_type}, cost: {step_cost})")
        
        print(f"Total path cost: {total_cost}")
        return True
    else:
        print(f"❌ No path found: {world.message}")
        return False

def main():
    print("🎯 CLEAN COST FUNCTION VERIFICATION")
    print("=" * 40)
    
    cost_test = test_cost_function()
    nav_test = test_world_navigation()
    
    print(f"\n📋 Final Results:")
    print(f"✅ Cost functions: {'PASS' if cost_test else 'FAIL'}")
    print(f"✅ Navigation: {'PASS' if nav_test else 'FAIL'}")
    
    if cost_test and nav_test:
        print("\n🎉 All systems working correctly!")
    else:
        print("\n⚠️ Issues detected")

if __name__ == "__main__":
    main()