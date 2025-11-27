#!/usr/bin/env python3
"""
Final verification script for A* implementation
Checks all submission criteria
"""
import os
import sys
from wumpus_world import BangaloreWumpusWorld, load_config

def check_implementation():
    """Verify all implementation requirements"""
    
    print("🔍 CHECKING A* IMPLEMENTATION REQUIREMENTS")
    print("=" * 50)
    
    checks = []
    
    # 1. Check if find_path_astar() is implemented
    config = load_config()
    world = BangaloreWumpusWorld(config)
    
    # Test basic functionality
    try:
        path = world.find_path_astar()
        if path is not None:
            checks.append("✅ find_path_astar() implemented and working")
        else:
            checks.append("❌ find_path_astar() returns None")
    except Exception as e:
        checks.append(f"❌ find_path_astar() error: {str(e)}")
    
    # 2. Check Manhattan distance heuristic
    try:
        h1 = world.heuristic((0, 0), (3, 4))
        expected = abs(0-3) + abs(0-4)
        if h1 == expected:
            checks.append("✅ Manhattan distance heuristic correct")
        else:
            checks.append(f"❌ Heuristic wrong: got {h1}, expected {expected}")
    except:
        checks.append("❌ Heuristic function missing or broken")
    
    # 3. Check movement restrictions (no diagonals)
    try:
        neighbors = world._get_neighbors(1, 1)
        expected_neighbors = [(1, 0), (1, 2), (0, 1), (2, 1)]
        if len(neighbors) == 4 and all(n in expected_neighbors for n in neighbors):
            checks.append("✅ Only orthogonal movement (no diagonals)")
        else:
            checks.append(f"❌ Wrong neighbors: {neighbors}")
    except:
        checks.append("❌ _get_neighbors() function issue")
    
    # 4. Check obstacle costs
    try:
        # Test different cell types
        costs_correct = True
        
        # Set up test cells
        world.grid[0][0]['type'] = 'pit'
        world.grid[0][1]['type'] = 'traffic_light' 
        world.grid[0][2]['type'] = 'cow'
        world.grid[0][3]['type'] = 'empty'
        world.grid[0][3]['weight'] = 5
        
        pit_cost = world.get_cell_cost(0, 0)
        traffic_cost = world.get_cell_cost(0, 1)
        cow_cost = world.get_cell_cost(0, 2)
        normal_cost = world.get_cell_cost(0, 3)
        
        if pit_cost == float('inf'):
            checks.append("✅ Pit cost = infinity (avoided)")
        else:
            checks.append(f"❌ Pit cost wrong: {pit_cost}")
            
        if traffic_cost == 20:
            checks.append("✅ Traffic light cost = 20")
        else:
            checks.append(f"❌ Traffic light cost wrong: {traffic_cost}")
            
        if cow_cost == 10:
            checks.append("✅ Cow cost = 10")
        else:
            checks.append(f"❌ Cow cost wrong: {cow_cost}")
            
        if normal_cost == 5:
            checks.append("✅ Normal cell uses weight")
        else:
            checks.append(f"❌ Normal cell cost wrong: {normal_cost}")
            
    except Exception as e:
        checks.append(f"❌ Cost function error: {str(e)}")
    
    # 5. Check path format
    try:
        path = world.find_path_astar()
        if path:
            if isinstance(path, list) and all(isinstance(p, tuple) and len(p) == 2 for p in path):
                checks.append("✅ Path format correct: list of (x,y) tuples")
            else:
                checks.append(f"❌ Wrong path format: {path}")
        else:
            # Test with unreachable goal
            # Temporarily block all paths
            for y in range(5):
                for x in range(1, 10):
                    if (x, y) != world.goal_pos:
                        world.grid[y][x]['type'] = 'pit'
            
            blocked_path = world.find_path_astar()
            if blocked_path is None and "Path Not Found" in world.message:
                checks.append("✅ Returns None and sets message when no path")
            else:
                checks.append("❌ Doesn't handle unreachable goal correctly")
                
    except Exception as e:
        checks.append(f"❌ Path format check error: {str(e)}")
    
    # 6. Check game integration
    try:
        # Reset world for integration test
        world = BangaloreWumpusWorld(config)
        original_pos = list(world.agent_pos)
        path = world.find_path_astar()
        
        if path and len(path) >= 2:
            # Test that path starts at agent position
            if path[0] == tuple(original_pos):
                checks.append("✅ Path starts at agent position")
            else:
                checks.append(f"❌ Path doesn't start at agent: {path[0]} vs {original_pos}")
                
            # Test that path ends at goal
            if path[-1] == world.goal_pos:
                checks.append("✅ Path ends at goal position")
            else:
                checks.append(f"❌ Path doesn't end at goal: {path[-1]} vs {world.goal_pos}")
        
    except Exception as e:
        checks.append(f"❌ Game integration error: {str(e)}")
    
    return checks

def main():
    print("🎯 AI CODEFIX 2025 - A* IMPLEMENTATION VERIFICATION")
    print("=" * 60)
    
    checks = check_implementation()
    
    print("\n📋 VERIFICATION RESULTS:")
    print("-" * 40)
    for check in checks:
        print(f"  {check}")
    
    # Count successes
    success_count = len([c for c in checks if c.startswith("✅")])
    total_count = len(checks)
    
    print(f"\n📊 SUMMARY:")
    print(f"✅ Passed: {success_count}/{total_count} checks")
    print(f"📈 Success Rate: {success_count/total_count*100:.1f}%")
    
    if success_count == total_count:
        print("\n🎉 CONGRATULATIONS! All requirements met!")
        print("🚀 Ready for submission!")
    else:
        print(f"\n⚠️  Need to fix {total_count - success_count} issues")
        
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()