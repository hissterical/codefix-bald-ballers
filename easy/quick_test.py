#!/usr/bin/env python3
"""
Quick test to verify A* in the actual game environment
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from wumpus_world import BangaloreWumpusWorld, load_config

def main():
    config = load_config()
    world = BangaloreWumpusWorld(config)
    
    print("=== Quick A* Test ===")
    print(f"Agent Start: {world.agent_start}")
    print(f"Goal Position: {world.goal_pos}")
    
    # Test A* pathfinding
    print("\nExecuting A* pathfinding...")
    path = world.find_path_astar()
    
    if path:
        print(f"✅ SUCCESS! Path: {path}")
        print("Testing path execution...")
        
        # Execute path step by step and show results
        original_pos = list(world.agent_pos)
        for i, (x, y) in enumerate(path):
            if i == 0:
                continue  # Skip start position
            
            print(f"Step {i}: Moving to ({x}, {y})")
            world.move_agent(x, y)
            print(f"  Agent now at: {world.agent_pos}")
            print(f"  Message: {world.message}")
            
            if world.game_won:
                print("🎉 GOAL REACHED!")
                break
            elif world.game_over:
                print("💀 GAME OVER!")
                break
    else:
        print("❌ FAILED! No path found")
        print(f"Message: {world.message}")

if __name__ == "__main__":
    main()