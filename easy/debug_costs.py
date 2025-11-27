#!/usr/bin/env python3
"""
Debug the actual cell types in the generated world
"""
from wumpus_world import BangaloreWumpusWorld, load_config

def debug_world_costs():
    """Debug what's actually in the world"""
    config = load_config()
    world = BangaloreWumpusWorld(config)
    
    print("🔍 DEBUGGING WORLD GENERATION")
    print("=" * 40)
    
    # Find examples of each cell type in the actual world
    pit_examples = []
    traffic_examples = []
    cow_examples = []
    normal_examples = []
    
    for y in range(5):
        for x in range(10):
            cell = world.grid[y][x]
            cell_type = cell['type']
            
            if cell_type == 'pit':
                pit_examples.append((x, y))
            elif cell_type == 'traffic_light':
                traffic_examples.append((x, y))
            elif cell_type == 'cow':
                cow_examples.append((x, y))
            elif cell_type == 'empty':
                normal_examples.append((x, y))
    
    print(f"Found pits: {pit_examples}")
    print(f"Found traffic lights: {traffic_examples}")
    print(f"Found cows: {cow_examples}")
    print(f"Found normal cells: {normal_examples[:3]}...")  # Just show first 3
    
    # Test cost function on actual cells
    print("\n🧮 Testing costs on actual cells:")
    
    if pit_examples:
        x, y = pit_examples[0]
        cost = world.get_cell_cost(x, y)
        print(f"Pit at {(x, y)}: cost = {cost}")
        
    if traffic_examples:
        x, y = traffic_examples[0]
        cost = world.get_cell_cost(x, y)
        print(f"Traffic light at {(x, y)}: cost = {cost}")
        
    if cow_examples:
        x, y = cow_examples[0]
        cost = world.get_cell_cost(x, y)
        print(f"Cow at {(x, y)}: cost = {cost}")
        
    if normal_examples:
        x, y = normal_examples[0]
        cost = world.get_cell_cost(x, y)
        weight = world.grid[y][x]['weight']
        print(f"Normal cell at {(x, y)}: cost = {cost}, weight = {weight}")
    
    # Test by directly creating cells
    print("\n🧪 Testing direct cell creation:")
    
    # Create a fresh cell and test
    test_cell_pit = {'type': 'pit', 'percepts': [], 'weight': 5}
    world.grid[2][2] = test_cell_pit
    pit_cost = world.get_cell_cost(2, 2)
    print(f"Direct pit cell: cost = {pit_cost}")
    
    test_cell_traffic = {'type': 'traffic_light', 'percepts': [], 'weight': 5}
    world.grid[2][3] = test_cell_traffic
    traffic_cost = world.get_cell_cost(2, 3)
    print(f"Direct traffic cell: cost = {traffic_cost}")
    
    test_cell_cow = {'type': 'cow', 'percepts': [], 'weight': 5}
    world.grid[2][4] = test_cell_cow
    cow_cost = world.get_cell_cost(2, 4)
    print(f"Direct cow cell: cost = {cow_cost}")

def main():
    debug_world_costs()

if __name__ == "__main__":
    main()