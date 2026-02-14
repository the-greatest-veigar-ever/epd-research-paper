#!/usr/bin/env python3
"""Quick test to verify command parsing fixes"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ghost_agents.agent import GhostAgentFactory

# Test with a few scenarios
test_scenarios = [
    {"action": "BLOCK_IP", "target": "192.168.1.100"},
    {"action": "TERMINATE_INSTANCE", "target": "i-abc123"},
    {"action": "REVOKE_SESSIONS", "target": "user-admin-123"}
]

print("=" * 60)
print("TESTING COMMAND PARSING FIXES")
print("=" * 60)

correct = 0
total = len(test_scenarios)

for i, scenario in enumerate(test_scenarios, 1):
    print(f"\n[Test {i}/{total}]")
    print(f"  Action: {scenario['action']}")
    print(f"  Target: {scenario['target']}")
    
    try:
        # Create agent
        agent = GhostAgentFactory.create_agent(
            f"Perform {scenario['action']} on {scenario['target']}",
            rotate_model=False  # Use first model only for consistency
        )
        
        # Execute
        result = agent.execute_remediation(scenario)
        
        # Check result
        tool_used = result.get("tool_used", "unknown")
        status = result.get("status", "failed")
        
        print(f"  Status: {status}")
        print(f"  Tool: {tool_used}")
        
        if tool_used == "aws":
            print(f"  ✅ CORRECT")
            correct += 1
        else:
            print(f"  ❌ INCORRECT (expected 'aws', got '{tool_used}')")
            
    except Exception as e:
        print(f"  ❌ ERROR: {e}")

print("\n" + "=" * 60)
print(f"RESULTS: {correct}/{total} correct ({correct/total*100:.1f}%)")
print("=" * 60)
