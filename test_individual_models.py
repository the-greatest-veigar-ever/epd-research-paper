#!/usr/bin/env python3
"""
Test Individual Models
Helps debug which models are working and which are failing
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ghost_agents.agent import GhostAgentFactory

MODELS = ["llama3.2:3b", "phi", "gemma2:2b"]

test_plan = {
    "action": "BLOCK_IP",
    "target": "192.168.1.100"
}

print("="*70)
print("TESTING INDIVIDUAL MODELS")
print("="*70)

for model in MODELS:
    print(f"\n{'='*70}")
    print(f"Testing: {model}")
    print(f"{'='*70}")
    
    try:
        # Create agent with specific model
        agent = GhostAgentFactory.create_agent(
            "Block malicious IP address",
            rotate_model=False
        )
        agent.model = model  # Force specific model
        
        # Execute
        result = agent.execute_remediation(test_plan)
        
        # Print results
        print(f"\nStatus: {result.get('status')}")
        print(f"Tool Used: {result.get('tool_used', 'N/A')}")
        print(f"Command: {result.get('command', 'N/A')[:100]}")
        
        if result.get('error'):
            print(f"Error: {result['error']}")
        
        if result.get('status') in ['success', 'simulated_success']:
            print("✅ SUCCESS")
        else:
            print("❌ FAILED")
            
    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")

print(f"\n{'='*70}")
print("TEST COMPLETE")
print(f"{'='*70}")
