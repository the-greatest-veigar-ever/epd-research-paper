#!/bin/bash
# Quick Start Guide for Ghost Agents Evaluation

echo "============================================================"
echo "GHOST AGENTS EVALUATION - QUICK START"
echo "============================================================"

# Check Ollama
echo ""
echo "1. Checking Ollama..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "   ✅ Ollama is running"
else
    echo "   ❌ Ollama is NOT running"
    echo "   Please start with: ollama serve"
    exit 1
fi

# Check data files
echo ""
echo "2. Checking data files..."
if [ -f "data/watchers/cse-cic-ids2018/Processed Traffic Data for ML Algorithms/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv" ]; then
    echo "   ✅ Traffic data found"
else
    echo "   ❌ Traffic data missing"
fi

if [ -f "ai/data/ghost_agents/combined_scenarios.jsonl" ]; then
    echo "   ✅ Scenarios data found"
else
    echo "   ❌ Scenarios data missing"
fi

echo ""
echo "============================================================"
echo "EVALUATION OPTIONS"
echo "============================================================"
echo ""
echo "Option 1: Quick Test (3 scenarios, ~30 seconds)"
echo "  python3 test_parsing_fix.py"
echo ""
echo "Option 2: Baseline Evaluation (~10-15 min)"
echo "  python3 src/ghost_agents/baseline_evaluation.py"
echo ""
echo "Option 3: Proposed Evaluation (~10-15 min)"
echo "  python3 src/ghost_agents/evaluate.py"
echo ""
echo "Option 4: Injection Resistance (~5 min)"
echo "  python3 src/ghost_agents/injection_resistance_evaluation.py"
echo ""
echo "Option 5: Full Pipeline (~30-60 min)"
echo "  python3 src/ghost_agents/full_pipeline_evaluation.py --mode lightweight"
echo ""
echo "Option 6: Complete Suite (ALL evaluations)"
echo "  python3 run_full_evaluation.py"
echo ""
echo "============================================================"
