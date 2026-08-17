$ErrorActionPreference = "Stop"

Write-Host "Kích hoạt môi trường Python (venv)..." -ForegroundColor Cyan
.\venv\Scripts\activate

$models = @("phi3_mini", "llama32_3b", "qwen25_3b", "deepseek_r1_1_5b", "gpt_20b_oss")

foreach ($model in $models) {
    Write-Host "`n========================================================" -ForegroundColor Cyan
    Write-Host "Bắt đầu chạy benchmark cho model: $model" -ForegroundColor Green
    Write-Host "========================================================`n" -ForegroundColor Cyan
    
    python -m src.ghost_agents.approach_evaluation.benchmark_evaluator `
        --approaches $model `
        --seeds 42 43 44 `
        --max-per-benchmark 200 `
        --output report-output/ghost_agents/benchmark_results
}

Write-Host "`nHoàn thành chạy tất cả benchmarks!" -ForegroundColor Green
