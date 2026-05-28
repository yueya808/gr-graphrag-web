param(
    [string]$PythonExe = "python",
    [string]$SiliconflowModel = "Qwen/Qwen3-8B",
    [string]$Neo4jUri = "neo4j+s://01a0e5bf.databases.neo4j.io",
    [string]$Neo4jDatabase = "01a0e5bf",
    [string]$Neo4jUser = "01a0e5bf",
    [switch]$SkipAggregate
)

$ErrorActionPreference = "Stop"

function Read-Required([string]$PromptText) {
    while ($true) {
        $v = Read-Host $PromptText
        if (-not [string]::IsNullOrWhiteSpace($v)) {
            return $v.Trim()
        }
        Write-Host "输入不能为空，请重试。" -ForegroundColor Yellow
    }
}

Write-Host "==== GraphRAG 对比实验一键运行 ====" -ForegroundColor Cyan

$sfKey = $env:SILICONFLOW_API_KEY
if ([string]::IsNullOrWhiteSpace($sfKey)) {
    $sfKey = Read-Required "请输入 SILICONFLOW_API_KEY"
}

$neo4jPassword = $env:NEO4J_PASSWORD
if ([string]::IsNullOrWhiteSpace($neo4jPassword)) {
    $neo4jPassword = Read-Required "请输入 NEO4J_PASSWORD"
}

$env:SILICONFLOW_API_KEY = $sfKey
$env:SILICONFLOW_MODEL = $SiliconflowModel
$env:NEO4J_URI = $Neo4jUri
$env:NEO4J_DATABASE = $Neo4jDatabase
$env:NEO4J_USER = $Neo4jUser
$env:NEO4J_PASSWORD = $neo4jPassword

Write-Host "`n[1/2] 运行 run_experiment.py ..." -ForegroundColor Green
& $PythonExe "experiments/graphrag_benchmark/run_experiment.py"

if (-not $SkipAggregate) {
    Write-Host "`n[2/2] 运行 aggregate_manual_scores.py ..." -ForegroundColor Green
    & $PythonExe "experiments/graphrag_benchmark/aggregate_manual_scores.py" --input "experiments/graphrag_benchmark/manual_scores_template.csv"
}

Write-Host "`n完成。评分表位置：" -ForegroundColor Cyan
Write-Host "experiments/graphrag_benchmark/manual_scores_template.csv"
