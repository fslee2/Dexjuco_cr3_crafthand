param(
    [string]$RepoName = "cr3-craft-teleop-showcase",
    [string]$Visibility = "public",
    [string]$Description = "CR3 + CRAFT hand integration and teleoperation experiments in DexJoCo/MuJoCo."
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI 'gh' is not installed or not on PATH. Install from https://cli.github.com/ and run 'gh auth login'."
}

gh auth status

if (-not (git remote get-url origin 2>$null)) {
    gh repo create $RepoName "--$Visibility" --description $Description --source . --remote origin --push
} else {
    git push -u origin main
}

Write-Host ""
Write-Host "[done] Published repository:"
gh repo view --web
