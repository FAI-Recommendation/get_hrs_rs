# ─────────────────────────────────────────────────────────────
# download_results.ps1
# Tai output/ evaluator/ tensorboard/ tu GPU pod ve may local
# Usage: .\download_results.ps1
# ─────────────────────────────────────────────────────────────

# ── Doc .env ──
$envFile = Join-Path $PSScriptRoot ".env"
if (-not (Test-Path $envFile)) {
    Write-Error ".env not found at $envFile"
    exit 1
}

$env_vars = @{}
Get-Content $envFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -match "^([A-Z_]+)=(.*)$") {
        $env_vars[$matches[1]] = $matches[2].Split("#")[0].Trim()
    }
}

$SSH_HOST = $env_vars["SSH_HOST"]
$SSH_PORT = $env_vars["SSH_PORT"]
$SSH_USER = $env_vars["SSH_USER"]

if (-not $SSH_HOST -or -not $SSH_PORT) {
    Write-Error "SSH_HOST / SSH_PORT chua duoc set trong .env"
    exit 1
}

# ── Destination ──
$DEST = Join-Path $PSScriptRoot "report\v1"
New-Item -ItemType Directory -Force -Path $DEST | Out-Null

$REMOTE_DIR = "~/get_hrs_rs/rs/lightgcn_pyg"

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host "📥 Download results from GPU pod"
Write-Host "   Host : $SSH_USER@$SSH_HOST : $SSH_PORT"
Write-Host "   From : $REMOTE_DIR/{output,evaluator,tensorboard}"
Write-Host "   To   : $DEST"
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host ""
Write-Host "🔑 Nhap password khi duoc hoi (1 lan duy nhat)..."
Write-Host ""

# ── 1 lenh SSH: tar tat ca, pipe ve local ──
ssh -p $SSH_PORT "${SSH_USER}@${SSH_HOST}" `
    "tar -czf - -C $REMOTE_DIR output evaluator tensorboard 2>/dev/null" `
    | tar -xzf - -C $DEST

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    Write-Host "✅ Done! Files saved to:"
    Write-Host "   $DEST\output"
    Write-Host "   $DEST\evaluator"
    Write-Host "   $DEST\tensorboard"
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
} else {
    Write-Host "❌ Download failed. Kiem tra lai SSH_HOST / SSH_PORT trong .env"
}
