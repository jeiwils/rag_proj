# Stop on errors
$ErrorActionPreference = "Stop"

function Get-ModelFile {
    param(
        [Parameter(Mandatory=$true)][string]$Repo,
        [Parameter(Mandatory=$true)][string]$Filename
    )
    Write-Host "=== Downloading $Repo -> $Filename ===" -ForegroundColor Cyan
    hf download $Repo $Filename | Out-Host

    # Resolve path in HF cache
    $owner,$name = $Repo.Split('/',2)
    $snapshotsDir = Join-Path $env:USERPROFILE ".cache\huggingface\hub\models--$owner--$name\snapshots"
    if (-not (Test-Path $snapshotsDir)) {
        Write-Warning "Snapshots folder not found for $Repo."
        return
    }

    $match = Get-ChildItem $snapshotsDir -Directory |
             Sort-Object LastWriteTime -Descending |
             ForEach-Object {
                $candidate = Join-Path $_.FullName $Filename
                if (Test-Path $candidate) { $candidate }
             } | Select-Object -First 1

    if ($match) {
        Write-Host "Downloaded: $match" -ForegroundColor Green
    } else {
        Write-Warning "Could not locate $Filename in snapshots for $Repo (download may have failed)."
    }
    Write-Host ""
}

# === MoE Models ===
Get-ModelFile -Repo "backyardai/InfinityKuno-2x7B-GGUF" `
              -Filename "InfinityKuno-2x7B.Q6_K.gguf"

Get-ModelFile -Repo "mradermacher/MonarchCoder-MoE-2x7B-GGUF" `
              -Filename "MonarchCoder-MoE-2x7B.Q6_K.gguf"

Get-ModelFile -Repo "DavidAU/Qwen2.5-MOE-2X7B-DeepSeek-Abliterated-UNCensored-19B-gguf" `
              -Filename "Qwen2.5-MOE-2X7B-DeepSeek-Abliterated-Censored-19B-D_AU-Q6_K.gguf"

# === Qwen2.5 Instruct ===
Get-ModelFile -Repo "QuantFactory/Qwen2.5-7B-Instruct-GGUF" `
              -Filename "Qwen2.5-7B-Instruct.Q6_K.gguf"

Get-ModelFile -Repo "bartowski/Qwen2.5-14B-Instruct-GGUF" `
              -Filename "Qwen2.5-14B-Instruct-Q6_K.gguf"

# === DeepSeek-R1-Distill-Qwen ===
Get-ModelFile -Repo "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF" `
              -Filename "DeepSeek-R1-Distill-Qwen-7B-Q6_K.gguf"

Get-ModelFile -Repo "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF" `
              -Filename "DeepSeek-R1-Distill-Qwen-14B-Q6_K.gguf"

Write-Host "=== All Q6 downloads attempted. Look for 'Downloaded:' lines above with final paths. ===" -ForegroundColor Yellow
