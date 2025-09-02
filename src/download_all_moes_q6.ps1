# Stop on errors
$ErrorActionPreference = "Stop"

function Download-Model {
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

# 1) InfinityKuno MoE 2×7B — Q6_K
Download-Model -Repo "backyardai/InfinityKuno-2x7B-GGUF" `
               -Filename "InfinityKuno-2x7B.Q6_K.gguf"

# 2) MonarchCoder MoE 2×7B — Q6_K
Download-Model -Repo "mradermacher/MonarchCoder-MoE-2x7B-GGUF" `
               -Filename "MonarchCoder-MoE-2x7B.Q6_K.gguf"

# 3) DavidAU Qwen2.5-MoE-2×7B DeepSeek 19B — Q6_K
# (File name says "Censored" — that’s normal for this export)
Download-Model -Repo "DavidAU/Qwen2.5-MOE-2X7B-DeepSeek-Abliterated-UNCensored-19B-gguf" `
               -Filename "Qwen2.5-MOE-2X7B-DeepSeek-Abliterated-Censored-19B-D_AU-Q6_K.gguf"

Write-Host "=== All Q6 MoE downloads attempted. Look for 'Downloaded:' lines with final paths. ===" -ForegroundColor Yellow
