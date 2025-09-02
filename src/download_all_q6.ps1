# Stop on errors
$ErrorActionPreference = "Stop"

function Get-ModelFile {
    param(
        [Parameter(Mandatory=$true)][string]  $Repo,
        [Parameter(Mandatory=$true)][string[]]$Filenames  # try these in order until one succeeds
    )
    $owner,$name = $Repo.Split('/',2)
    $snapshotsDir = Join-Path $env:USERPROFILE ".cache\huggingface\hub\models--$owner--$name\snapshots"

    foreach ($fn in $Filenames) {
        Write-Host "=== Downloading $Repo -> $fn ===" -ForegroundColor Cyan
        try {
            hf download $Repo $fn | Out-Host

            if (-not (Test-Path $snapshotsDir)) {
                Write-Warning "Snapshots folder not found for $Repo (yet)."
                continue
            }

            # Find newest snapshot containing the file
            $match = Get-ChildItem $snapshotsDir -Directory |
                     Sort-Object LastWriteTime -Descending |
                     ForEach-Object {
                        $candidate = Join-Path $_.FullName $fn
                        if (Test-Path $candidate) { $candidate }
                     } | Select-Object -First 1

            if ($match) {
                Write-Host "Downloaded: $match" -ForegroundColor Green
                Write-Host ""
                return
            } else {
                Write-Warning "Downloaded but could not locate in snapshots (unexpected): $fn"
            }
        }
        catch {
            $msg = $_.Exception.Message.Split("`n")[0]
            Write-Warning "Failed: $fn  ($msg)"
        }
        Write-Host ""
    }

    Write-Warning "No matching filename worked for $Repo. You may need to check the repo's Files list."
}

# === MoE Models ===
# InfinityKuno 2×7B — Q6_K (note: current GGUF may be incompatible with llama.cpp MoE; kept here for completeness)
Get-ModelFile -Repo "backyardai/InfinityKuno-2x7B-GGUF" `
              -Filenames @("InfinityKuno-2x7B.Q6_K.gguf")

# MonarchCoder 2×7B — Q6_K (some exports may be incompatible; verify with test-gguf if needed)
Get-ModelFile -Repo "mradermacher/MonarchCoder-MoE-2x7B-GGUF" `
              -Filenames @("MonarchCoder-MoE-2x7B.Q6_K.gguf")

# DavidAU Qwen2.5-MoE-2×7B (DeepSeek Abliterated) — Q6_k  (lower-case 'k' is common; try both just in case)
Get-ModelFile -Repo "DavidAU/Qwen2.5-MOE-2X7B-DeepSeek-Abliterated-UNCensored-19B-gguf" `
              -Filenames @(
                "Qwen2.5-MOE-2X7B-DeepSeek-Abliterated-Censored-19B-D_AU-Q6_k.gguf",
                "Qwen2.5-MOE-2X7B-DeepSeek-Abliterated-Censored-19B-D_AU-Q6_K.gguf"
              )

# DavidAU Power-CODER 2×7B — Q6_K (try common versioned filenames)
Get-ModelFile -Repo "DavidAU/Qwen2.5-MOE-2x-4x-6x-8x__7B__Power-CODER__19B-30B-42B-53B-gguf" `
              -Filenames @(
                "Qwen2.5-2X7B-Power-Coder-V4-Q6_K.gguf",
                "Qwen2.5-2X7B-Power-Coder-V3-Q6_K.gguf",
                "Qwen2.5-2X7B-Power-Coder-V2-Q6_K.gguf",
                "Qwen2.5-2X7B-Power-Coder-V1-Q6_K.gguf"
              )

# State-of-the-MoE_RP 2×7B — Q6_K
Get-ModelFile -Repo "mradermacher/State-of-the-MoE_RP-2x7B-GGUF" `
              -Filenames @("State-of-the-MoE_RP-2x7B.Q6_K.gguf")

# === Qwen2.5 Instruct ===
Get-ModelFile -Repo "QuantFactory/Qwen2.5-7B-Instruct-GGUF" `
              -Filenames @("Qwen2.5-7B-Instruct.Q6_K.gguf")

Get-ModelFile -Repo "bartowski/Qwen2.5-14B-Instruct-GGUF" `
              -Filenames @("Qwen2.5-14B-Instruct-Q6_K.gguf")

# === DeepSeek-R1-Distill-Qwen ===
Get-ModelFile -Repo "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF" `
              -Filenames @("DeepSeek-R1-Distill-Qwen-7B-Q6_K.gguf")

Get-ModelFile -Repo "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF" `
              -Filenames @("DeepSeek-R1-Distill-Qwen-14B-Q6_K.gguf")

Write-Host "=== All Q6 downloads attempted. Look for 'Downloaded:' lines above with final paths. ===" -ForegroundColor Yellow
