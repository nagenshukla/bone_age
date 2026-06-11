# Overnight ensemble driver: trains additional members (different seeds) on top
# of the already-trained seed-42 base model, then evaluates the full ensemble.
#
# Resumable: a member with final_model_seed{N}.pth is treated as complete and
# skipped, so a re-launch after an interruption only does the unfinished work.
#
# Uses Start-Process with OS-level stdout/stderr redirection. This avoids the
# PowerShell 5.1 trap where a native program's stderr (e.g. tqdm progress) is
# wrapped as a terminating error under $ErrorActionPreference='Stop'.
$ErrorActionPreference = "Stop"
$py = ".venv\Scripts\python.exe"
$seeds = 43, 44, 45   # additional members; seed 42 is the existing best_model.pth

function Invoke-Py($argList, $outFile, $errFile) {
    $p = Start-Process -FilePath $py -ArgumentList $argList -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $outFile -RedirectStandardError $errFile
    return $p.ExitCode
}

Write-Output "=== Preserving seed-42 base model as ensemble member ($(Get-Date -Format HH:mm:ss)) ==="
Copy-Item "checkpoints\best_model.pth" "checkpoints\best_model_seed42.pth" -Force

foreach ($s in $seeds) {
    if (Test-Path "checkpoints\final_model_seed$s.pth") {
        Write-Output "=== Member seed $s already complete, skipping ==="
        continue
    }
    Write-Output "=== Training ensemble member seed $s ($(Get-Date -Format HH:mm:ss)) ==="
    $code = Invoke-Py @('-u','train.py','--seed',"$s",'--tag',"_seed$s") "train_seed$s.log" "train_seed$s.err"
    if ($code -ne 0) { Write-Output "member seed $s FAILED (exit $code)"; exit 1 }
}

$ckpts = (@(42) + $seeds | ForEach-Object { "checkpoints\best_model_seed$_.pth" }) -join ","
Write-Output "=== Evaluating ensemble: $ckpts ($(Get-Date -Format HH:mm:ss)) ==="
$code = Invoke-Py @('-u','ensemble_eval.py','--checkpoints',$ckpts,'--tta') "ensemble_eval.log" "ensemble_eval.err"
if ($code -ne 0) { Write-Output "ensemble eval FAILED (exit $code)"; exit 1 }

Write-Output "ENSEMBLE PIPELINE COMPLETE ($(Get-Date -Format HH:mm:ss))"
