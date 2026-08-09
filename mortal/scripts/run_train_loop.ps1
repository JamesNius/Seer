# 训练守护脚本 — 崩溃自动重启
# 用法: powershell -File run_train_loop.ps1

while ($true) {
    Write-Host "===== 开始训练 $(Get-Date) ====="
    python train.py

    $exitCode = $LASTEXITCODE
    Write-Host "===== 训练退出, code=$exitCode, 时间=$(Get-Date) ====="

    if ($exitCode -eq 0) {
        # 正常完成一个 epoch，退出
        Write-Host "训练完成!"
        break
    }

    Write-Host "3 秒后自动重启..."
    Start-Sleep 3
}
