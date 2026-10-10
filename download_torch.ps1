# 下载 torch 2.4.1+cu124 三件套（阿里云镜像，curl 断点续传）
$ErrorActionPreference = "Continue"
$urls = @(
  "https://mirrors.aliyun.com/pytorch-wheels/cu124/torch-2.4.1%2Bcu124-cp311-cp311-win_amd64.whl",
  "https://mirrors.aliyun.com/pytorch-wheels/cu124/torchvision-0.19.1%2Bcu124-cp311-cp311-win_amd64.whl",
  "https://mirrors.aliyun.com/pytorch-wheels/cu124/torchaudio-2.4.1%2Bcu124-cp311-cp311-win_amd64.whl"
)
$dir = "E:\Binge project\gd12345_tmf_project\wheels"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
foreach ($u in $urls) {
  $f = Join-Path $dir ([System.IO.Path]::GetFileName($u))
  Write-Output "DOWNLOADING: $f"
  curl.exe -L -C - --connect-timeout 30 -o $f --progress-bar $u 2>&1
  if (Test-Path $f) {
    $sz = (Get-Item $f).Length
    Write-Output ("DONE: {0}  size={1:N0} bytes" -f $f, $sz)
  } else {
    Write-Output "FAILED: $f"
  }
}
Write-Output "ALL DOWNLOADS FINISHED"
