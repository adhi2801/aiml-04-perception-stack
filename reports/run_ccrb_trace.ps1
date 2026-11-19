Set-Location 'C:\Users\adhis\Downloads\capstone\aiml-04-perception-stack'
$log = 'reports\carla_trace.log'
"waiting for batch $(Get-Date -Format s)" | Out-File -Encoding utf8 $log
# The NCAP batch runs in PowerShell process 45012; the GPU only fits one CARLA client at a time.
function Start-Carla {
  for ($attempt = 0; $attempt -lt 3; $attempt++) {
    if (-not (Get-Process CarlaUE4-Win64-Shipping -ErrorAction SilentlyContinue)) {
      "launching carla $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
      Start-Process '..\CARLA_0.9.16\CarlaUE4.exe' -ArgumentList '-quality-level=Low','-dx12','-windowed','-ResX=320','-ResY=180','-nosound','"-ini:Engine:[/Script/Engine.RendererSettings]:r.Streaming.PoolSize=300"'
    }
    for ($i = 0; $i -lt 40; $i++) {
      & .venv\Scripts\python -c "import carla; c=carla.Client('127.0.0.1',2000); c.set_timeout(10); print(c.get_world().get_map().name)" 2>$null
      if ($LASTEXITCODE -eq 0) { "carla ready $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log; return }
      Start-Sleep 5
    }
    "carla not ready, killing $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
    Get-Process CarlaUE4* -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep 10
  }
  throw 'CARLA never became ready'
}
$env:CUDA_VISIBLE_DEVICES = ''
Start-Carla
"root cause $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
& .venv\Scripts\python -u carla_tools\root_cause.py --config configs\pipeline_lowvram.yaml --weights models\yolo11s_carla.pt *>> $log
"exit $LASTEXITCODE $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
$variants = @(@('--trial', '0'), @('--trial', '0', '--oracle-detection'), @('--trial', '0', '--oracle-depth'), @('--trial', '0', '--oracle-detection', '--oracle-depth'))
foreach ($v in $variants) {
  Start-Carla
  "trace ccrb 40 $($v -join ' ') $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
  & .venv\Scripts\python -u carla_tools\trace_scenario.py --scenario ccrb --speed 40 @v *>> $log
  "exit $LASTEXITCODE $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
}
"done $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
