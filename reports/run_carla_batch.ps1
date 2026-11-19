Set-Location 'C:\Users\adhis\Downloads\capstone\aiml-04-perception-stack'
$log = 'reports\carla_batch.log'
"start $(Get-Date -Format s)" | Out-File -Encoding utf8 $log
function Start-Carla {
  # Ready = a client can fetch the world, not just an open TCP port (the port opens before the map has loaded).
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
Start-Carla
$env:CUDA_VISIBLE_DEVICES = ''
$common = @('--config', 'configs\pipeline_lowvram.yaml', '--weights', 'models\yolo11s_carla.pt', '--trials', '3')
$runs = @(
  @('--scenario', 'cpna25', '--speeds', '20,30,40'),
  @('--scenario', 'cpfa50', '--speeds', '20,30,40'),
  @('--scenario', 'ccrb', '--speeds', '30,40'),
  @('--scenario', 'cpna25', '--speeds', '20,30,40', '--oracle-depth'),
  @('--scenario', 'cpna25', '--speeds', '20,30,40', '--oracle-detection'),
  @('--scenario', 'cpna25', '--speeds', '20,30,40', '--oracle-detection', '--oracle-depth')
)
foreach ($r in $runs) {
  Start-Carla
  "run $($r -join ' ') $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
  & .venv\Scripts\python -u carla_tools\run_scenarios.py @r @common *>> $log
  "exit $LASTEXITCODE $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
}
Start-Carla
"falsify $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
& .venv\Scripts\python -u carla_tools\falsify_carla.py --budget 30 --config configs\pipeline_lowvram.yaml --weights models\yolo11s_carla.pt *>> $log
if (Test-Path reports\falsification.json) {
Start-Carla
"root cause $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
& .venv\Scripts\python -u carla_tools\root_cause.py --config configs\pipeline_lowvram.yaml --weights models\yolo11s_carla.pt *>> $log
}
"done $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
