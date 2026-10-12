param([switch]$CheckOnly)
$ErrorActionPreference='Stop'
$taskRepo=Split-Path -Parent $PSScriptRoot
$taskDrive=$taskRepo.Substring(0,1).ToLowerInvariant()
$taskWslRepo='/mnt/'+$taskDrive+$taskRepo.Substring(2).Replace('\','/')
$taskPython='/home/ubuntu/miniforge3/envs/landing-drqv2/bin/python'
$taskDistro='Ubuntu-20.04'
function Invoke-TaskWsl([string[]]$TaskArguments) {
    $taskOutput=& wsl.exe @TaskArguments
    if($LASTEXITCODE -ne 0){throw "WSL command failed ($LASTEXITCODE). Training was not restarted."}
    return ($taskOutput -join "`n")
}
$taskModeScript='grep "RDP backend: use_gfxredir =" /mnt/wslg/weston.log | tail -n 1'
$taskMode=Invoke-TaskWsl @('--system','--user','root','--','sh','-c',$taskModeScript)
Write-Host $taskMode
if($CheckOnly){
    Invoke-TaskWsl @('-d',$taskDistro,'--cd',$taskWslRepo,'--',$taskPython,'-m','landing_repro.display_recovery_state','inspect')
    exit 0
}
if($taskMode -match 'use_gfxredir = 1\s*$'){
    Invoke-TaskWsl @('-d',$taskDistro,'--cd',$taskWslRepo,'--',$taskPython,'-m','landing_repro.show_training_window')
    exit 0
}
# Keep the distro alive so SectionFs is not destroyed again during reconnect.
$taskKeep=Start-Process -FilePath wsl.exe -ArgumentList @('-d',$taskDistro,'--','sleep','180') -WindowStyle Hidden -PassThru
$taskState=Invoke-TaskWsl @('-d',$taskDistro,'--cd',$taskWslRepo,'--',$taskPython,'-m','landing_repro.display_recovery_state','save')
$taskSaved=$taskState | ConvertFrom-Json
Write-Host ('Verified checkpoint at step '+$taskSaved.saved_steps)
$taskRestartScript='task_pid=$(pgrep -x weston); test -n "$task_pid" && test "$(readlink /proc/$task_pid/exe)" = /usr/bin/weston && kill -TERM "$task_pid"'
Invoke-TaskWsl @('--system','--user','root','--','sh','-c',$taskRestartScript)
$taskReady=$false
for($taskAttempt=0;$taskAttempt -lt 10;$taskAttempt++){
    Start-Sleep -Milliseconds 500
    $taskMode=Invoke-TaskWsl @('--system','--user','root','--','sh','-c',$taskModeScript)
    if($taskMode -match 'use_gfxredir = 1\s*$'){$taskReady=$true;break}
}
if(-not $taskReady){throw 'Shared-memory display is still unavailable. Model is saved; do not launch duplicate training.'}
$taskLog=Join-Path $taskRepo 'runs\display_recovery'
New-Item -ItemType Directory -Path $taskLog -Force | Out-Null
$taskStamp=Get-Date -Format 'yyyyMMdd_HHmmss'
Start-Process -FilePath wsl.exe -ArgumentList @('-d',$taskDistro,'--','bash',($taskWslRepo+'/start_height_curriculum_training.sh')) -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskLog ($taskStamp+'_stdout.log')) -RedirectStandardError (Join-Path $taskLog ($taskStamp+'_stderr.log')) | Out-Null
Write-Host 'Shared-memory display restored. Training is resuming from the verified checkpoint.'
