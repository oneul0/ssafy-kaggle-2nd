# 학습 중 화면보호기/화면 꺼짐을 막는 임시 프로그램 (윈도우 설정은 바꾸지 않는다)
#  - 학습이 모두 끝나면(outputs\logs\lora_all_done.txt 생성) 자동 종료
#  - 최대 10시간이 지나면 자동 종료
#  - 수동 종료: 작업 관리자에서 이 PowerShell 프로세스를 끝내거나, 아래 명령 실행
#      Get-CimInstance Win32_Process | ? { $_.CommandLine -match 'keep_awake' } | % { Stop-Process -Id $_.ProcessId -Force }
param([string]$DoneName = 'lora_all_done.txt')
Add-Type -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);' -Name Power -Namespace Win
$ES_CONTINUOUS = [uint32]2147483648  # 0x80000000
$ES_SYSTEM     = [uint32]1
$ES_DISPLAY    = [uint32]2
$root = Split-Path -Parent $PSScriptRoot
$done = Join-Path $root (Join-Path 'outputs\logs' $DoneName)
$end  = (Get-Date).AddHours(10)
while ((Get-Date) -lt $end -and -not (Test-Path $done)) {
    [void][Win.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM -bor $ES_DISPLAY)
    Start-Sleep -Seconds 30
}
[void][Win.Power]::SetThreadExecutionState($ES_CONTINUOUS)  # 원래 상태로 복원
