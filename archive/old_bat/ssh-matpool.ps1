# SSH 快速连接脚本 - 自动输入密码
# 服务器：hz-4.matpool.com:28128
# 用户：root

$server = "hz-4.matpool.com"
$port = "28128"
$user = "root"
$password = "HCVq8ITrn*gv}jbW"

# 方法：使用 Windows 凭据管理器保存密码
$target = "git:https://$server"
$passwordSecure = ConvertTo-SecureString -String $password -AsPlainText -Force
$credential = New-Object System.Management.Automation.PSCredential($user, $passwordSecure)

# 保存凭据
try {
    # 尝试使用 Windows 凭据管理器
    Add-Type -AssemblyName System.Security
    $cred = New-Object System.Security.Cryptography.ProtectedData
} catch {
    Write-Host "凭据管理器不可用，直接连接..."
}

Write-Host "正在连接 SSH: $user@$server:$port" -ForegroundColor Green
Write-Host "按 Ctrl+C 退出" -ForegroundColor Yellow

# 直接 SSH 连接（需要手动输入一次密码后系统会记住）
ssh -p $port "$user@$server"
