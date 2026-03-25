@echo off
REM SSH 连接脚本 - 使用 PowerShell 自动输入密码
REM 密码：HCVq8ITrn*gv}jbW

powershell -Command "$Password = 'HCVq8ITrn*gv}jbW' | ConvertTo-SecureString -AsPlainText -Force; $Cred = New-Object System.Management.Automation.PSCredential('root', $Password); Start-Process ssh -ArgumentList '-p 28128 root@hz-4.matpool.com' -NoNewWindow -Wait"
