@echo off
REM 直接 SSH 连接服务器（带密码）
REM 使用 plink（PuTTY 组件）支持 -pw 参数

REM 检查 plink 是否存在
where plink >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Connecting with plink...
    plink -ssh -p 28128 root@hz-4.matpool.com -pw "HCVq8ITrn*gv}jbW" -noagent
) else (
    echo plink not found. Please install PuTTY from:
    echo https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html
    echo.
    echo Or use: ssh -p 28128 root@hz-4.matpool.com (manual password entry)
)
