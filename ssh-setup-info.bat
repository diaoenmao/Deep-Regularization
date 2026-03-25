@echo off
REM ================================
REM  SSH 免密登录配置说明
REM ================================
REM
REM 方法 1: 使用 SSH 密钥（推荐）
REM -------------------------
REM 1. 在终端执行以下命令（需要输入一次密码）:
REM    type %USERPROFILE%\.ssh\id_ed25519_matpool.pub | ssh -p 28128 root@hz-4.matpool.com "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && echo '公钥已添加'"
REM
REM 2. 之后使用以下命令免密登录:
REM    ssh -p 28128 -i %USERPROFILE%\.ssh\id_ed25519_matpool -o IdentitiesOnly=yes root@hz-4.matpool.com
REM
REM 方法 2: 使用 WinSCP 保存会话
REM -------------------------
REM 1. 下载 WinSCP: https://winscp.net
REM 2. 新建会话，输入：
REM    - 主机：hz-4.matpool.com
REM    - 端口：28128
REM    - 用户名：root
REM    - 密码：HCVq8ITrn*gv}jbW
REM 3. 点击"保存"，之后可直接连接
REM
REM 方法 3: 使用 Plink（PuTTY 组件）保存密码
REM -------------------------
REM 1. 下载 PuTTY: https://www.chiark.greenend.org.uk/~sgtatham/putty/
REM 2. 执行：plink -ssh -p 28128 root@hz-4.matpool.com -pw "HCVq8ITrn*gv}jbW"
REM
REM ================================

echo.
echo 当前保存的密码：HCVq8ITrn*gv}jbW
echo.
echo 快速连接命令:
echo   ssh -p 28128 -i %%USERPROFILE%%/.ssh/id_ed25519_matpool -o IdentitiesOnly=yes root@hz-4.matpool.com
echo.
pause
