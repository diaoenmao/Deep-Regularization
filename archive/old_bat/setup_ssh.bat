@echo off
REM SSH 免密登录配置脚本
REM 1. 首先手动输入一次密码，将公钥传到服务器
echo Step 1: Copy public key to remote server
echo Please enter the password when prompted
ssh -p 28128 root@hz-4.matpool.com "mkdir -p ~/.ssh && echo %USERPROFILE%/.ssh/id_ed25519_matpool.pub >> ~/.ssh/temp_key && chmod 644 ~/.ssh/temp_key"
echo.
echo If above failed, manually run:
echo   type %%USERPROFILE%%\.ssh\id_ed25519_matpool.pub | ssh -p 28128 root@hz-4.matpool.com "cat >> ~/.ssh/authorized_keys"
echo.
echo Step 2: Test connection with private key
ssh -p 28128 -i %%USERPROFILE%%/.ssh/id_ed25519_matpool -o IdentitiesOnly=yes root@hz-4.matpool.com "echo 'SSH key authentication successful!'"
