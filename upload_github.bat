@echo off
REM Push this English mirror to GitHub.
REM Prerequisites (one-time):
REM   1. gh auth login            (or: set GH_TOKEN=ghp_xxx)
REM   2. git config --global user.name / user.email
REM Usage:
REM   upload_github.bat <your-github-username> [repo-name]
setlocal
set GH="C:\Program Files\GitHub CLI\gh.exe"
set GIT="C:\Program Files\Git\cmd\git.exe"
set USER=%1
if "%USER%"=="" (echo Usage: upload_github.bat ^<github-username^> [repo-name] & exit /b 1)
set REPO=%2
if "%REPO%"=="" set REPO=heart-protocol-en

cd /d "%~dp0"

%GIT% init -b main 2>nul
%GIT% add -A
%GIT% commit -m "English reference edition of the 16-Sephirot Heart Protocol" 2>nul

REM Create the remote repo (public) if it does not exist yet
%GH% repo create %USER%/%REPO% --public --source . --remote origin --push || (
    %GIT% remote remove origin 2>nul
    %GIT% remote add origin https://github.com/%USER%/%REPO%.git
    %GIT% push -u origin main
)
echo Done -^> https://github.com/%USER%/%REPO%
endlocal
