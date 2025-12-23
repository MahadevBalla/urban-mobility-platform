@echo off
REM Urban Transit Tool - Quick Start Script (Windows)

echo ======================================
echo  Urban Transit Tool - Docker Setup
echo ======================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo [X] Docker is not installed!
    echo Please install Docker Desktop from: https://docs.docker.com/get-docker/
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo [X] Docker Compose is not installed!
    echo Please install Docker Compose from: https://docs.docker.com/compose/install/
    exit /b 1
)

echo [OK] Docker and Docker Compose found
echo.

REM Create .env file if it doesn't exist
if not exist .env (
    echo [*] Creating .env file from example...
    copy .env.example .env
    echo [OK] .env file created
    echo [!] Please review .env and update passwords if needed
    echo.
)

REM Create necessary directories
echo [*] Creating necessary directories...
if not exist data mkdir data
if not exist cache mkdir cache
if not exist output_cache mkdir output_cache
if not exist logs mkdir logs
echo [OK] Directories created
echo.

REM Pull images
echo [*] Pulling Docker images...
docker-compose pull
echo [OK] Images pulled
echo.

REM Build application
echo [*] Building application...
docker-compose build
echo [OK] Application built
echo.

REM Start services
echo [*] Starting services...
docker-compose up -d
echo.

REM Wait for database
echo [*] Waiting for database to initialize...
timeout /t 10 /nobreak >nul
echo.

REM Check if services are running
docker-compose ps | findstr /C:"Up" >nul
if errorlevel 1 (
    echo.
    echo [X] Services failed to start!
    echo Check logs with: docker-compose logs
    exit /b 1
)

echo.
echo ======================================
echo  [OK] Setup Complete!
echo ======================================
echo.
echo Services:
echo   - Dashboard: http://localhost:8501
echo   - Database:  localhost:5432
echo   - pgAdmin:   http://localhost:5050 (start with: docker-compose --profile tools up -d)
echo.
echo Default Credentials:
echo   - Database User: urban_admin
echo   - Database Password: urban_transit_2024 (change in .env)
echo.
echo View Logs:
echo   docker-compose logs -f
echo.
echo Stop Services:
echo   docker-compose stop
echo.
echo Full Documentation:
echo   See DOCKER_SETUP.md for complete guide
echo.

pause
