@echo off
echo 🚀 Starting EarlyPark System
echo ==========================

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker first.
    exit /b 1
)

REM Build and start containers
echo 🐳 Starting Docker containers...
docker-compose up --build -d

REM Wait for services to be ready
echo ⏳ Waiting for services to start...
timeout /t 10 /nobreak >nul

REM Check service health
echo 🏥 Checking service health...
curl -f http://localhost:5000/api/health || echo ⚠️ API health check failed
curl -f http://localhost:8080/health || echo ⚠️ Web health check failed

echo.
echo ✅ EarlyPark is running!
echo 📱 Web App: http://localhost:8080
echo 🔌 API: http://localhost:5000
echo 📊 API Health: http://localhost:5000/api/health
echo.
echo To stop: docker-compose down
echo To view logs: docker-compose logs -f