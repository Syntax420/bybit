@echo off
echo Beginne mit der Bereinigung...

REM Prüfen, ob der Logs-Ordner existiert
if not exist "C:\Users\Doktor\Desktop\Bybit\main\logs\" (
    echo Der Logs-Ordner existiert nicht!
) else (
    echo Leere die Log-Dateien...
    for %%F in ("C:\Users\Doktor\Desktop\Bybit\main\logs\*.log") do (
        echo Leere %%F
        type nul > "%%F"
    )
    echo Alle Log-Dateien wurden geleert.
)

REM Prüfen, ob der Cache-Ordner existiert
if not exist "C:\Users\Doktor\Desktop\Bybit\main\cache\" (
    echo Der Cache-Ordner existiert nicht!
) else (
    echo Lösche Dateien im Cache-Ordner...
    del /Q /F "C:\Users\Doktor\Desktop\Bybit\main\cache\*.*"
    echo Alle Dateien im Cache-Ordner wurden gelöscht.
)

echo Bereinigungsvorgang abgeschlossen.

REM Prüfen, ob die main.py existiert und starten
if not exist "C:\Users\Doktor\Desktop\Bybit\main\main.py" (
    echo Die Datei main.py wurde nicht gefunden!
    pause
    exit /b
)

echo Starte main.py mit Python 3...
cd /d "C:\Users\Doktor\Desktop\Bybit\main"
python3 main.py

REM Falls Python 3 nicht gefunden wird, versuche mit python zu starten
if %ERRORLEVEL% NEQ 0 (
    echo "Versuche mit 'python' statt 'python3'..."
    python main.py
)