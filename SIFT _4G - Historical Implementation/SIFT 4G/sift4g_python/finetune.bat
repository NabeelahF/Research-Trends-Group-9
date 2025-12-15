@echo off
REM Fine-tune SIFT4G with different parameters (Windows batch script)

echo ========================================
echo SIFT4G Fine-Tuning
echo ========================================

REM Baseline (default parameters)
echo.
echo [1/4] Running baseline configuration...
python -m sift4g_python.main --query benchmark_query.fasta --database ..\sift4g-master\test_files\sample_protein_database.fa --out ./results_baseline --prob-cutoff 0.05 --seq-identity 90

REM Stringent (lower threshold, higher identity)
echo.
echo [2/4] Running stringent configuration...
python -m sift4g_python.main --query benchmark_query.fasta --database ..\sift4g-master\test_files\sample_protein_database.fa --out ./results_stringent --prob-cutoff 0.03 --seq-identity 95

REM Sensitive (higher threshold, lower identity, more sequences)
echo.
echo [3/4] Running sensitive configuration...
python -m sift4g_python.main --query benchmark_query.fasta --database ..\sift4g-master\test_files\sample_protein_database.fa --out ./results_sensitive --prob-cutoff 0.07 --seq-identity 85 --max-aligns 600

REM Balanced
echo.
echo [4/4] Running balanced configuration...
python -m sift4g_python.main --query benchmark_query.fasta --database ..\sift4g-master\test_files\sample_protein_database.fa --out ./results_balanced --prob-cutoff 0.05 --seq-identity 88 --max-aligns 500

echo.
echo ========================================
echo Fine-tuning complete!
echo ========================================
echo.
echo Results saved in:
echo   - results_baseline
echo   - results_stringent
echo   - results_sensitive
echo   - results_balanced
echo.
echo Next: Compare results using analyze_predictions.py
pause
