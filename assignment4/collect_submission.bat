@echo off
if exist assignment4_submission.zip del /F /Q assignment4_submission.zip
tar -a -c -f assignment4_submission.zip models Machine_Translation.ipynb