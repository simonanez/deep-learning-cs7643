@echo off
if exist assignment_1_submission.zip del /F /Q assignment_1_submission.zip
tar -a -c -f assignment_1_submission.zip configs models/*.py optimizer/*.py *.py
