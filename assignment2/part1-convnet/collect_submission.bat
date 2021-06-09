@echo off
if exist assignment_2_part_1_submission.zip del /F /Q assignment_2_part_1_submission.zip
tar -a -c -f assignment_2_part_1_submission.zip modules/*.py optimizer/*.py *.py