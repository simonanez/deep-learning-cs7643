@echo off
if exist assignment_2_part_2_submission.zip del /F /Q assignment_2_part_2_submission.zip
tar -a -c -f assignment_2_part_2_submission.zip configs losses/*.py checkpoints models/*.py main.py