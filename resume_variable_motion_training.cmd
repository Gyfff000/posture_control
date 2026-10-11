@echo off
title UAV Landing - Resume Variable Motion Training
wsl -d Ubuntu-20.04 --cd /mnt/d/paper_reproductions/posture_control --exec bash start_variable_motion_training.sh
echo.
echo Training launcher has exited. Saved models remain in the runs directory.
pause
