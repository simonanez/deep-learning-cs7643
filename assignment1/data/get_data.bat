@echo off
if exist mnist_train.csv del /F /Q mnist_train.csv
if exist mnist_test.csv del /F /Q mnist_test.csv
curl -o mnist_train.csv https://pjreddie.com/media/files/mnist_train.csv
curl -o mnist_test.csv https://pjreddie.com/media/files/mnist_test.csv
