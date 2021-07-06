#!/bin/bash
wget -O CoLA.zip "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FCoLA.zip?alt=media&token=46d5e637-3411-4188-bc44-5809b5bfb5f4"
unzip CoLA.zip
rm CoLA.zip
rm -r CoLA/original/
