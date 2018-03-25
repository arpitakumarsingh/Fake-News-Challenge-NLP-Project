#!/bin/sh


echo "Running Stance Analysis"
cd FNC-Project-Stance
python3 main.py

echo "Running Satire Analysis"

cd ../satire
python3 main.py
