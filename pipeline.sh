#!/bin/sh


echo "Running Stance Analysis"
cd FNC-Project-Stance
python3 stance_main.py

echo "Running Satire Analysis"

cd ../satire
python3 satire_main.py
