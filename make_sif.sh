#!/bin/bash
#SBATCH --partition=small-creator
#SBATCH --time=01:00:00
#SBATCH --output=log/build_%J.log

rm -f env/env.sif

apptainer build env/env.sif env/env.def

if [ $? -eq 0 ]; then
    echo "Build successful!"
else
    echo "Build failed..."
    exit 1
fi
