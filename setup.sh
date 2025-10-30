#!/bin/bash

conda env create -f environment.yml

# Connect external kernel to jupyter
source activate base
conda activate als-env
PY_PATH=$(which python3)
echo "Using Python at: $PY_PATH"

# Create the kernels directory if it doesn't exist
mkdir -p ~/.local/share/jupyter/kernels/als-env/

# Create the kernel.json file with the correct path to Python
cat > ~/.local/share/jupyter/kernels/als-env/kernel.json << EOF
{
 "argv": [
  "$PY_PATH",
  "-m",
  "ipykernel",
  "-f",
  "{connection_file}"
 ],
 "display_name": "Python (als-env)",
 "language": "python"
}
EOF