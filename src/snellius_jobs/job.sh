#!/bin/bash
#SBATCH --job-name=run_sor    
#SBATCH --output=sor_out_%j.txt
#SBATCH --error=sor_error_%j.txt
#SBATCH --ntasks=1             # number of tasks
#SBATCH --nodes=1              # number of nodes              
#SBATCH --ntasks-per-node=1    # number of tasks per node
#SBATCH --cpus-per-task=128    # number of cpus per task
#SBATCH --time=03:00:00        # time limit hrs:min:sec
#SBATCH --partition=rome       # partition, check here https://servicedesk.surf.nl/wiki/display/WIKI/Snellius+partitions+and+accounting

# load python if needed
module load 2024
# here is the module you needed, but for this example, we don't need it

# check python version
echo "Checking Python version:"
python --version

# check numba
# echo "Checking Numba installation:"
# python -c "import numba; print('Numba version:', numba.__version__)"

# install some packages if allowed, but seems not allowed...
# pip install --user numba

# set OpenMP for numba(numba based on OpenMP)
export OMP_NUM_THREADS=128  # 128 threads for numba, like prange
python task.py
