#!/bin/csh

#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --job-name=lammps_oley/ebind/ebind14long.

##SBATCH --mail-type=FAIL
##SBATCH --mail-user=u1276798@utah.edu ?

#SBATCH --account=owner-guest
#SBATCH --partition=kingspeak-guest

# Load lammps modules
module load gcc/8.5.0 openmpi/4.1.4 lammps/20220623

# Set relevant directories
setenv WORKDIR /uufs/chpc.utah.edu/common/home/u1276798/oley_au/sims/ebind/ebind14long.
setenv SCRDIR /scratch/local/$USER/$SLURM_JOB_ID

# Copy INPUT to SCRDIR
mkdir -p $SCRDIR 
cd $SCRDIR
cp -p $WORKDIR/* $SCRDIR

# Remove old trajectories
rm -rf log.lammps

# Run simulation
mpirun -np $SLURM_NTASKS lmp -in in.NP >& NP.out

# Copy OUTPUT to WORKDIR
cd $WORKDIR
cp -pR $SCRDIR/* $WORKDIR


