#!/bin/bash

#PBS -l walltime=20:00:00
#PBS -l select=1:ncpus=4:mem=10gb:scratch_ssd=5gb
#PBS -j oe

# -j oe  - error + output merged

trap 'clean_scratch' TERM EXIT

DATADIR="/storage/brno2/home/$PBS_O_LOGNAME/"
OUTDIR="$out"
DATASET_TAR="$dataset.tar.gz"
if [ ! -d "$SCRATCHDIR" ] ; then echo "Scratch directory is not created!" 1>&2; exit 1; fi #checks if scratch directory is created

mkdir $SCRATCHDIR/datasets
mkdir $SCRATCHDIR/outputs_training
cp -r $DATADIR/$DATASET_TAR $SCRATCHDIR/datasets || exit 1
cp -r $DATADIR/Bachelor/* $SCRATCHDIR || exit 1
cp -r $DATADIR/FullIJCNN2013.zip $SCRATCHDIR || exit 1
cd $SCRATCHDIR || exit 2
unzip $SCRATCHDIR/FullIJCNN2013.zip  || exit 1

cd $SCRATCHDIR/datasets/
tar -xvf *.tar.gz || exit 1
cd ../

bench_dataset="$SCRATCHDIR/FullIJCNN2013/"
touch output

module add python36-modules-gcc
source /storage/brno2/home/$PBS_O_LOGNAME/soft/etc/profile.d/conda.sh

pyt36_env="/storage/brno2/home/xpanic00/soft/envs/pyt36/bin"
conda_path="/storage/brno2/home/xpanic00/soft/condabin"
export PATH="$pyt36_env:$conda_path:$PATH"
export PYTHONPATH="$pyt36_env:$conda_path:$PYTHONPATH"


python3.6 training.py $config --use-cuda >> output
echo "$config finished"

model_name=$(basename -- "${model%.*}")
cp $SCRATCHDIR/$benchmark $OUTDIR
cp $SCRATCHDIR/output $OUTDIR 
cp -r $SCRATCHDIR/outputs_folder $OUTDIR
cp $SCRATCHDIR/models/$model $OUTDIR
