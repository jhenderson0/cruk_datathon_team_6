# These are flags you must include - Two memory and one runtime.
# Runtime is either seconds or hours:min:sec
# 12hr selected to allow for instances of extended collapsing and server delay

#$ -l tmem=4G
#$ -l h_vmem=4G
#$ -l h_rt=1:00:00

# These are optional flags but you probably want them in all jobs

#$ -S /bin/bash
#$ -j y
#$ -N tcr_pipeline
#$ -cwd
#$ -l h=!arbuckle

# Most recent sequencing protocols return raw data that is already demultiplexed.
# Therefore, for most cases nowadays, running Demultiplexor is no longer required.

# Print useful troubleshooting information
hostname
date
echo $PWD

# Specify location of tags
PROJECTDIR=/SAN/colcc/tcr_decombinator/datathon

# Setup python enviroment
source /share/apps/source_files/python/python-3.11.9.source
source $PROJECTDIR/dumpy/bin/activate
python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))'

# Get file name from directory and strip any directory information
FILENAME=$(find . -type f -name *.tsv.gz -exec basename {} \;)
echo $FILENAME

python3 $PROJECTDIR/predict.py

echo "Job complete."
