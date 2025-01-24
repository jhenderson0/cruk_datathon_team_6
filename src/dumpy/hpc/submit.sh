#!/usr/bin/env bash
# A script to be used alongside decombinator to automatically batch submit jobs

# Record script start location
STARTDIR=$(pwd)

TEMPDIR=temp
# Remember to change job script based on species and protocol
JOB=dumpy.qsub.sh

mkdir $TEMPDIR

# Move all fastq files to directory
find . -type f -name *.tsv.gz -exec mv {} $TEMPDIR/ \;

# Loop over fastq files and create folders per sample
for i in $TEMPDIR/*
do
    # checks that i exists pre loop iteration
    [ -e "$i" ] || continue
    ID=${i%%_?.*}
    echo $ID
    # create folder only if doesn't already exist e.g. X_2.fq
    # doesn't replace folder created by X_1.fq
    mkdir -p $ID
    mv $i $ID/
done

# Loop over sample directories, copy job in, submit job
for i in $TEMPDIR/*
do
    [ -e "$i" ] || continue
    cp $JOB $i/
    cd $i/
    qsub $JOB
    cd $STARTDIR
done
