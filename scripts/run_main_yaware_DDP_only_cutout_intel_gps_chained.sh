#!/bin/bash
# -*- coding: utf-8 -*-

iterations=1 # 총 몇 번이나 연속으로 돌릴 것인지
jobid=$(sbatch --parsable /global/cfs/cdirs/m3898/yAwareContrastiveLearning/scripts/run_main_yaware_DDP_only_cutout_intel_gps.slurm)

for((i=0; i<$iterations; i++)); do            
    dependency="afterany:${jobid}"
    echo "dependency: $dependency"
    jobid=$(sbatch --parsable --dependency=$dependency /global/cfs/cdirs/m3898/yAwareContrastiveLearning/scripts/run_main_yaware_DDP_only_cutout_intel_gps.slurm)
    dependency=",${dependency}afterany:${jobid}"
done
