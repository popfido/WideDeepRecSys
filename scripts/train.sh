#!/usr/bin/env bash
# The script train the model.
# Usage:
#		1. bash train.sh  # default to train all 3 type model
#       2. bash train.sh  $model_name  # explict model name to train that type model.

set -e

cur_dir=$(cd `dirname $0`; pwd)
log_dir=`dirname $cur_dir`/log
exec_file=`dirname $cur_dir`/train.py

if [ $# -eq 0 ]; then
    nohup python $exec_file --model_type wide_deep > $log_dir/wide_deep.log 2>&1 &
    nohup python $exec_file --model_type wide > $log_dir/wide.log 2>&1 &
    nohup python $exec_file --model_type deep > $log_dir/deep.log 2>&1 &
elif [ $# -eq 1 ]; then
    nohup python $exec_file --model_type $1 > $log_dir/$1 2>&1 &
fi

