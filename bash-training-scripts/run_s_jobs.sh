#!/bin/bash 


single_skript="/storage/brno2/home/xpanic00/singleproc_train.sh"
multi_skript="/storage/brno2/home/xpanic00/multiproc_train.sh"
jobs_out="/storage/brno2/home/xpanic00/jobs_out"

configs_dir="/storage/brno2/home/xpanic00/to_be_done/queue/"
env_file="$HOME/env_file.sh"

for config_path in "$configs_dir"/*.json
    do
	python3 $HOME/Bachelor/prepare_enviroment.py $env_file $config_path 
	. $env_file

	name=$(basename -- "${config_path%.*}")
	model="$name.pt"
	config_folder="$jobs_out/$name"
	if [ ! -d "$config_folder" ]; then
	    mkdir $config_folder
        fi
	stamp=$(date +"%d_%m_%H_%M")
	output_folder="$config_folder/output_$stamp"
	mkdir $output_folder  || exit 1

	cp $config_path $output_folder
	new_config_path="$output_folder/${name}.json"

	qsub -v dataset="$DATASET",clsmode="$MODE",out="$output_folder",config="$new_config_path",model="$model" $single_skript

    done
