#!/bin/bash

for i in {1..3}
do
  echo "Run $i started..."

  nohup python class_il_experiment.py --strategy naive --gpu_id 0 --log_dir class_il_experiment_naive_run$i &
  nohup python domain_il_experiment.py --strategy naive --gpu_id 0 --log_dir domain_il_experiment_naive_run$i &

  nohup python class_il_experiment.py --strategy cumulative --gpu_id 2 --log_dir class_il_experiment_cumulative_run$i &
  nohup python domain_il_experiment.py --strategy cumulative --gpu_id 3 --log_dir domain_il_experiment_cumulative_run$i &

  nohup python class_il_experiment.py --strategy expreplay --gpu_id 1 --log_dir class_il_experiment_expreplay_run$i &
  nohup python domain_il_experiment.py --strategy expreplay --gpu_id 1 --log_dir domain_il_experiment_expreplay_run$i &

  echo "Run $i launched."
done

echo "All runs launched."
