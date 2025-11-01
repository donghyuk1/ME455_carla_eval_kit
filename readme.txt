conda env create -f environment.yml
conda activate carla_eval

# Terminal 1
./CarlaUE4.sh --world-port=2000 -opengl
# Terminal 2
./leaderboard/scripts/local_evaluation_auto_myauto.sh {carla path} {path to this dir, ex. ~/carla_eval_kit}

Tip
To stop Carla evaluation, run

KW=leaderboard bash kill_all_process.sh 
