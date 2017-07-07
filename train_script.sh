#!/bin/bash
DATE_TIME="`date +%m_%d_%H_%M`"
LOG_DIR="./model/$DATE_TIME"
#LOG_DIR="./model/04_18_21_31"
TMUX_SESSION_NAME="a3c"
NUM_THREADS=8
TENSORBOARD_PORT=15000

# Create the log directory.
mkdir -p $LOG_DIR

# Kill previous tmux session. Ignore potential "can't find session" messages
tmux kill-session -t $TMUX_SESSION_NAME 2> /dev/null

# Initial a new tmux session.
tmux new-session -s $TMUX_SESSION_NAME -n master -d

# Create a window for each learning thread.
for thread_id in $(seq 0 $((NUM_THREADS - 1))); do
    tmux new-window -t $TMUX_SESSION_NAME -n thread-$thread_id
done

# Create a window for TensorBoard.
tmux new-window -t $TMUX_SESSION_NAME -n tensorboard

# Create a window for observing hardware usage.
tmux new-window -t $TMUX_SESSION_NAME -n htop

# Wait for tmux to finish setting uo.
sleep 2

# Start the master thread, which synchronized worker threads.
#tmux send-keys -t $TMUX_SESSION_NAME:master "CUDA_VISIBLE_DEVICES=3" \
tmux send-keys -t $TMUX_SESSION_NAME:master "CUDA_VISIBLE_DEVICES=1" \
                                            " python thread.py" \
                                            " --log_dir=$LOG_DIR" \
                                            " --num_threads=$NUM_THREADS" \
                                            " $@" Enter

# Start workers threads.
for thread_id in $(seq 0 $(($NUM_THREADS - 1))); do
    tmux send-keys -t $TMUX_SESSION_NAME:thread-$thread_id \
        "CUDA_VISIBLE_DEVICES=$[thread_id % 3]" \
        " python thread.py" \
        " --log_dir=$LOG_DIR" \
        " --worker_index=$thread_id" \
        " $@" Enter
done

# Start Tensorboard.
tmux send-keys -t $TMUX_SESSION_NAME:tensorboard "tensorboard" \
                                                 " --port $TENSORBOARD_PORT" \
                                                 " --logdir $LOG_DIR" Enter
# start top
tmux send-keys -t $TMUX_SESSION_NAME:htop "htop" Enter

echo "Started the learning session."
echo "Started TensorBoard at localhost:$TENSORBOARD_PORT."
echo "Use 'tmux attach -t $TMUX_SESSION_NAME' to connect to the session."
echo "Use 'tmux kill-session -t $TMUX_SESSION_NAME' to end the session."
