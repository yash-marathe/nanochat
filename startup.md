#Pull the latest Docker Repo
docker pull rocm/pytorch

#Run the Container
docker run -it \
    --network host \
    --ipc host \
    --device /dev/dri \
    --device /dev/kfd \
    --device /dev/infiniband \
    --group-add video \
    --group-add render \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    -v $HOME:$HOME \
    -v $HOME/.ssh:/root/.ssh \
    --shm-size 1024G \
    --name nano \
    rocm/pytorch

#install amd-smi for GPU monitering
sudo apt update
sudo apt install amd-smi-lib
export PATH="${PATH:+${PATH}:}~/opt/rocm/bin"
amd-smi --help

git clone https://github.com/yashmarathe/nanochat.git
cd nanochat

uv venv
source .venv/bin/activate

#Speedrun
export PYTHONPATH=/nanochat:$PYTHONPATH
bash speedrun.sh

#8x steup
torchrun --standalone --nproc_per_node=8 -m scripts.base_train.py --device_batch_size=64 --depth=26 --total_batch_size=8388608 --run=$WANDB_RUN

export WANDB_RUN="my-llm-training-run-001"
export PYTHONPATH=/nanochat:$PYTHONPATH
screen -L -Logfile nanochat.log -S nanochat
torchrun --standalone --nproc_per_node=8 scripts/base_train.py \
    -- \
    --device_batch_size=32 --depth=26 --total_batch_size=8388608 --run=$WANDB_RUN
