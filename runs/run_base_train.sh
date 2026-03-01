# d24 model (slightly overtrained is enough to beat GPT-2 => increase data:params ratio from compute optimal 10.5 (default) to 12)
torchrun --standalone --nproc_per_node=16 --master-addr=127.0.0.1 -m scripts.base_train -- \
--depth=26 \
--target-param-data-ratio=8.25 \
--device-batch-size=8 \
--eval-every=100 \
--save-every=100 \
--run=$WANDB_RUN
