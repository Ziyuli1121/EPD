### Train ###

SOLVER_FLAGS="--sampler_stu=noise_ensemble --sampler_tea=dpm --num_steps=5 --M=3 --afs=True \
              --num_points=2 --noise_max=0.1 --noise_damping=0.0 --scale_dir=0.0 --scale_time=0.0"
SCHEDULE_FLAGS="--schedule_type=time_uniform --schedule_rho=1"

torchrun --standalone --nproc_per_node=1 --master_port=11111 \
  train.py --dataset_name=cifar10 --batch=8 --total_kimg=10 \
  $SOLVER_FLAGS $SCHEDULE_FLAGS



### Sample ###

torchrun --standalone --nproc_per_node=1 --master_port=22222 \
  sample.py --predictor_path=/home/perry/ziyul6/EPD/exps/00035-cifar10-5-7-noise_ensemble-dpm-3-uni1.0-afs/network-snapshot-000010.pkl --batch=128 --seeds=0-49999


### Evaluate ###

python fid.py calc \
  --images="/home/perry/ziyul6/EPD/samples/cifar10/noise_ensemble_nfe7_npoints_2" \
  --ref="/home/perry/ziyul6/EPD/cifar10-32x32.npz"