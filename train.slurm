#!/bin/bash
#SBATCH -N 1  # Nombre de noeud
#SBATCH --ntasks-per-node=1 # nb of tasks = nb GPUs
#SBATCH --output=array_tree_optimizer_lion_loss_recall_ce_lr0001.log 
#SBATCH --cpus-per-task=2 #nb_cpus_total/nb_gpus , nb_cpus_total = nb_gpu + nb_data_worker, classicgpu=6, gpu40G=16, gpu80G=6
#SBATCH --gres=gpu:1
#SBATCH --time=0-05:00:00 # 40h
#SBATCH --job-name="lion" ##
#SBATCH --mem-per-cpu=3G  # memory per node: RAM total/nb_gpus 32G#SBATCH  
#SBATCH -p prismgpup #gpuv100 #gpu80G #prismgpup #gpu80G #classicgpu #gpu40G #gpup6000 #gpu40G #classicgpu #gpu40G #classicgpu #gpuv100 #classicgpu  ## --mem=0
#SBATCH --mail-user=matthieu.paques@cea.fr
#SBATCH --mail-type=ALL

module load anaconda/4.9.2
#conda activate pytorch
conda activate libjpeg

echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Starting..."

#python -m torch.distributed.launch --nproc_per_node=1 --master_port=$((RANDOM + 10000))  train.py --data_path datasets/cone/ --pretrained_weights weights/upernet_augreg_adapter_tiny_512_160_ade20k.pth --optimizer lion --loss ce --output_dir output_optim_lion_loss_ce_lr0001/ --lr 0.001
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$((RANDOM + 10000))  train_tree.py --data_path datasets/tree/ --pretrained_weights weights/upernet_augreg_adapter_tiny_512_160_ade20k.pth --optimizer lion --loss recall_ce --output_dir output_tree_optim_lion_loss_recall_ce_lr0001/ --lr 0.001


echo "Done!"


