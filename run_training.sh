network=("swin_s" "mba_net")
SIZE=(512)

#training for the combined approach
for i in ${!network[@]}; do
  for t in ${!SIZE[@]}; do
      echo "${network[$i]} for embedding size ${SIZE[$t]}"  
      # echo "${network[$i]} marging ${gestures[$j]} with probability ${prob[$k]} and aplha ${prob[$t]}"  
      python train.py --data_dir $1 --mask_dir . --output_path $2 --num_features ${SIZE[$t]} --batch_size 128 --val_split 0.1 --hand right --backbone ${network[$i]} --num_workers 2 --max_epochs 30 --val_split 0.002 --val_freq 5
    done
done


