network="swin_s"
gesture=('call' 'dislike' 'fist' 'four' 'like' 'mute' 'ok' 'one' 'palm' 'peace' 'peace_inverted' 'rock' 'stop' 'stop_inverted' 'three' 'three2' 'two_up' 'two_up_inverted' 'None')

#training for the combined approach
for i in ${!gesture[@]}; do
    echo "Evaluating ${gesture[$i]}"  
    # echo "${network[$i]} marging ${gestures[$j]} with probability ${prob[$k]} and aplha ${prob[$t]}"  
    python test.py --data_dir $1 --mask_dir . --output_path "$2/${gesture[$i]}" --weights $3 --batch_size 64 --hand right --backbone $network --gesture ${gesture[$i]}
done


