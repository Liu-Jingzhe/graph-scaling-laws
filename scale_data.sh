data_array=(0.1 0.2 0.3 0.4 0.5 0.6)
for a in ${data_array[@]}
do
    python main_graph.py --dataset ogbg-molpcba --training_ratio $a --scale_type data
done