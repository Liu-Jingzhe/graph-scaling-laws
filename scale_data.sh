data_array=(0.01 0.02 0.04 0.06 0.1 0.2)
for a in ${data_array[@]}
do
    python main_graph.py --training_ratio $a --scale_type data
done