emb_array=(50 100 200 300)
for a in ${emb_array[@]}
do
    python main_graph.py --training_ratio $a --scale_type model
done