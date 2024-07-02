emb_array=(50 100 200 300 400 500)
for a in ${emb_array[@]}
do
    python main_graph.py --emb_dim $a --scale_type model
done