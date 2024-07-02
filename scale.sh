task_level="graph"
scale_type="data"
data_array=(0.1 0.2 0.3 0.4 0.5 0.6)
emb_array=(50 100 200 300 400 500)
if [ "$task_level" == "graph" ]; then
    if [ "$scale_type" == "data" ]; then
        for a in ${data_array[@]}
        do
            python main_graph.py --dataset ogbg-molpcba --training_ratio $a --scale_type data
        done
    else
        for a in ${emb_array[@]}
        do
            python main_graph.py --dataset ogbg-molpcba --emb_dim $a --scale_type model
        done
    fi
elif [ "$task_level" == "node" ]; then
    if [ "$scale_type" == "data" ]; then
        for a in ${data_array[@]}
        do
            python main_node.py --dataset ogbn-arxiv --training_ratio $a --scale_type data
        done
    else
        for a in ${emb_array[@]}
        do
            python main_node.py --dataset ogbn-arxiv --emb_dim $a --scale_type model
        done
    fi
else
    if [ "$scale_type" == "data" ]; then
        for a in ${data_array[@]}
        do
            python main_link.py --dataset ogbl-collab --training_ratio $a --scale_type data
        done
    else
        for a in ${emb_array[@]}
        do
            python main_link.py --dataset ogbl-collab --emb_dim $a --scale_type model
        done
    fi
fi