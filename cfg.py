import argparse
# vessl_on
# map_name1 = '6h_vs_8z'
# GNN = 'GAT'
def get_cfg():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--cuda", type=str, default='cuda:0', help="")
    parser.add_argument("--lmbda", type=float, default=0.95, help="GAE lmbda")
    parser.add_argument("--gamma", type=float, default=0.99, help="GAE lmbda")
    parser.add_argument("--eps_clip", type=float, default=0.18, help="clipping epsilon")
    parser.add_argument("--K_epoch", type=int, default=2, help="K-epoch")

    parser.add_argument("--ppo_layers", type=str, default='[128,64,48,32]', help="layer 구조")
    parser.add_argument("--optimizer", type=str, default='ADAM', help="optimizer")
    parser.add_argument("--scheduler_step", type =int, default=1000, help= "scheduler step")
    parser.add_argument("--scheduler_ratio", type=float, default=0.5, help= "scheduler ratio")
    parser.add_argument("--negativeslope", type=float, default=0.1, help="leaky relu negative slope")
    parser.add_argument("--map_name", type=str, default='6h_vs_8z', help="map name")

    parser.add_argument("--hidden_size_obs", type=int, default=32, help="")
    parser.add_argument("--hidden_size_action", type=int, default=36, help="")
    parser.add_argument("--n_representation_obs", type=int, default=28, help="")
    parser.add_argument("--n_representation_action", type=int, default=24, help="")
    parser.add_argument("--graph_embedding", type=int, default=32, help="")
    parser.add_argument("--lr", type=float, default=1.e-4, help="learning rate")
    parser.add_argument("--lr_graph", type=float, default=1.e-3, help="learning rate")
    parser.add_argument("--n_data_parallelism", type=int, default=5, help="")
    parser.add_argument("--grad_clip", type=float, default=5, help="")
    parser.add_argument("--vessl_on", type=bool, default=False, help="")
    parser.add_argument("--gamma1", type=float, default=1.0, help="")

    parser.add_argument("--sweep", type=bool, default=True, help="")



    # "hidden_size_obs": cfg.hidden_size_obs,
    # "hidden_size_action": cfg.hidden_size_action,
    # "n_representation_obs": cfg.n_representation_obs,
    # "n_representation_action": cfg.n_representation_action,
    # "graph_embedding": cfg.graph_embedding,
    # "learning_rate": cfg.lr,
    # "gamma": cfg.gamma,
    # "lmbda": cfg.eps_clip,
    # "K_epoch": cfg.K_epoch,
    # "layers": cfg.ppo_layers,
    # "feature_size": env.get_env_info()["node_features"],
    # "action_size": env.get_env_info()["n_actions"],


    # parser.add_argument("--GNN", type=str, default='GAT', help="map name")
    # parser.add_argument("--hidden_size_obs", type=int, default=32, help="GTN 해당")
    # parser.add_argument("--hidden_size_comm", type=int, default=60, help="")
    # parser.add_argument("--hidden_size_Q", type=int, default=84, help="GTN 해당")
    # parser.add_argument("--hidden_size_meta_path", type=int, default=42, help="GTN 해당")
    # parser.add_argument("--n_representation_obs", type=int, default=36, help="GTN 해당")
    # parser.add_argument("--n_representation_comm", type=int, default=69, help="")
    # parser.add_argument("--buffer_size", type=int, default=150000, help="")
    # parser.add_argument("--batch_size", type=int, default=32, help="")
    # parser.add_argument("--teleport_probability", type=float, default=0.9, help="teleport_probability")
    # parser.add_argument("--gtn_beta", type=float, default=0.1, help="teleport_probability")
    # parser.add_argument("--gamma", type=float, default=0.99, help="discount ratio")
    # parser.add_argument("--lr", type=float, default=1.3e-4, help="learning rate")
    # parser.add_argument("--n_multi_head", type=int, default=1, help="number of multi head")
    # parser.add_argument("--dropout", type=float, default=0.6, help="dropout")
    # parser.add_argument("--num_episode", type=int, default=1000000, help="number of episode")
    # parser.add_argument("--train_start", type=int, default=10, help="number of train start")
    # parser.add_argument("--epsilon", type=float, default=1.0, help="initial value of epsilon greedy")
    # parser.add_argument("--min_epsilon", type=float, default=0.05, help="minimum value of epsilon greedy")
    # parser.add_argument("--anneal_steps", type=int, default=50000, help="anneal ratio of epsilon greedy")
    # parser.add_argument("--algorithm", type=str, default="ppo", help="anneal ratio of epsilon greedy")







    return parser.parse_args()