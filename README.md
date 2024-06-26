# FedRL
This is the code for paper: A Reinforcement Learning Federated Recommender System for Efficient Communication Using Reinforcement Selector and Hypernet Generator.
![image](https://github.com/diyicheng/FedRL/blob/main/A3.png)
## Abstract
The field of recommender systems is currently experiencing robust development, with the primary goal of predicting users' latent interests in items through analyzing their preferences and behaviors. However, the collection of user data raises privacy concerns, leading to challenges related to incomplete initial information and data sparsity in recommender systems. To address these privacy concerns in the context of recommender systems, the concept of Federated Learning has emerged as a solution, allowing edge devices to train models locally without sharing their private data. However, federated recommender system faces the issue of heterogeneity among edge devices concerning data features and sample sizes. Moreover, disparities in computational and storage capabilities of these devices introduce communication overhead and processing delays during the aggregation of model parameters at the third-party server. This paper introduces a novel framework named A Reinforcement Learning Federated Recommender System for Efficient Communication Using Reinforcement Selector and Hypernet Generator (FedRL). The framework addresses the challenges posed by communication inefficiencies in federated recommender system through the integration of two key components: the Reinforcement Selector (RLS) and the Hypernet Generator (HNG). The RLS dynamically selects participating edge devices and assists them in maximizing the utilization of local data resources. Meanwhile, the HNG optimizes communication bandwidth consumption during the federated learning parameter transmission process, enabling rapid deployment and updates of new model architectures or hyperparameters. Furthermore, the proposed framework incorporates item attributes as content embeddings within the edge devices' recommender models, thereby enriching the models with global information. The efficacy of the proposed FedRL is evaluated within the context of initial information incompleteness and data sparsity in recommender systems. The collaborative approach between edge devices and a third-party server employs federated learning to train the recommender system model while preserving user privacy. Real world dataset experiments demonstrate that the proposed solution achieves a balance between recommender quality and communication efficiency, surpassing existing methods.
# Citation
Please cite our paper if you find this code useful for your research. If you have any questions, you can contact us at the email address dycwq123@gmail.com.
```
@article{DBLP: journals/tors/FedRL24,
  author       = {Yicheng Di, Hongjian Shi, Ruhui Ma, 
                  Honghao Gao, Yuan Liu and Weiyu Wang},
  title        = {FedRL: A Reinforcement Learning
                  Federated Recommender System for 
                  Efficient Communication Using
                  Reinforcement Selector and 
                  Hypernet Generator},
  journal      = {Trans. Recomm. Syst.},
  volume       = {X},
  number       = {X},
  pages        = {XXX},
  year         = {2024},
  url          = {XXX},
  doi          = {XXX},
}
```
# Get Data

Please find the data on [Google Drive](https://drive.google.com/drive/folders/1rBNzDV7F-60920h3RDac6HYTI9wyQuR8)

# Dependencies
At least 10GB GPU memory. At least 32GB memory.

- --tensorflow=2.3.0

- --numpy=1.16.0

- --python=3.7

- --keras=2.4.3

- --matplotlib=2.2.3
# Hyperparameters
If you use the same setting as our papers, you can simply adopt the hyperparameters reported in our paper. If you try a setting different from our paper, please tune the hyperparameters of FedRL. The user batch size in each training round is searched from {16, 32, 64, 128, 256}, the learning rate is searched from {0.001, 0.01, 0.1} based on the validation set performance, and the embedding size is adjusted from {4, 8, 16, 32, 64}. The coefficient for the target importance in the reward function of RLS is set to [a1,a2,a3] = [1,0.1,0.1].

