# WholeGraph Benchmark
To evaluate the performance of WholeGraph, we focus on the GNN gathering feature operation, which accesses the embedding table stored in different types of WholeMemory, and measure its impact on the end-to-end GNN training tasks.

## Gathering Performance 
The gatering operation is a common and time-consuming task in large-scale GNN training systems. It involves fetching random node or edge features from the entire embedding table. The table below shows the benchmark results of the gathering operation on a DGX-A100 system.
The theoretical top algorithm bandwidth (AlgoBW) of DGX-A100 is:

$ BusBW / (7/8) = 342.86 $ GB/s.

As we can see, the chunked device WholeMemory is the best option.
The efficiency of the gatering operation using chunked device of WholeMemory can reach up to $77\%$, when embedding dimension is $32$. 
The host type of WholeMemory is limited to the bandwidth of PCIe. 


| Embedding dimension |    Chunked device  |   Continous device  |  Distributed device |  Continuous/Chunked host |  Distributed host |
|       :----:        |      :------:      |       :------:      |       :------:      |        :------:          |       :------:    |
|         4           |        51.26       |        0.37         |        28.61        |          0.31            |         1.75      |
|         8           |        102.53      |        0.73         |        50.97        |          0.62            |         3.17      |
|         16          |        205.02      |        1.43         |        80.48        |          1.23            |         6.05      |
|         32          |        264.16      |        2.78         |        113.29       |          2.47            |         11.73     |
|         64          |        260.99      |        5.35         |        133.25       |          4.91            |         12.2      |
|         128         |        261.03      |        10.35        |        144.61       |          9.73            |         12.31     |
|         256         |        261.18      |        19.74        |        149.51       |          13.18           |         12.34     |
|         512         |        261.45      |        36.93        |        151.82       |          12.89           |         12.34     |
|         1024        |        260.25      |        68.66        |        155.28       |          13.18           |         12.34     |

Table : The AlgoBW bandwidth (GB/s) of gatering operation using different WholeMemory types on one DGX-A100 machine node. The driver version is 515.65.01. The cuda version is 12.1. The size of embdding table is 200 GB. Each GPU gathers 4 GB features. 


## End-to-end GNN Task Traning Performance 

For GNN training, we use PyTorch as the training framework. 
The chunked device memory type is used to store the graph structure with CSR format. The WholeMemory type of embedding feature table is either chunked device or continuous host
Regarding the cache influence on the end-to-end GNN training, we also show the results. 

### Node Classification Task

We use the [ogbn-papers100M](https://ogb.stanford.edu/docs/nodeprop/#ogbn-papers100M) dataset from [OGB](https://ogb.stanford.edu/docs/dataset_overview/) repository for the node classification task. This dataset is a citation graph with 111 million nodes and 1.6 billion edges. The task is a multi-class calssification, where the goal is to predict the category of the node in the graph. 
The tables below show the epoch time for training GNN models with different memory types and cache policies. The valid and test accuracy is also listed. As we can see, when the embedding feature table is chunked device, the performance is best. However, when the embedding feature table is bigger and cannot be placed in the whole GPU memory, the embedding feature table needs to be stored in host memory. The performance is limited by the connection between CPU and GPU.
In this situation, with cache support, the perfomance will be comparable with the chunked device memory storage. From our observation, with a cache ratio of 50%, we can obtain 4.89X and 2.02X speedup for cuGraph layer for training GraphSage and GAT models, respectively. For DGL layer, we can get 4.07X and 1.97X for training GraphSage and GAT models, respectively. 


|    model     |  DGL + WholeGraph | cuGraph + WholeGraph |   
|    :---:     |     :------:      |      :----------:     |
|   GraphSage  |       5.72        |         4.62          |
|      GAT     |       20.79       |         20.72         |

Table: One epoch time (s) for training GraphSage  and GAT models. The graph structure is chunked deivce. The embedding feature table is chunked device. 


|    model     |  DGL + WholeGraph Valid | DGL + WholeGraph Test |  cuGraph + WholeGraph Valid  |  cuGraph + WholeGraph Test  | 
|    :---:     |         :------:        |      :----------:     |         :------------:        |     :-----------------:      |
|   GraphSage  |           68.46%        |         65.21%        |            68.41%             |             65.04%           |       
|      GAT     |           68.11%        |         65.05%        |            68.03%             |             64.84%           |

Table: The valid and test accuracy of training GraphSage and GAT after 20 epochs. The graph structure is chunked deivce. The embedding feature table is chunked device.


|    model     |  DGL + WholeGraph | cuGraph + WholeGraph | 
|    :---:     |     :------:      |      :----------:     |
|   GraphSage  |       27.72       |         26.42         |
|      GAT     |       42.73       |         43.36         |

Table: One epoch time (s) for training GraphSage  and GAT models. The graph structure is chunked deivce. The embedding feature table is continuous host with no cache.  


|    model     |  DGL + WholeGraph Valid | DGL + WholeGraph Test |  cuGraph + WholeGraph Valid  |  cuGraph + WholeGraph Test  | 
|    :---:     |         :------:        |      :----------:     |         :------------:        |     :-----------------:      |
|   GraphSage  |           68.89%        |         65.54%        |            68.47%             |             65.01%           |       
|      GAT     |           68.19%        |         65.19%        |            67.93%             |             64.90%           |

Table: The valid and test accuracy of training GraphSage and GAT after 20 epochs. The graph structure is chunked deivce. The embedding feature table is chunked device.



|    model     |  DGL + WholeGraph | cuGraph + WholeGraph | 
|    :---:     |     :------:      |      :----------:     |
|   GraphSage  |       6.81        |         5.4           |
|      GAT     |       21.64       |         21.50         |

Table: One epoch time (s) for training GraphSage  and GAT models. The graph structure is chunked deivce. The embedding feature table is continuous host with 50% cache ratio. 


|    model     |  DGL + WholeGraph Valid | DGL + WholeGraph Test |  cuGraph + WholeGraph Valid  |  cuGraph + WholeGraph Test  | 
|    :---:     |         :------:        |      :----------:     |         :------------:        |     :-----------------:      |
|   GraphSage  |           68.23%        |         65.09%        |            68.00%             |             64.90%           |       
|      GAT     |           68.37%        |         65.11%        |            68.37%             |             65.15%           |

Table: The valid and test accuracy of training GraphSage and GAT after 20 epochs. The graph structure is chunked deivce. The embedding feature table is chunked device.




### Link Prediction Task






