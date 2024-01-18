# ALTEGRAD Kaggle competition

The goal of this project is to study and apply machine learning/artificial intelligence techniques to retrieve molecules (graphs) using natural language queries. Natural language and molecules encode information in very different ways, which leads to the exciting but challenging problem of integrating these two very different modalities. In this challenge, given a text query and list of molecules (represented as graphs), without any reference or textual information of the molecule, you need to retrieve the molecule corresponding to the query. This requires the integration of two very different types of information: the structured knowledge represented by text and the chemical properties present in molecular graphs. 

The pipeline to deal with this task can be achieved by co-training a text encoder and a molecule encoder using contrastive learning. This involves simultaneously training two separate encoders—one specialized in handling textual data and the other focused on molecular structures. Through contrastive learning, the model learns to map similar text-molecule pairs closer together in the learned representation space while pushing dissimilar pairs apart.
