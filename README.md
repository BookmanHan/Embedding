# Embedding
This Project is contributed by Xiao Han in Tsinghua University.

## Datasets
-	KGE.zip

## Supported Papers
-	ManifoldE (IJCAI.2016): http://www.ibookman.net/IJCAI.2016.ManifoldE.pdf
-	TransG (ACL.2016): http://www.ibookman.net/ACL.2016.TransG.pdf
-	SSP (AAAI.2017): http://www.ibookman.net/AAAI.2017.SSP.pdf
-	TransA (Arxiv): http://www.ibookman.net/Arxiv.TransA.pdf
-	KSR (submitting to ACL.2017): http://www.ibookman.net/Arixv.KSR.pdf

## Citation
Conventionally, if this project helps you, please cite our paper, corresponddingly.
-	Han Xiao, Minlie Huang, Xiaoyan Zhu. From One Point to A Manifold: Orbit Models for Knowledge Graph Embedding. The 25th International Joint Conference on Artificial Intelligence (IJCAI'16).
-	Han Xiao, Minlie Huang, Xiaoyan Zhu. TransG: A Generative Mixture Model for Knowledge Graph Embedding. The 54th Annual Meeting of the Association for Computational Linguistics (ACL'2016).
-	Han Xiao, Minlie Huang, Lian Meng, Xiaoyan Zhu. SSP: Semantic Space Projection for Knowledge Graph Embedding with Text Descriptions. The Thirty-First AAAI Conference on Artificial Intelligence (AAAI'17).

## BibTex
```
@article{Xiao2015From,
  title={From One Point to A Manifold: Orbit Models for Knowledge Graph Embedding},
  author={Xiao, Han and Huang, Minlie and Hao, Yu and Zhu, Xiaoyan},
  journal={Computer Science},
  year={2015},
}

@article{Xiao2016TransG,
  title={TransG : A Generative Mixture Model for Knowledge Graph Embedding},
  author={Xiao, Han and Huang, Minlie and Hao, Yu and Zhu, Xiaoyan},
  journal={Computer Science},
  year={2016},
}

@article{Xiao2016SSP,
  title={SSP: Semantic Space Projection for Knowledge Graph Embedding with Text Descriptions},
  author={Xiao, Han and Huang, Minlie and Zhu, Xiaoyan},
  year={2016},
}
```

## Dependency
-	Armadillo
	-	Armadillo is a high quality linear algebra library (matrix maths) for the C++ language, aiming towards a good balance between speed and ease of use 
	-	I bet you could master it, just by scanning the examples.
	-	Download URL: [http://arma.sourceforge.net/download.html](http://arma.sourceforge.net/download.html "http://arma.sourceforge.net/download.html")
	-	What all you should do is to copy the headers into your environment.
-	Boost
	-	C++ Standard Extensive Library.
	-	Download URL:[http://www.boost.org/users/download/](http://www.boost.org/users/download/ "http://www.boost.org/users/download/")
	-	What all you should do is to copy the headers into your environment. Certainly, you could compile the code just as explained in the website.
-	MKL
	-	**Not Necessary**, but I strongly suggest you could take advantage of your devices.


## Basic Configuration
-	Windows
	-	This project is naturally built on Visual Studio 2013 with Intel C++ Compiler 2016. If we share the same development perference, I guess you could start your work, right now.
	-	When you decide to compile it with MSC, there is a little trouble, because you shoud adjust your configuration.

-	Linux / MAC
	-	I also apply the Intel C++ Compiler, which could be substituted by GCC, theoretically.
	-	`icc -std=c++11 -O3 -xHost -qopenmp -m32 Embedding.cpp`

## Start
-	To justify your data source, please modify the `MultiChannelEmbedding\DetailedConfig.hpp`.
-	To explore the correspondding method, just fill the template in `MultiChannelEmbedding\Embedding.cpp` with hyper-parameters.
	
	-	`	model = new MFactorE(FB15K, LinkPredictionTail, report_path, 10, 0.01, 0.1, 0.01, 10);`
	-	`	model->run(10000);`
	-	`	model->test();`
	-	`	delete model;`
-	**Notably, our code needs a little more turns to converge, we suggest 10,000 rounds for each experiment.** This is a critical trick for repeating our experiments.

## Alias
-	OrbitE = ManifoldE
-	MFactorE = KSR

## Code Structures
-	Import.hpp imports the headers of project.
-	DataModel.hpp specifies the data structure of knowledge graph.
-	Model.hpp specifies the training and testing process of knowledge embedding model.
-	DetailedConfig.hpp specifies the detailed configuration of project, such as file path.
-	Embedding.cpp is the main source file.
-	GeometricModel.hpp implements TransE, TransH and TransR.
-	OrbitModel.hpp implements ManifoldE.
-	SemanticModel.hpp implements SSP.
-	LatentModel.hpp implements KSR. 
