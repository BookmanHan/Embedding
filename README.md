# Embedding
This Project is contributed by Xiao Han in Tsinghua University.

# Datasets
http://www.ibookman.net
http://ciia.cs.tsinghua.edu.cn

## Supported Papers
ManifoldE (IJCAI.2016): https://arxiv.org/pdf/1512.04792.pdf
TransG (ACL.2016): https://arxiv.org/pdf/1509.05488.pdf
TransA (submitting to AAAI.2017): https://arxiv.org/pdf/1509.05490.pdf
SSP (submitting to AAAI.2017): https://arxiv.org/pdf/1604.04835.pdf
KSR (submitting to AAAI.2017): https://arxiv.org/pdf/1608.07685.pdf

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


## Basic Config
-	Windows
	-	This project is naturally built on Visual Studio 2013 with Intel C++ Compiler 2016. If we share the same development perference, I guess you could start your work, right now.
	-	When you decide to compile it with MSC, there is a little trouble, because you shoud adjust your configuration.

-	Linux / MAC
	-	I also apply the Intel C++ Compiler, which could be substituted by GCC, theoretically.
	-	`icc -std=c++11 -O3 -xHost -qopenmp -m32 Embedding.cpp`

## Start
-	To justify you data source, please modify the `MultiChannelEmbedding\DetailedConfig.hpp`.
-	To explore the correspondding method, just fill the template in `MultiChannelEmbedding\Embedding.cpp` with hyper-parameters.
	
`	model = new MFactorE(FB15K, LinkPredictionTail, report_path, 10, 0.01, 0.1, 0.01, 10);`

`	model->run(500);`

`	model->test();`

`	delete model;`


## Alias
-	OrbitE = ManifoldE
-	MFactorE = KSR
