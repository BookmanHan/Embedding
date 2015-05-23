#include "Model.hpp"
#include "Embedding_MultiChannel.hpp"
#include "Embedding_Geometric.hpp"
#include <omp.h>
#include <cstdlib>
#include <windows.h>

// 放回采样
// GeometricEmbeddingHadamard(100, 0.02) = 55.5%
// TransGGMPR(20, 0.02, 1) @ Round.240 = 84.2 %
// TransGMPE(200, 0.02, 1) @ Round.300 = 82.9 % 
// TransGGMPR(50, 0.02, 1) = 84.5 %
// TransGGMPR(5, 0.02, 1) = 81.2 %

// 不放回采样。
// Wordnet18 @ TransE : TransGMPE = 90.0% : 97.3%
// Wordnet11 @ TransE : TransGMPE = 75.9% : 84.3%
// Wordnet11 @ TransR : TransGGMPR = 85.5% : 86.2%
// Freebase13 @ TransE : TransGMPE =

// TransGMPE(50, 0.005) @ FB15K = 274, 26.2%;
int main(int argc, char* argv[])
{
	omp_set_num_threads(4);
	EmbeddingModel*	Model = new TransA2(50, 0.003);
	Model->run(5000);

	return 0;
}