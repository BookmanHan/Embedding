#include "Model.hpp"
#include "Embedding_MultiChannel.hpp"
#include "Embedding_Geometric.hpp"
#include <omp.h>

// 放回采样
// GeometricEmbeddingHadamard(100, 0.02) = 55.5%
// TransGMPE(50, 0.02, 1) @ Round.100 = 82.9 %
// TransGMPE(20, 0.02, 1) @ Round.350 = 83.6 %
// TransGGMPR(20, 0.02, 1) @ Round.240 = 84.2 %
// TransGMPE(200, 0.02, 1) @ Round.300 = 82.9 % 
// TransGGMPR(50, 0.02, 1) = 84.5 
// TransGGMPR(5, 0.02, 1) = 81.2 %

// 不放回采样。
// Wordnet18 @ TransE : TransGMPE = 90.0% : 97.3%
// Wordnet11 @ TransE : TransGMPE = 75.9% : 83.1%
// Wordnet11 @ TransR : TransGMPE = 85.5% : 85.5%

int main(int argc, char* argv[])
{
	omp_set_num_threads(4);
	EmbeddingModel*	Model = new TransGGMPR(50, 0.02, 1);
	Model->run();

	return 0;
}