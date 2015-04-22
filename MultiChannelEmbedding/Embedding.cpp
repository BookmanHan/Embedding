#include "Model.hpp"
#include "Embedding_MultiChannel.hpp"
#include "Embedding_Geometric.hpp"
#include <omp.h>

// 放回采样
// GeometricEmbeddingHadamard(100, 0.02) = 55.5%
// TransGGMPR(20, 0.02, 1) @ Round.240 = 84.2 %
// TransGMPE(200, 0.02, 1) @ Round.300 = 82.9 % 
// TransGGMPR(50, 0.02, 1) = 84.5 %
// TransGGMPR(5, 0.02, 1) = 81.2 %

// 不放回采样。
// Wordnet18 @ TransE : TransGMPE = 90.0% : 97.3%
// Wordnet11 @ TransE : TransGMPE = 75.9% : 83.1% (83.6%)
// Wordnet11 @ TransR : TransGGMPR = 85.5% : 85.7% (86.7%)

int main(int argc, char* argv[])
{
	omp_set_num_threads(4);
	EmbeddingModel*	Model = new TransGGMPRM(50, 0.01, 1);
	Model->run(1000);

	return 0;
}