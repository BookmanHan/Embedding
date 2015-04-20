#include "Model.hpp"
#include "Embedding_MultiChannel.hpp"
#include "Embedding_Geometric.hpp"

// GeometricEmbeddingHadamard(100, 0.02) = 55.5%
// TransGMPE(50, 0.02, 1) @ Round.100 = 82.9 %
// TransGMPE(20, 0.02, 1) @ Round.350 = 83.6 %
// TransGGMPR(20, 0.02, 1) @ Round.240 = 83.9 % + 0.1 %
// TransGGMPR(50, 0.02, 1) = 84.5 %
// TransGMPE(200, 0.02, 1) @ Round.300 = 82.9 % 
// TransGGMPR(20, 0.02, 1) = 

int main(int argc, char* argv[])
{
	EmbeddingModel*	Model = new TransGGMPR(20, 0.02, 1);
	Model->run();

	return 0;
}