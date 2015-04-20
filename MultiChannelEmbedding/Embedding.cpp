#include "Model.hpp"
#include "Embedding_MultiChannel.hpp"
#include "Embedding_Geometric.hpp"

// GeometricEmbeddingHadamard(100, 0.02) = 55.5%
// TransGMPE(50, 0.02, 1) = 82.9 %
// TransGMPE(20, 0.02, 1) = 83.6 %;
int main(int argc, char* argv[])
{
	EmbeddingModel*	Model = new TransGGMPR(40, 0.02, 1);
	Model->run();

	return 0;
}