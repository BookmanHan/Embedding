#include "Model.hpp"
#include "Embedding_MultiChannel.hpp"
#include "Embedding_Geometric.hpp"
#include "Embedding_ProbCube.hpp"

// GeometricEmbeddingHadamard(100, 0.02) = 55.5%
// TransMP(50, 0.02, 1) = 82.90 %

int main(int argc, char* argv[])
{
	EmbeddingModel*	Model = new TransMPIP(50, 0.02, 1);
	Model->run();

	return 0;
}