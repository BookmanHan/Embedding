#include "Model.hpp"
#include "Embedding_MultiChannel.hpp"
#include "Embedding_Geometric.hpp"
#include "Embedding_ProbCube.hpp"

// GeometricEmbeddingHadamard(100, 0.02) = 55.5%
int main(int argc, char* argv[])
{
	EmbeddingModel*	Model = new TransMP(50, 0.02);
	Model->run();

	return 0;
}