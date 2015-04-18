#include "Model.hpp"
#include "Embedding_MultiChannel.hpp"
#include "Embedding_Geometric.hpp"
#include "Embedding_ProbCube.hpp"

int main(int argc, char* argv[])
{
	EmbeddingModel*	Model = new TransG(100, 0.02);
	Model->run();

	return 0;
}