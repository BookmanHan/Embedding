#define Freebase
#define LP
#include "Model.hpp"
#include "Embedding_MultiChannel.hpp"
#include "Embedding_Geometric.hpp"
#include <cstdlib>
#include <omp.h>

int main(int argc, char* argv[])
{
	EmbeddingModel*	model = nullptr;
	
	model = new TransE(2, 0.01);
	model->run(1000);

	//model = new TransG(50, 4, exp(2), 0.001, 0.01, true);
	//model->log("TransG(50, 4, exp(2), 0.001, 0.01, true) @ WN.18 : ");
	//model->run(10000, true);
	delete model;

	return 0;
}