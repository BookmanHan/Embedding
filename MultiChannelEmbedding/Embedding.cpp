#define Freebase
#define LP
#include "Model.hpp"
#include "Embedding_MultiChannel.hpp"
#include "Embedding_Geometric.hpp"
#include <cstdlib>
#include <omp.h>

int main(int argc, char* argv[])
{
	omp_set_num_threads(4);
	EmbeddingModel*	model = nullptr;

	model = new TransGPA(50, 2, exp(1), 0.001, 0.01, 10000, 1.0, false);
	model->log("TransGPA(50, 2, exp(1), 0.001, 0.01, 10000, 1.0, false) @ FB.15K : ");
	model->run(10000);
	delete model;

	return 0;
}