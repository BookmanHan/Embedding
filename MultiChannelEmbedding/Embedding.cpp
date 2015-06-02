#define FreebaseTC
#include "Model.hpp"
#include "Embedding_MultiChannel.hpp"
#include "Embedding_Geometric.hpp"
#include <cstdlib>
#include <omp.h>

//TransG(50, 5, 8, 0.001) @ WN18.Round5000 = 468, 457, 82.5, 94.4

//TransGA(50, 5, 500, 2000,  0.00001, 0.01) @ WN11.Round500 = 87.7 %
//TransG(50, 4, exp(1), 0.0001) @ FB13.Round500 = 82.9 %
int main(int argc, char* argv[])
{
	omp_set_num_threads(8);
	EmbeddingModel*	model = new TransG(50, 4, exp(1), 0.00012);
	model->run(10000);

	return 0;
}