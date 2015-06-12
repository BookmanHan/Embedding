#define BOOST_NO_0X_HDR_INITIALIZER_LIST
#define Freebase
#define LP
#include "Model.hpp"
#include "Embedding_MultiChannel.hpp"
#include "Embedding_Geometric.hpp"
#include <cstdlib>
#include <omp.h>

int main(int argc, char* argv[])
{
	omp_set_num_threads(3);

	EmbeddingModel*	model = nullptr;

	model = new TransGPA(50, 2, exp(1), 0.001, 0.01, 8000, 0.5, false);
	model->log("TransGPA(50, 2, exp(1), 0.001, 0.01, 8000, 0.5, false) @ FB.15K : ");
	model->run(8000);
	delete model;

	model = new TransGPA(50, 2, exp(1), 0.001, 0.01, 8000, 1.2, false);
	model->log("TransGPA(50, 2, exp(1), 0.001, 0.01, 8000, 1.2, false) @ FB.15K : ");
	model->run(8000);
	delete model;

	model = new TransGPA(50, 2, exp(0.5), 0.001, 0.01, 8000, 0.8, false);
	model->log("TransGPA(50, 2, exp(0.5), 0.001, 0.01, 8000, 0.8, false) @ FB.15K : ");
	model->run(8000);
	delete model;

	model = new TransGPA(50, 2, exp(2), 0.001, 0.01, 8000, 0.8, false);
	model->log("TransGPA(50, 2, exp(2), 0.001, 0.01, 8000, 0.8, false) @ FB.15K : ");
	model->run(8000);
	delete model;

	model = new TransGPA(50, 5, exp(1), 0.001, 0.01, 8000, 0.8, false);
	model->log("TransGPA(50, 5, exp(1), 0.001, 0.01, 8000, 0.8, false) @ FB.15K : ");
	model->run(8000);
	delete model;

	return 0;
}