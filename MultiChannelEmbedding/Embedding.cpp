#define Freebase
#define LP
#include "Model.hpp"
#include "Embedding_MultiChannel.hpp"
#include "Embedding_Geometric.hpp"
#include <cstdlib>
#include <omp.h>

int main(int argc, char* argv[])
{
	//omp_set_num_threads(4);

	EmbeddingModel*	model = nullptr;
	
	model = new TransE(2, 0.01);
	
	for(auto i=0; i<50; ++i)
	{
		model->train(0.01);
		cout<<i<<endl;
	}

	for(auto i=0; i<18; ++i)
	{
		model->draw("D:\\TransE.FB15K.", 100, i);
	}

	//model->run(1000);

	//model = new TransG(50, 4, exp(2), 0.001, 0.01, true);
	//model->log("TransG(50, 4, exp(2), 0.001, 0.01, true) @ WN.18 : ");
	//model->run(10000, true);
	delete model;

	return 0;
}