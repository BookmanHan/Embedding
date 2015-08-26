#include "Import.hpp"
#include "Model.hpp"
#include "DetailedConfig.hpp"
#include "GeometricModel.hpp"
#include <omp.h>

// Add loggin.
int main(int argc, char* argv[])
{
	Model*	model = nullptr;
	
	model = new TransM(FB15K, LinkPredictionTail, "G:\\สตั้\\Report\\Experiment.Embedding\\", 
		50, exp(1), 2.714, 5, 0.01);
	model->run(10000);
	model->test();

	delete model;

	return 0;
}