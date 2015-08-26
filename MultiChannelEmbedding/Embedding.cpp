#include "Import.hpp"
#include "Model.hpp"
#include "DetailedConfig.hpp"
#include "GeometricModel.hpp"
#include <omp.h>

int main(int argc, char* argv[])
{
	omp_set_num_threads(4);
	Model*	model = nullptr;
	
	model = new TransM(WN18, LinkPredictionTail, report_path, 50, 0.001, exp(2), 10, 0.01);
	model->run(10000);
	model->test();
	delete model;
	
	model = new TransM(FB15K, LinkPredictionTail, report_path, 50, 0.0015, exp(1), 5, 0.01);
	model->run(10000);
	model->test();
	delete model;

	model = new TransM(FB15K, LinkPredictionHead, report_path, 50, 0.0015, exp(1), 5, 0.01);
	model->run(10000);
	model->test();
	delete model;

	model = new TransM(FB15K, LinkPredictionTail, report_path, 50, 0.0015, exp(1), 5, 0);
	model->run(10000);
	model->test();
	delete model;

	return 0;
}