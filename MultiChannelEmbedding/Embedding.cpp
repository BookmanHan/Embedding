#include "Import.hpp"
#include "Model.hpp"
#include "DetailedConfig.hpp"
#include "GeometricModel.hpp"
#include <omp.h>

int main(int argc, char* argv[])
{
	omp_set_num_threads(8);
	Model*	model = nullptr;

	model = new TransA(FB15K, LinkPredictionTail, report_path, 100, 0.00175, 3.2);
	model->run(5000);
	model->test();
	delete model;

	//model = new TransM(FB15K, LinkPredictionTail, report_path, 400, 0.00175, 5 * exp(0.25), 5, 0.01, false, true);
	//model->run(10000);
	//model->test();
	//delete model;

	return 0;
}