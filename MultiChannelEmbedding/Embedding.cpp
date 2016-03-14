#include "Import.hpp"
#include "DetailedConfig.hpp"
#include "GeometricModel.hpp"
#include "Task.hpp"
#include <omp.h>

int main(int argc, char* argv[])
{
	srand(time(nullptr));
	//omp_set_num_threads(6);

	Model* model = nullptr;
	DataModel dm(FB15K);

	model = new TransG_Hiracherical(FB15K, LinkPredictionTail, report_path, 100, 0.0015, 3.0, 1, 0.1);
	model->run(5000);
	model->test();
	delete model;

	model = new TransG_Hiracherical(FB15K, LinkPredictionTail, report_path, 200, 0.0015, 3.0, 1, 0.1);
	model->run(5000);
	model->test();
	delete model;

	model = new TransG_Hiracherical(FB15K, LinkPredictionTail, report_path, 300, 0.0015, 3.0, 1, 0.1);
	model->run(5000);
	model->test();
	delete model;

	return 0;
}