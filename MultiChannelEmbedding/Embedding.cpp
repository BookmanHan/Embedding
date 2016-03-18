#define SSD_LOAD
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
	model = new SemanticModel_Joint(FB15K, LinkPredictionTail, report_path,
		semantic_vfile_FB15K, semantic_tfile_FB15K, 100, 0.001, 2.0, -0.2, 0.05);
	model->load("D:\\สตั้\\Model\\SSP.Joint.model");
	model->run(100);
	model->test();
	model->run(400);
	model->test();
	model->run(4500);
	model->test();
	model->run(4500);
	model->test();
	delete model;

	return 0;
}