#include "Import.hpp"
#include "DetailedConfig.hpp"
#include "GeometricModel.hpp"
#include "LatentModel.hpp"
#include <omp.h>

// 84.2 : 85.0
int main(int argc, char* argv[])
{
	srand(time(nullptr));
	//omp_set_num_threads(6);

	Model* model = nullptr;
	
	model = new MFactorE(FB15K, LinkPredictionTail, report_path, 100, 0.005, 0.1, 0.5, 2);
	model->run(1000);
	model->test();
	delete model;

	return 0;
}