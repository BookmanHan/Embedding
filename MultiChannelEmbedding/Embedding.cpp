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
	model = new MFactorE(FB15K, TripletClassification, report_path, 20, 0.01, 0.05, 5);
	model->run(1000);
	model->test();
	delete model;

	return 0;
}