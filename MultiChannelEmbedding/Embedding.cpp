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
	model = new MTopicE(WN11, TripletClassification, report_path, 5, 1e-6, 0.1, 1.25, 100);
	model->run(500);
	model->test();
	delete model;

	return 0;
}