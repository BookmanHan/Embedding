#include "Import.hpp"
#include "DetailedConfig.hpp"
#include "LatentModel.hpp"
#include <omp.h>

//	TODO:
//	New Embedding Framework
//	... It invokes an emergency.

int main(int argc, char* argv[])
{
	srand(time(nullptr));
	//omp_set_num_threads(6);

	Model* model = nullptr;
	model = new TopicE(WN11, TripletClassification, report_path, 100, 0, 0, 1e-2, 0);
	model->run(500);
	model->test_link_prediction();
	delete model;

	return 0;
}