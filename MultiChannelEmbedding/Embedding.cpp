#include "Import.hpp"
#include "DetailedConfig.hpp"
#include "SemanticModel.hpp"
#include <omp.h>

int main(int argc, char* argv[])
{
	srand(time(nullptr));
	//omp_set_num_threads(6);

	Model* model = nullptr;

	model = new SemanticModel
		(WN18, LinkPredictionTail, report_path,
			semantic_vfile_WN18, 100, 0.001, 1.0, -0.2);
	model->run(3000);
	model->test();
	delete model;

	return 0;
}