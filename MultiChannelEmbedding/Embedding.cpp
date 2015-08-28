#include "Import.hpp"
#include "Model.hpp"
#include "DetailedConfig.hpp"
#include "GeometricModel.hpp"
#include <omp.h>

int main(int argc, char* argv[])
{
	omp_set_num_threads(8);
	Model*	model = nullptr;

	model = new TransM(FB15K, LinkPredictionTail, report_path, 400, 0.00175, 5 * exp(0.2), 5, 0.01, false, true);
	for(auto i=0; i<10; ++i)
	{
		model->run(500);
		model->test();
	}
	delete model;

	return 0;
}