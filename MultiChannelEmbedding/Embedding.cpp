#include "Import.hpp"
#include "DetailedConfig.hpp"
#include "SemanticModel.hpp"
#include <omp.h>

int main(int argc, char* argv[])
{
	srand(time(nullptr));
	//omp_set_num_threads(6);

	Model* model = nullptr;

	model = new SemanticModel_ZeroShot
		(FB15K, triple_zeroshot_FB15K, LinkPredictionTailZeroShot, report_path,
			semantic_vfile_FB15KZS, semantic_tfile_FB15KZS, 100, 0.01, 1.8, -0.2, 0.2);
	model->run(100);
	model->test();
	delete model;

	return 0;
}