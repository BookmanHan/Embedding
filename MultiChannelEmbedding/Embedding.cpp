#define SSD_LOAD
#include "Import.hpp"
#include "Model.hpp"
#include "Task.hpp"
#include "DetailedConfig.hpp"
#include "GeometricModel.hpp"
#include "OrbitModel.hpp"
#include "LatentModel.hpp"
#include "SemanticModel.hpp"
#include <omp.h>

int main(int argc, char* argv[])
{
	srand(time(nullptr));
	omp_set_num_threads(6);

	Model* model = nullptr;

	{
		model = new SemanticModel
			(FB15K, General, report_path, semantic_vfile_FB15K,
				100, 0.001, 1.8, -0.2);
		model->run(5000);

		TopicRegressionTask type_regression((TransE*)model, type_file_FB15K);
		type_regression.train(100, 0.1);
		type_regression.test();
		delete model;
	}

	{
		model = new SemanticModel_Joint
			(FB15K, General, report_path, semantic_vfile_FB15K, semantic_tfile_FB15K,
				100, 0.001, 1.8, -0.2, 0.2);
		model->run(5000);

		TopicRegressionTask type_regression((TransE*)model, type_file_FB15K);
		type_regression.train(100, 0.1);
		type_regression.test();
		delete model;
	}

	{
		model = new SemanticModel
			(FB15K, triple_zeroshot_FB15K, General, report_path, semantic_vfile_FB15KZS,
				100, 0.001, 1.8, -0.2);
		model->run(5000);

		TopicRegressionTaskZeroShot type_regression((TransE*)model, type_file_FB15KZS);
		type_regression.train(100, 0.1);
		type_regression.test();
		delete model;
	}

	{
		model = new SemanticModel_Joint
			(FB15K, triple_zeroshot_FB15K, General, report_path, semantic_vfile_FB15KZS, semantic_tfile_FB15K,
				100, 0.001, 1.8, -0.2, 0.2);
		model->run(5000);

		TopicRegressionTaskZeroShot type_regression((TransE*)model, type_file_FB15KZS);
		type_regression.train(100, 0.1);
		type_regression.test();
		delete model;
	}

	return 0;
}