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
#include <boost/format.hpp>  
#include <boost/tokenizer.hpp>  
#include <boost/algorithm/string.hpp> 

int main(int argc, char* argv[])
{
	srand(time(nullptr));
	//omp_set_num_threads(6);
	Model* model = nullptr;

	//freebase_LSI();

	//model = new SemanticModel
	//	(FB15K, General, report_path, semantic_vfile_FB15K,
	//		100, 0.001, 1.8, -0.2);
	//model->run(100);
	//{
	//	TopicRegressionTask type_regression((TransE*)model, type_file_FB15K);
	//	type_regression.train(200, 0.01);
	//	type_regression.test();
	//	delete model;
	//}

	//model = new SemanticModel_Joint
	//	(FB15K, General, report_path, semantic_vfile_FB15K, semantic_tfile_FB15K,
	//		100, 0.001, 1.8, -0.2, 0.2);
	//model->run(100);
	//{
	//	TopicRegressionTask type_regression((TransE*)model, type_file_FB15K);
	//	type_regression.train(200, 0.01);
	//	type_regression.test();
	//	delete model;
	//}

	//model = new SemanticModel
	//	(FB15K, triple_zeroshot_FB15K, General, report_path, semantic_vfile_FB15KZS,
	//		100, 0.001, 1.8, -0.2);
	//model->run(100);
	//{
	//	TopicRegressionTaskZeroShot type_regression((TransE*)model, type_file_FB15KZS);
	//	type_regression.train(200, 0.01);
	//	type_regression.test();
	//	delete model;
	//}

	model = new SemanticModel_Joint
		(FB15K, triple_zeroshot_FB15K, General, report_path, semantic_vfile_FB15KZS, semantic_tfile_FB15KZS,
			100, 0.001, 1.8, -0.2, 0.05);
	model->run(1000);
	{
		TopicRegressionTaskZeroShot type_regression((TransE*)model, type_file_FB15KZS);
		type_regression.train(500, 0.01);
		type_regression.test();
	}

	//int epos = 10;
	//while (epos-- > 0)
	//{
	//	model->run(1);
	//	{
	//		TopicRegressionTaskZeroShot type_regression((TransE*)model, type_file_FB15KZS);
	//		type_regression.train(200, 0.01);
	//		type_regression.test();
	//	}
	//}

	delete model;

	return 0;
}