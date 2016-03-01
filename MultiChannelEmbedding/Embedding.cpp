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

	model = new SemanticModel
		(FB15K, General, report_path, semantic_vfile_FB15K,
			100, 0.001, 1.8, -2.0);
	model->run(300);
	{
		TopicRegressionTask type_regression((TransE*)model, type_file_FB15K);
		type_regression.train(500, 0.01);
		type_regression.test();
	}
	delete model;

	//model = new SemanticModel_Joint
	//	(FB15K, General, report_path, semantic_vfile_FB15K, semantic_tfile_FB15K,
	//		100, 0.001, 1.8, -0.2, 0.2);
	//model->run(3000);
	//{
	//	TopicRegressionTask type_regression((TransE*)model, type_file_FB15K);
	//	type_regression.train(500, 0.01);
	//	type_regression.test();
	//
	//}
	//delete model;

	return 0;
}