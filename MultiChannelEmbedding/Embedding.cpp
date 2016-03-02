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
		(FB15K, LinkPredictionRelation, report_path, 
			semantic_vfile_FB15K, 100, 0.003, 1.8, -0.2);
	model->run(3000);
	model->test();
	delete model;

	model = new SemanticModel
		(WN18, LinkPredictionTail, report_path,
			semantic_vfile_FB15K, 100, 0.001, 6.0, -0.2);
	model->run(3000);
	model->test();
	delete model;

	return 0;
}