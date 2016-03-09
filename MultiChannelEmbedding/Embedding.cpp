#include "Import.hpp"
#include "DetailedConfig.hpp"
#include "SemanticModel.hpp"
#include "OrbitModel.hpp"
#include "Task.hpp"
#include <omp.h>

int main(int argc, char* argv[])
{
	srand(time(nullptr));
	//omp_set_num_threads(6);

	Model* model = nullptr;
	model = new OrbitE_HDA(FB15K, LinkPredictionTail, report_path, 1000, 0.001, 0.2);
	model->run(500);
	model->test();
	model->run(500);
	model->test();
	delete model;

	return 0;

	DataModel dm(FB15K);
	model = new TransH
		(FB15K, LinkPredictionTail, report_path, 100, 0.001, 1.8);
	model->load("D:\\SSP.model");

	Model* cmodel = new TransE(FB15K, General, report_path, 100, 0.001, 1.0);
	cmodel->load("D:\\TransE.model");

	int a[5010] = { 0 };
	ModelLogging logging("D:\\result.");
	for (auto i = dm.data_test_true.begin(); i != dm.data_test_true.end(); ++i)
	{
		auto tmp = *i;
		auto pos = 0;
		for (auto j = 0; j < dm.set_entity.size(); ++j)
		{
			tmp.first.second = j;
			if (model->prob_triplets(*i) < model->prob_triplets(tmp))
			{
				++pos;
			}
		}

		if (pos > 100)
			continue;

		auto cpos = 0;
		for (auto j = 0; j < dm.set_entity.size(); ++j)
		{
			tmp.first.second = j;
			if (cmodel->prob_triplets(*i) < cmodel->prob_triplets(tmp))
			{
				++cpos;
			}
		}

		logging.record()
			<< pos << '\t' << cpos << '\t'
			<< dm.entity_id_to_name[i->first.first] << '\t'
			<< dm.relation_id_to_name[i->second] << '\t'
			<< dm.entity_id_to_name[i->first.second];

		++ a[cpos];
	}

	for (auto i = 0; i < 5001; ++i)
	{
		a[i + 1] += a[i];
		logging.record() << i << a[i];
	}

	delete model;
	delete cmodel;

	message("Finished");
	return 0;
}