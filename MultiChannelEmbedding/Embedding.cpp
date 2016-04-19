#include "Import.hpp"
#include "DetailedConfig.hpp"
#include "LatentModel.hpp"
#include "OrbitModel.hpp"
#include "Task.hpp"
#include <omp.h>

// 400s for each experiment.
int main(int argc, char* argv[])
{
	srand(time(nullptr));
	//omp_set_num_threads(6);

	Model* model = nullptr;

	model = new MFactorSemantics
		(FB15K, General, report_path, semantic_tfile_FB15K, 10, 0.01, 0.1, 0.04, 100);
	model->load("D:\\Temp\\MFactorE.10-0.01-0.1-0.04-100.model");
	((MFactorSemantics*)model)->analyze();
	
	MFactorSemantics& m = *((MFactorSemantics*)model);
	while (true)
	{
		string str_in;
		cout << "Ask:";
		getline(cin, str_in);
		vector<int> re = m.infer_entity(str_in, 10);
		for (auto & elem : re)
		{
			cout << m.tells[elem].substr(0, 120) <<endl;
		}
	}
	delete model;

	return 0;
}