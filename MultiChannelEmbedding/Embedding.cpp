#include "Import.hpp"
#include "DetailedConfig.hpp"
#include "GeometricModel.hpp"
#include "LatentModel.hpp"
#include <omp.h>

// 400s for each experiment.
int main(int argc, char* argv[])
{
	srand(time(nullptr));
	//omp_set_num_threads(6);

	Model* model = nullptr;

	model = new FactorE(FB15K, LinkPredictionTail, report_path, 100, 0.01, 0.1, 0.03);
	model->run(1500);
	model->test();
	delete model;

	model = new FactorE(FB15K, LinkPredictionTail, report_path, 100, 0.01, 0.1, 0.05);
	model->run(1500);
	model->test();
	delete model;

	model = new FactorE(FB15K, LinkPredictionTail, report_path, 100, 0.01, 0.1, 0.07);
	model->run(1500);
	model->test();
	delete model;

	model = new FactorE(FB15K, LinkPredictionTail, report_path, 100, 0.01, 0.1, 0.1);
	model->run(1500);
	model->test();
	delete model;

	model = new FactorE(FB15K, LinkPredictionTail, report_path, 100, 0.01, 0.1, 0.2);
	model->run(1500);
	model->test();
	delete model;

	model = new FactorE(FB15K, LinkPredictionTail, report_path, 100, 0.01, 0.1, 0.5);
	model->run(1500);
	model->test();
	delete model;

	model = new FactorE(FB15K, LinkPredictionTail, report_path, 100, 0.01, 0.05, 0.02);
	model->run(1500);
	model->test();
	delete model;

	model = new FactorE(FB15K, LinkPredictionTail, report_path, 100, 0.01, 0.2, 0.02);
	model->run(1500);
	model->test();
	delete model;

	model = new FactorE(FB15K, LinkPredictionTail, report_path, 100, 0.01, 0.3, 0.02);
	model->run(1500);
	model->test();
	delete model;

	model = new FactorE(FB15K, LinkPredictionTail, report_path, 100, 0.001, 0.1, 0.02);
	model->run(1500);
	model->test();
	delete model;

	model = new FactorE(FB15K, LinkPredictionTail, report_path, 100, 0.003, 0.1, 0.02);
	model->run(1500);
	model->test();
	delete model;

	model = new FactorE(FB15K, LinkPredictionTail, report_path, 100, 0.005, 0.1, 0.02);
	model->run(1500);
	model->test();
	delete model;

	model = new MFactorE(FB15K, LinkPredictionTail, report_path, 50, 0.01, 0.1, 0.01, 2);
	model->run(500);
	model->test();
	delete model;

	model = new MFactorE(FB15K, LinkPredictionTail, report_path, 10, 0.01, 0.1, 0.01, 10);
	model->run(500);
	model->test();
	delete model;

	return 0;
}