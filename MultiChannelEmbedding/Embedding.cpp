#include "Import.hpp"
#include "Model.hpp"
#include "DetailedConfig.hpp"
#include "GeometricModel.hpp"
#include "OrbitModel.hpp"
#include <omp.h>

//TODO: 
//1.TransG : new componenet = tail - head;
//2.TransG_Hirachical.

//TransE-WN18-100-57%
//TransE-WN18-500-89%
//TransG-WN11-10-70%
//TransG-WN11-30-80%
//TransG-H-WN11-100-84%
//TransG-H-WN11-100-83.7%
//TransE-FB15K-50-19%.
int main(int argc, char* argv[])
{
	srand(time(nullptr));
	//omp_set_num_threads(4);

	Model*	model = nullptr;

	model = new OrbitE(FB15K, LinkPredictionTail, report_path, 400, 0.01, 3.0); 
	for(auto i=1; i<5; ++i)
	{
		model->run(i*500);
		model->test(1);
		model->test();
	}
	delete model;

	//model = new OrbitE_H(FB15K, LinkPredictionTail, report_path, 400, 0.001, 3.0); 
	//model->run(500);
	//model->test(1);
	//model->test();
	//delete model;

	//model = new OrbitE_H(FB15K, LinkPredictionHead, report_path, 400, 0.001, 4.0); 
	//model->run(500);
	//model->test(10);
	//model->test(1);
	//delete model;

	//model = new OrbitE_H(WN18, LinkPredictionTail, report_path, 400, 0.001, 4.0); 
	//model->run(500);
	//model->test(10);
	//model->test(1);
	//delete model;

	//Model* model = new TransE(FB13, DrawEmbedding, report_path, 2, 0.01, 10.0);
	//model->run(100);
	//for(auto i=0; i<13; ++i)
	//{
	//	model->draw("E:\\Freebase.", 250, i);
	//}
	//delete model;

	//Model*	model = nullptr;
	//model = new TransG_Hiracherical(FB13, TransM_ReportClusterNumber, report_path, 400, 0.001, 3.0, 1, 0.1, 5);
	//model->run(300);
	//model->report("");
	//model->test();
	//delete model;

	return 0;
}

//Model*	model = nullptr;
//model = new OrbitE2(Freebase, DrawEmbedding, report_path, 2, 0.01, 10.0);
////model = new TransE(Freebase, DrawEmbedding, report_path, 2, 0.01, 1.0);
//model->run(100);
//model->draw("D:\\OrbitE.philosopher.profession.big.ppm", 500, model->get_data_model().entity_name_to_id.at("philosopher"), 
//	model->get_data_model().relation_name_to_id.at("profession"));
//delete model;

//DataModel dm(FB15K);
//ModelLogging	log("F:\\");
//for(auto i=dm.rel_heads.begin(); i!=dm.rel_heads.end(); ++i)
//{
//	for(auto j=i->second.begin(); j!=i->second.end(); ++j)
//	{
//		for(auto a=j->second.begin(); a<j->second.end(); ++a)
//		{
//			for(auto b=a+1; b<j->second.end(); ++b)
//			{
//				for(auto c=b+1; c<j->second.end(); ++c)
//				{
//					if (dm.rel_finder.find(make_pair(*a, *c)) != dm.rel_finder.end() &&
//						dm.rel_finder.find(make_pair(*b, *c))->second == dm.rel_finder.find(make_pair(*a, *c))->second)
//					{
//						if (dm.rel_finder.find(make_pair(*a, *c))->second == i->first)
//							continue;
//						log.record();
//						log.record()<<dm.relation_id_to_name[i->first];
//						log.record()<<dm.relation_id_to_name[dm.rel_finder.find(make_pair(*a, *c))->second];
//						log.record()<<dm.entity_id_to_name[j->first];
//						log.record()<<dm.entity_id_to_name[*a];
//						log.record()<<dm.entity_id_to_name[*b];						
//						log.record()<<dm.entity_id_to_name[*c];							
//						log.record();	
//						cout<<'A';
//					}
//				}
//			}
//		}
//	}
//}

//model = new TransE(WN18, DrawEmbedding, report_path, 2, 0.01, 10);
//model->run(200);
//for(auto i=0; i<18; ++i)
//{
//	model->draw("F:\\Wordnet.", 250, i);
//}
//delete model;

//model = new TransA(FB13, TripletClassification, report_path, 200, 0.00175, 3.2);
//model->run(8000);
//model->test();
//delete model;