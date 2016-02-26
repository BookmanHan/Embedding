#include "Import.hpp"
#include "Model.hpp"
#include "DetailedConfig.hpp"
#include "GeometricModel.hpp"
#include "OrbitModel.hpp"
#include "LatentModel.hpp"
#include "SemanticModel.hpp"
#include <omp.h>

int main(int argc, char* argv[])
{
	srand(time(nullptr));
	//omp_set_num_threads(1);

	Model* model = nullptr;
	
	model = new SemanticModel(FB15K, LinkPredictionTail, report_path, semantic_vector_file, 100, 0.01, 0.2);
	model->run(1000);
	model->test();
	delete model;

	model = new SemanticModel(FB15K, LinkPredictionTail, report_path, semantic_vector_file, 100, 0.01, 0.5);
	model->run(1000);
	model->test();
	delete model;

	model = new SemanticModel(FB15K, LinkPredictionTail, report_path, semantic_vector_file, 100, 0.01, 0.8);
	model->run(1000);
	model->test();
	delete model;

	model = new SemanticModel(FB15K, LinkPredictionTail, report_path, semantic_vector_file, 100, 0.01, 1.2);
	model->run(1000);
	model->test();
	delete model;

	model = new SemanticModel(FB15K, LinkPredictionTail, report_path, semantic_vector_file, 100, 0.01, 1.5);
	model->run(1000);
	model->test();
	delete model;

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