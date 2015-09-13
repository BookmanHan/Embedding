#include "Import.hpp"
#include "Model.hpp"
#include "DetailedConfig.hpp"
#include "GeometricModel.hpp"
#include "OrbitModel.hpp"
#include <omp.h>

int main(int argc, char* argv[])
{
	//omp_set_num_threads(6); 
	
	Model*	model = nullptr;
	model = new OrbitBall(WN11, TripletClassification, report_path, 20, 0.01, 1.0);
	model->run(5000);
	model->test();
	delete model;

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

	return 0;
}