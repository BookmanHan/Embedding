#pragma once
#include "Import.hpp"
#include "ModelConfig.hpp"
#include "DataModel.hpp"
#include "Model.hpp"
#include "SemanticModel.hpp"
#include <boost/format.hpp>  
#include <boost/tokenizer.hpp>  
#include <boost/algorithm/string.hpp>  

class TopicRegressionTask
{
protected:
	vector<set<int>>			topics;
	vector<vector<int>>			topics_training;
	vector<int>					topic_list_id_to_raw;
	vector<int>					topic_list_raw_to_id;
	map<string, int>			topic_name_to_id;
	vector<int>					topic_count;
	TransE *					model;
	vector<vec>					input_sample;

protected:
	vector<vec>					lrc;

public:
	TopicRegressionTask(
		TransE* model,
		const string& type_file,
		const int evaluted_type = 50)
		:model(model), lrc(evaluted_type)
	{
		model->logging.record() << "\t[Task]\tEntity Classification";
		model->logging.record() << "\t[Evaluated Type Number]\t" << evaluted_type;

		int dim_init = model->entity_representation(1).n_rows;
		for (auto i = lrc.begin(); i != lrc.end(); ++i)
		{
			*i = randn(dim_init);
		}

		topics.resize(model->count_entity());

		fstream fin(type_file);
		while (!fin.eof())
		{
			boost::char_separator<char>	sep("\t ");
			string strin;

			getline(fin, strin);
			boost::tokenizer<boost::char_separator<char>> token(strin, sep);

			int pos = 0;
			for (auto i = token.begin(); i != token.end(); ++i)
			{
				if (i == token.begin())
				{
					pos = model->get_data_model().entity_name_to_id.find(*i)->second;
				}
				else
				{
					if (topic_name_to_id.find(*i) == topic_name_to_id.end())
					{
						topic_name_to_id[*i] = topic_name_to_id.size();
						topic_count.push_back(1);
					}

					topics[pos].insert(topic_name_to_id[*i]);
					++topic_count[topic_name_to_id[*i]];
				}
			}
		}
		fin.close();

		vector<int>	sorted_count = topic_count;
		sort(sorted_count.begin(), sorted_count.end(), greater<int>());
		int threshold = sorted_count[evaluted_type] - 1;

		set<int> filtered_topic;
		topic_list_raw_to_id.resize(topic_count.size());
		for (auto i = 0; i < topic_count.size(); ++i)
		{
			if (topic_count[i] > threshold && topic_count[i]!=sorted_count[0])
			{
				topic_list_raw_to_id[i] = topic_list_id_to_raw.size();
				topic_list_id_to_raw.push_back(i);
				filtered_topic.insert(i);
			}
		}

		for (auto i = topics.begin(); i != topics.end(); ++i)
		{
			vector<int>	filter_one(evaluted_type);
			for (auto it = i->begin(); it != i->end(); ++it)
			{
				if (filtered_topic.find(*it) != filtered_topic.end())
				{
					filter_one[topic_list_raw_to_id[*it]] = 1;
				}
			}

			topics_training.push_back(filter_one);
		}

		cout << "File Loaded." << endl;

		input_sample.resize(model->count_entity());
		for (auto i = 0; i < input_sample.size(); ++i)
		{
			const vec input_entity = model->entity_representation(i);
			input_sample[i] = input_entity;
		}
	}

public:
	void train(int epos, double alpha)
	{
		cout << endl;
		while (epos-- > 0)
		{
			cout << epos << ',';
			int pos = 0;

			for (auto i = topics_training.begin(); i != topics_training.end(); ++i)
			{
				++pos;
				if (pos % 10 == 0)
					continue;

				const vec& input = input_sample[i - topics_training.begin()];
				auto i_lrc = lrc.begin();
				for (auto it = i->begin(); it != i->end() && i_lrc != lrc.end(); ++it)
				{
					vec& lr_weight = *i_lrc;

					double score = 1.0 / (1.0 + exp(-as_scalar(lr_weight.t()*input)));
					lr_weight -= alpha * (score - *it) * input;
					++i_lrc;
				}
			}
		}
	}

	void test()
	{
		double hits = 0;
		int pos = 0;
		for (auto i = topics_training.begin(); i != topics_training.end(); ++i)
		{
			++pos;
			if (pos % 10 != 0)
				continue;

			const vec& input = input_sample[i - topics_training.begin()];
			auto i_lrc = lrc.begin();
			double hit = 0;
			for (auto it = i->begin(); it != i->end(); ++i_lrc, ++it)
			{
				vec& lr_weight = *i_lrc;

				double score = 1.0 / (1.0 + exp(-as_scalar(lr_weight.t()*input)));
				if ((score >= 0.5 && *it == 1) || (score <= 0.5 && *it == 0))
				{
					++hit;
				}
			}

			hits += hit;
		}

		hits /= topic_list_id_to_raw.size() * topics.size() / 10.0;
		 
		cout <<endl << "Entity Classification = " << hits <<endl;
		model->logging.record() << "Entity Classification MAP = " << hits;
	}
};

class TopicRegressionTaskZeroShot
	:public TopicRegressionTask
{
public:
	TopicRegressionTaskZeroShot(
		TransE* model,
		const string& type_file,
		const int evaluted_type = 50)
		:TopicRegressionTask(model, type_file, evaluted_type)
	{
		;
	}

public:
	void train(int epos, double alpha)
	{
		cout << endl;
		while (epos-- > 0)
		{
			cout << epos << ',';
			int pos = 0;
			for (auto i = topics_training.begin(); i != topics_training.end(); ++i)
			{
				++pos;
				if (pos >= model->get_data_model().zeroshot_pointer)
					break;

				const vec& input = input_sample[i - topics_training.begin()];
				auto i_lrc = lrc.begin();
				for (auto it = i->begin(); it != i->end(); ++it)
				{
					vec& lr_weight = *i_lrc;
					double score = 1.0 / (1.0 + exp(-as_scalar(lr_weight.t()*input)));
					lr_weight -= alpha * (score - *it) * input;
					++i_lrc;
				}
			}
		}
	}

	void test()
	{
		double hits = 0;
		int pos = 0;
		for (auto i = topics_training.begin(); i != topics_training.end(); ++i)
		{
			++pos;
			if (pos < model->get_data_model().zeroshot_pointer)
				continue;

			const vec& input = input_sample[i - topics_training.begin()];
			auto i_lrc = lrc.begin();
			double hit = 0;
			for (auto it = i->begin(); it != i->end(); ++i_lrc, ++it)
			{
				vec& lr_weight = *i_lrc;
				double score = 1.0 / (1.0 + exp(-as_scalar(lr_weight.t()*input)));
				if ((score >= 0.5 && *it == 1) || (score <= 0.5 && *it == 0))
				{
					++hit;
				}
			}

			hits += hit;
		}

		hits /= topic_list_id_to_raw.size() * topics.size() / 10.0;

		cout << endl << "Entity Classification = " << hits << endl;
		model->logging.record() << "Entity Classification MAP = " << hits;
	}
};

class ZeroShot
{
protected:
	Model* model;

public:
	ZeroShot(Model* model, const string dataset)
	{
		;
	}
};

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