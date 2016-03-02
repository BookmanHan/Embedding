#pragma once
#include "Import.hpp"
#include "ModelConfig.hpp"

class DataModel
{
public:
	set<pair<pair<int, int>, int> >		check_data_train;
	set<pair<pair<int, int>, int> >		check_data_all;

public:
	vector<pair<pair<int, int>, int> >	data_train;
	vector<pair<pair<int, int>, int> >	data_dev_true;
	vector<pair<pair<int, int>, int> >	data_dev_false;
	vector<pair<pair<int, int>, int> >	data_test_true;
	vector<pair<pair<int, int>, int> >	data_test_false;

public:
	set<int>			set_tail;
	set<int>			set_head;
	set<string>			set_entity;
	set<string>			set_relation;

public:
	vector<set<int>>	set_relation_tail;
	vector<set<int>>	set_relation_head;

public:
	vector<int>	relation_type;

public:
	vector<string>		entity_id_to_name;
	vector<string>		relation_id_to_name;
	map<string, int>	entity_name_to_id;
	map<string, int>	relation_name_to_id;

public:
	vector<double>		prob_head;
	vector<double>		prob_tail;
	vector<double>		relation_tph;
	vector<double>		relation_hpt;
	map<string, int>	count_entity;

public:
	map<int, map<int, int> >	tails;
	map<int, map<int, int> >	heads;

public:
	map<int, map<int, vector<int> > >     rel_heads;
	map<int, map<int, vector<int> > >     rel_tails;
	map<pair<int, int>, int>		     rel_finder;
	
public:
	int zeroshot_pointer;

public:
	DataModel(const Dataset& dataset)
	{
		load_training(dataset.base_dir + dataset.training);
		relation_hpt.resize(set_relation.size());
		relation_tph.resize(set_relation.size());
		for(auto i=0; i!=set_relation.size(); ++i)
		{
			double sum = 0;
			double total = 0;
			for(auto ds=rel_heads[i].begin(); ds!=rel_heads[i].end(); ++ds)
			{
				++ sum;
				total += ds->second.size();
			}
			relation_tph[i] = total / sum;
		}
		for(auto i=0; i!=set_relation.size(); ++i)
		{
			double sum = 0;
			double total = 0;
			for(auto ds=rel_tails[i].begin(); ds!=rel_tails[i].end(); ++ds)
			{
				++ sum;
				total += ds->second.size();
			}
			relation_hpt[i] = total / sum;
		}

		zeroshot_pointer = set_entity.size();
		load_testing(dataset.base_dir + dataset.developing, data_dev_true, data_dev_false, dataset.self_false_sampling);
		load_testing(dataset.base_dir + dataset.testing, data_test_true, data_test_false, dataset.self_false_sampling);
		
		set_relation_head.resize(set_entity.size());
		set_relation_tail.resize(set_relation.size());
		prob_head.resize(set_entity.size());
		prob_tail.resize(set_entity.size());
		for(auto i=data_train.begin(); i!=data_train.end(); ++i)
		{
			++ prob_head[i->first.first];
			++ prob_tail[i->first.second];

			++ tails[i->second][i->first.first];
			++ heads[i->second][i->first.second];

			set_relation_head[i->second].insert(i->first.first);
			set_relation_tail[i->second].insert(i->first.second);
		}

#pragma omp parallel for
#pragma ivdep
		for (auto elem = prob_head.begin(); elem != prob_head.end(); ++elem)
		{
			*elem /= data_train.size();
		}

#pragma omp parallel for
#pragma ivdep
		for (auto elem = prob_tail.begin(); elem != prob_tail.end(); ++elem)
		{
			*elem /= data_train.size();
		}

		double threshold = 1.5;
		relation_type.resize(set_relation.size());

 		for(auto i=0; i<set_relation.size(); ++i)
		{
			if (relation_tph[i]<threshold && relation_hpt[i]<threshold)
			{
				relation_type[i] = 1;
			}
			else if (relation_hpt[i] <threshold && relation_tph[i] >= threshold)
			{
				relation_type[i] = 2;
			}
			else if (relation_hpt[i] >=threshold && relation_tph[i] < threshold)
			{
				relation_type[i] = 3;
			}
			else
			{
				relation_type[i] = 4;
			}
		}
	}

	DataModel(const Dataset& dataset, const string& file_zero_shot)
	{
		load_training(dataset.base_dir + dataset.training);

		relation_hpt.resize(set_relation.size());
		relation_tph.resize(set_relation.size());
		for (auto i = 0; i != set_relation.size(); ++i)
		{
			double sum = 0;
			double total = 0;
			for (auto ds = rel_heads[i].begin(); ds != rel_heads[i].end(); ++ds)
			{
				++sum;
				total += ds->second.size();
			}
			relation_tph[i] = total / sum;
		}
		for (auto i = 0; i != set_relation.size(); ++i)
		{
			double sum = 0;
			double total = 0;
			for (auto ds = rel_tails[i].begin(); ds != rel_tails[i].end(); ++ds)
			{
				++sum;
				total += ds->second.size();
			}
			relation_hpt[i] = total / sum;
		}

		zeroshot_pointer = set_entity.size();
		load_testing(dataset.base_dir + dataset.developing, data_dev_true, data_dev_false, dataset.self_false_sampling);
		load_testing(dataset.base_dir + dataset.testing, data_dev_true, data_dev_false, dataset.self_false_sampling);
		load_testing(file_zero_shot, data_test_true, data_test_false, dataset.self_false_sampling);

		set_relation_head.resize(set_entity.size());
		set_relation_tail.resize(set_relation.size());
		prob_head.resize(set_entity.size());
		prob_tail.resize(set_entity.size());
		for (auto i = data_train.begin(); i != data_train.end(); ++i)
		{
			++prob_head[i->first.first];
			++prob_tail[i->first.second];

			++tails[i->second][i->first.first];
			++heads[i->second][i->first.second];

			set_relation_head[i->second].insert(i->first.first);
			set_relation_tail[i->second].insert(i->first.second);
		}

		for (auto & elem : prob_head)
		{
			elem /= data_train.size();
		}

		for (auto & elem : prob_tail)
		{
			elem /= data_train.size();
		}

		double threshold = 1.5;
		relation_type.resize(set_relation.size());
		for (auto i = 0; i < set_relation.size(); ++i)
		{
			if (relation_tph[i] < threshold && relation_hpt[i] < threshold)
			{
				relation_type[i] = 1;
			}
			else if (relation_hpt[i] < threshold && relation_tph[i] >= threshold)
			{
				relation_type[i] = 2;
			}
			else if (relation_hpt[i] >= threshold && relation_tph[i] < threshold)
			{
				relation_type[i] = 3;
			}
			else
			{
				relation_type[i] = 4;
			}
		}
	}

	void load_training(const string& filename)
	{
		fstream fin(filename.c_str());
		while(!fin.eof())
		{
			string head, tail, relation;
			fin>>head>>relation>>tail;

			if (entity_name_to_id.find(head) == entity_name_to_id.end())
			{
				entity_name_to_id.insert(make_pair(head, entity_name_to_id.size()));
				entity_id_to_name.push_back(head);
			}

			if (entity_name_to_id.find(tail) == entity_name_to_id.end())
			{
				entity_name_to_id.insert(make_pair(tail, entity_name_to_id.size()));
				entity_id_to_name.push_back(tail);
			}

			if (relation_name_to_id.find(relation) == relation_name_to_id.end())
			{
				relation_name_to_id.insert(make_pair(relation, relation_name_to_id.size()));
				relation_id_to_name.push_back(relation);
			}

			data_train.push_back(make_pair(
				make_pair(entity_name_to_id[head], entity_name_to_id[tail]),
				relation_name_to_id[relation]));

			check_data_train.insert(make_pair(
				make_pair(entity_name_to_id[head], entity_name_to_id[tail]),
				relation_name_to_id[relation]));
			check_data_all.insert(make_pair(
				make_pair(entity_name_to_id[head], entity_name_to_id[tail]),
				relation_name_to_id[relation]));

			set_entity.insert(head);
			set_entity.insert(tail);
			set_relation.insert(relation);

			++ count_entity[head];
			++ count_entity[tail];

			rel_heads[relation_name_to_id[relation]][entity_name_to_id[head]]
				.push_back(entity_name_to_id[tail]);
			rel_tails[relation_name_to_id[relation]][entity_name_to_id[tail]]
				.push_back(entity_name_to_id[head]);
			rel_finder[make_pair(entity_name_to_id[head], entity_name_to_id[tail])]
				= relation_name_to_id[relation];
		}

		fin.close();
	}

	void load_testing(	
		const string& filename, 
		vector<pair<pair<int, int>,int>>& vin_true,
		vector<pair<pair<int, int>,int>>& vin_false,
		bool self_sampling = false)
	{
		fstream fin(filename.c_str());
		if (self_sampling == false)
		{
			while(!fin.eof())
			{
				string head, tail, relation;
				int flag_true;

				fin>>head>>relation>>tail;
				fin>>flag_true;

				if (entity_name_to_id.find(head) == entity_name_to_id.end())
				{
					entity_name_to_id.insert(make_pair(head, entity_name_to_id.size()));
					entity_id_to_name.push_back(head);
				}

				if (entity_name_to_id.find(tail) == entity_name_to_id.end())
				{
					entity_name_to_id.insert(make_pair(tail, entity_name_to_id.size()));
					entity_id_to_name.push_back(tail);
				}

				if (relation_name_to_id.find(relation) == relation_name_to_id.end())
				{
					relation_name_to_id.insert(make_pair(relation, relation_name_to_id.size()));
					relation_id_to_name.push_back(relation);
				}

				set_entity.insert(head);
				set_entity.insert(tail);
				set_relation.insert(relation);

				if (flag_true == 1)
					vin_true.push_back(make_pair(make_pair(entity_name_to_id[head], entity_name_to_id[tail]),
					relation_name_to_id[relation]));
				else
					vin_false.push_back(make_pair(make_pair(entity_name_to_id[head], entity_name_to_id[tail]),
					relation_name_to_id[relation]));

				check_data_all.insert(make_pair(make_pair(entity_name_to_id[head], entity_name_to_id[tail]),
					relation_name_to_id[relation])); 
			}
		}
		else
		{
			while(!fin.eof())
			{
				string head, tail, relation;
				pair<pair<int, int>, int>	sample_false;

				fin>>head>>relation>>tail;

				if (entity_name_to_id.find(head) == entity_name_to_id.end())
				{
					entity_name_to_id.insert(make_pair(head, entity_name_to_id.size()));
					entity_id_to_name.push_back(head);
				}

				if (entity_name_to_id.find(tail) == entity_name_to_id.end())
				{
					entity_name_to_id.insert(make_pair(tail, entity_name_to_id.size()));
					entity_id_to_name.push_back(tail);
				}

				if (relation_name_to_id.find(relation) == relation_name_to_id.end())
				{
					relation_name_to_id.insert(make_pair(relation, relation_name_to_id.size()));
					relation_id_to_name.push_back(relation);
				}

				set_entity.insert(head);
				set_entity.insert(tail);
				set_relation.insert(relation);

				sample_false_triplet(make_pair(make_pair(entity_name_to_id[head], entity_name_to_id[tail]),
					relation_name_to_id[relation]), sample_false);

				vin_true.push_back(make_pair(make_pair(entity_name_to_id[head], entity_name_to_id[tail]),
					relation_name_to_id[relation]));
				vin_false.push_back(sample_false); 

				check_data_all.insert(make_pair(make_pair(entity_name_to_id[head], entity_name_to_id[tail]),
					relation_name_to_id[relation]));
			}
		}

		fin.close();
	}

	void sample_false_triplet(	
		const pair<pair<int,int>,int>& origin,
		pair<pair<int,int>,int>& triplet) const
	{

		double prob = relation_hpt[origin.second]/(relation_hpt[origin.second] + relation_tph[origin.second]);

		triplet = origin;
		while(true)
		{
			if(rand()%1000 < 1000 * prob)
			{
				triplet.first.second = rand()%set_entity.size();
			}
			else
			{
				triplet.first.first = rand()%set_entity.size();
			}

			if (check_data_train.find(triplet) == check_data_train.end())
				return;
		}
	}

	void sample_false_triplet_relation(	
		const pair<pair<int,int>,int>& origin,
		pair<pair<int,int>,int>& triplet) const
	{

		double prob = relation_hpt[origin.second]/(relation_hpt[origin.second] + relation_tph[origin.second]);

		triplet = origin;
		while(true)
		{
			if (rand()%100 < 50)
				triplet.second = rand()%set_relation.size();
			else if (rand() % 1000 < 1000 * prob)
			{
				triplet.first.second = rand() % set_entity.size();
			}
			else
			{
				triplet.first.first = rand() % set_entity.size();
			}

			if (check_data_train.find(triplet) == check_data_train.end())
				return;
		}
	}
};