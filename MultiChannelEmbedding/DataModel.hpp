#pragma once
#include "Import.hpp"
#include "ModelConfig.hpp"

class DataModel
{
public:
	set<pair<pair<unsigned, unsigned>, unsigned> >		check_data_train;
	set<pair<pair<unsigned, unsigned>, unsigned> >		check_data_all;

public:
	vector<pair<pair<unsigned, unsigned>, unsigned> >	data_train;
	vector<pair<pair<unsigned, unsigned>, unsigned> >	data_dev_true;
	vector<pair<pair<unsigned, unsigned>, unsigned> >	data_dev_false;
	vector<pair<pair<unsigned, unsigned>, unsigned> >	data_test_true;
	vector<pair<pair<unsigned, unsigned>, unsigned> >	data_test_false;

public:
	set<unsigned>			set_tail;
	set<unsigned>			set_head;
	set<string>			set_entity;
	set<string>			set_relation;

public:
	vector<set<unsigned>>	set_relation_tail;
	vector<set<unsigned>>	set_relation_head;

public:
	vector<unsigned>	relation_type;

public:
	vector<string>		entity_id_to_name;
	vector<string>		relation_id_to_name;
	map<string, unsigned>	entity_name_to_id;
	map<string, unsigned>	relation_name_to_id;

public:
	vector<double>		prob_head;
	vector<double>		prob_tail;
	vector<double>		relation_tph;
	vector<double>		relation_hpt;
	map<string, unsigned>	count_entity;

public:
	map<unsigned, map<unsigned, unsigned> >	tails;
	map<unsigned, map<unsigned, unsigned> >	heads;

public:
	map<unsigned, map<unsigned, vector<unsigned> > >     rel_heads;
	map<unsigned, map<unsigned, vector<unsigned> > >     rel_tails;

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

		load_testing(dataset.base_dir + dataset.developing, data_dev_true, data_dev_false, dataset.self_false_sampling);
		load_testing(dataset.base_dir + dataset.testing, data_test_true, data_test_false, dataset.self_false_sampling);
		
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

		for(auto & elem : prob_head)
		{
			elem /= data_train.size();
		}

		for(auto & elem : prob_tail)
		{
			elem /= data_train.size();
		}

		relation_type.resize(set_relation.size());
		for(auto i=0; i<set_relation.size(); ++i)
		{
			if (relation_tph[i]<1.5 && relation_hpt[i]<1.5)
			{
				relation_type[i] = 1;
			}
			else if (relation_hpt[i] <1.5 && relation_tph[i] >= 1.5)
			{
				relation_type[i] = 2;
			}
			else if (relation_hpt[i] >=1.5 && relation_tph[i] < 1.5)
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
		fstream fin(filename);
		while(!fin.eof())
		{
			string head, tail, relation;
			fin>>head>>relation>>tail;

			if (entity_name_to_id.find(head) == entity_name_to_id.end())
			{
				entity_name_to_id.insert(make_pair(head, entity_name_to_id.size()));
			}

			if (entity_name_to_id.find(tail) == entity_name_to_id.end())
			{
				entity_name_to_id.insert(make_pair(tail, entity_name_to_id.size()));
			}

			if (relation_name_to_id.find(relation) == relation_name_to_id.end())
			{
				relation_name_to_id.insert(make_pair(relation, relation_name_to_id.size()));
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
		}

		fin.close();
	}

	void load_testing(	
		const string& filename, 
		vector<pair<pair<unsigned, unsigned>,unsigned>>& vin_true,
		vector<pair<pair<unsigned, unsigned>,unsigned>>& vin_false,
		bool self_sampling = false)
	{
		fstream fin(filename);
		if (self_sampling == false)
		{
			while(!fin.eof())
			{
				string head, tail, relation;
				int flag_true;

				fin>>head>>relation>>tail;
				fin>>flag_true;

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
				pair<pair<unsigned, unsigned>, unsigned>	sample_false;

				fin>>head>>relation>>tail;

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
		const pair<pair<unsigned,unsigned>,unsigned>& origin,
		pair<pair<unsigned,unsigned>,unsigned>& triplet) const
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
};