#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <arma>
#include <map>
#include <set>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cctype>

using namespace std;
using namespace arma;

class EmbeddingModel
{
protected:
	set<pair<pair<string, string>, string>>	data_train;
	vector<pair<pair<string, string>, string>>	data_dev_true;
	vector<pair<pair<string, string>, string>>	data_dev_false;
	vector<pair<pair<string, string>, string>>	data_test_true;
	vector<pair<pair<string, string>, string>>	data_test_false;
	set<string>	set_entity;
	set<string>	set_relation;
	vector<string>	number_entity;
	vector<string>	number_relation;
	map<string, int>	name_entity;
	map<string, int>	name_relation;
	map<string, int>	count_entity;
	map<string, vector<string>>     rel_heads;
	map<string, vector<string>>     rel_tails;

protected:
	const double alpha;

public:
	EmbeddingModel(double alpha)
		:alpha(alpha)
	{
		load_training("D:\\Data\\Wordnet\\train.txt", data_train);
		load_testing("D:\\Data\\Wordnet\\dev.txt", data_dev_true, data_dev_false);
		load_testing("D:\\Data\\Wordnet\\test.txt", data_test_true, data_test_false);

		number_entity.resize(set_entity.size());
		number_relation.resize(set_relation.size());
		for(auto i=name_entity.begin(); i!=name_entity.end(); ++i)
		{
			number_entity[i->second] = i->first;
		}
		for(auto i=name_relation.begin(); i!=name_relation.end(); ++i)
		{
			number_relation[i->second] = i->first;
		}
	}

	void load_training(const string& filename, set<pair<pair<string, string>,string>>& vin)
	{
		fstream fin(filename);
		while(!fin.eof())
		{
			string head, tail, relation;
			fin>>head>>relation>>tail;
			vin.insert(make_pair(make_pair(head,tail),relation));

			set_entity.insert(head);
			set_entity.insert(tail);
			set_relation.insert(relation);
			++ count_entity[head];
			++ count_entity[tail];
			rel_heads[relation].push_back(head);
			rel_tails[relation].push_back(tail);

			if (name_entity.find(head) == name_entity.end())
			{
				name_entity.insert(make_pair(head, name_entity.size()));
			}

			if (name_entity.find(tail) == name_entity.end())
			{
				name_entity.insert(make_pair(tail, name_entity.size()));
			}

			if (name_relation.find(relation) == name_relation.end())
			{
				name_relation.insert(make_pair(relation, name_relation.size()));
			}
		}

		fin.close();
	}

	void load_testing(	const string& filename, 
		vector<pair<pair<string, string>,string>>& vin_true,
		vector<pair<pair<string, string>,string>>& vin_false)
	{
		fstream fin(filename);
		while(!fin.eof())
		{
			string head, tail, relation;
			int flag_true;

			fin>>head>>relation>>tail;
			fin>>flag_true;

			if (flag_true == 1)
				vin_true.push_back(make_pair(make_pair(head,tail),relation));
			else
				vin_false.push_back(make_pair(make_pair(head, tail),relation));
		}

		fin.close();
	}

public:
	virtual double prob_triplets(const pair<pair<string, string>,string>& triplet) = 0;
	virtual double train_once(	const pair<pair<string, string>,string>& triplet,
		double factor) = 0;

public:
	double test()
	{
		double real_hit = 0;
		for(auto r=0; r<set_relation.size(); ++r)
		{
			vector<pair<double, bool>>	threshold_dev;
			for(auto i=data_dev_true.begin(); i!=data_dev_true.end(); ++i)
			{
				if (name_relation[i->second] != r)
					continue;

				threshold_dev.push_back(make_pair(prob_triplets(*i), true));
			}
			for(auto i=data_dev_false.begin(); i!=data_dev_false.end(); ++i)
			{
				if (name_relation[i->second] != r)
					continue;

				threshold_dev.push_back(make_pair(prob_triplets(*i), false));
			}

			sort(threshold_dev.begin(), threshold_dev.end());

			double threshold;
			double vari_mark = 0;
			unsigned int total = 0;
			unsigned int hit = 0;
			for(auto i=threshold_dev.begin(); i!=threshold_dev.end(); ++i)
			{
				if (i->second == false)
					++ hit;
				++ total;

				if (vari_mark <= 2*hit - total + data_dev_true.size())
				{
					vari_mark = 2*hit - total + data_dev_true.size();
					threshold = i->first;
				}
			}

			for(auto i=data_test_true.begin(); i!=data_test_true.end(); ++i)
			{
				if (name_relation[i->second] != r)
					continue;

				if (prob_triplets(*i) > threshold)
					++ real_hit;
			}

			for(auto i=data_test_false.begin(); i!=data_test_false.end(); ++i)
			{
				if (name_relation[i->second] != r)
					continue;

				if (prob_triplets(*i) <= threshold)
					++ real_hit;
			}
		}

		cout<<"Accuracy = "<<real_hit/(data_test_true.size() + data_test_false.size())<<endl;
		return real_hit/(data_test_true.size() + data_test_false.size());
	}

	void train(double alpha)
	{
		for(auto i=data_train.begin(); i!=data_train.end(); ++i)
		{
			train_once(*i, alpha);
		}
	}

	void sample_false_triplet(	const pair<pair<string,string>,string>& origin,
								pair<pair<string,string>,string>& triplet)
	{
		triplet = origin;
		if (rand()%1000 > 1000 * count_entity[triplet.first.second]
			/(count_entity[triplet.first.second] + count_entity[triplet.first.first]))
			{
				triplet.first.second = rel_tails[triplet.second]
					[rand()%rel_tails[triplet.second].size()];
			}
		else
		{
			triplet.first.first = rel_heads[triplet.second]
				[rand()%rel_heads[triplet.second].size()];
		}
	}

public:
	virtual void run()
	{
		while(true)
		{
			train(alpha);
			test();
		}
	}
};

class MultiChannelEmbeddingModel
	:public EmbeddingModel
{
protected:
	const unsigned int dim;
	vector<vector<vec>> embeddings;

public:
	MultiChannelEmbeddingModel(int dim, double alpha)
		:dim(dim), EmbeddingModel(alpha)
	{
		embeddings.resize(set_relation.size());
		for(auto i=embeddings.begin(); i!=embeddings.end(); ++i)
		{
			i->resize(set_entity.size());
			for(auto e=i->begin(); e!=i->end(); ++e)
			{
				*e = randu(dim, 1);
			}
		}
	}
};

class CubeEmbeddingModel
	:public EmbeddingModel
{
protected:
	vector<mat>		embedding;

public:
	CubeEmbeddingModel(double alpha)
		:EmbeddingModel(alpha)
	{
		embedding.resize(set_relation.size());
		for(auto i=embedding.begin(); i!=embedding.end(); ++i)
		{
			(*i) = randu(set_entity.size(), set_entity.size());
		}
	}
};

class DeepCubeEmbeddingModel
	:public EmbeddingModel
{
protected:
	vector<field<vec>>		embedding;
	const unsigned	dim;

public:
	DeepCubeEmbeddingModel(int dim, double alpha)
		:dim(dim), EmbeddingModel(alpha)
	{
		embedding.resize(set_relation.size());
		for(auto i=embedding.begin(); i!=embedding.end(); ++i)
		{
			i->set_size(set_entity.size(), set_entity.size());
			for(auto j=i->begin(); j!=i->end(); ++j)
			{
				(*j) = randu(dim);
			}
		}
	}
};

class GeometricEmbeddingModel
	:public EmbeddingModel
{
protected:
	vector<vec>	embedding_entity;
	vector<vec>	embedding_relation;
	const unsigned	dim;

public:
	GeometricEmbeddingModel(int dim, double alpha)
		:dim(dim), EmbeddingModel(alpha)
	{
		embedding_entity.resize(set_entity.size());
		for_each(embedding_entity.begin(), embedding_entity.end(), [=](vec& elem){elem = randu(dim,1);});

		embedding_relation.resize(set_relation.size());
		for_each(embedding_relation.begin(), embedding_relation.end(), [=](vec& elem){elem = randu(dim,1);});
	}
};