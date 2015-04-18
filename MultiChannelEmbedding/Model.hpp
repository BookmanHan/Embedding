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

using namespace std;
using namespace arma;

class EmbeddingModel
{
protected:
	vector<pair<pair<string, string>, string>>	data_train;
	vector<pair<pair<string, string>, string>>	data_dev_true;
	vector<pair<pair<string, string>, string>>	data_dev_false;
	vector<pair<pair<string, string>, string>>	data_test_true;
	vector<pair<pair<string, string>, string>>	data_test_false;
	set<string>	set_entity;
	set<string>	set_relation;
	map<string, int>	name_entity;
	map<string, int>	name_relation;

public:
	EmbeddingModel()
	{
		load_training("D:\\Data\\Wordnet\\train.txt", data_train);
		load_testing("D:\\Data\\Wordnet\\dev.txt", data_dev_true, data_dev_false);
		load_testing("D:\\Data\\Wordnet\\test.txt", data_test_true, data_test_false);
	}

	void load_training(const string& filename, vector<pair<pair<string, string>,string>>& vin)
	{
		fstream fin(filename);
		while(!fin.eof())
		{
			string head, tail, relation;
			fin>>head>>relation>>tail;
			vin.push_back(make_pair(make_pair(head,tail),relation));

			set_entity.insert(head);
			set_entity.insert(tail);
			set_relation.insert(relation);

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

		return real_hit/(data_test_true.size() + data_test_false.size());
	}

	void train(double alpha)
	{
		for(auto i=data_train.begin(); i!=data_train.end(); ++i)
		{
			train_once(*i, alpha);
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
	const double alpha;
	vector<vector<vec>> embeddings;

public:
	MultiChannelEmbeddingModel(int dim, double alpha)
		:dim(dim), alpha(alpha)
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
	const double	alpha;

public:
	CubeEmbeddingModel(double alpha)
		:alpha(alpha)
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
	const double	alpha;
	const unsigned	dim;

public:
	DeepCubeEmbeddingModel(int dim, double alpha)
		:dim(dim), alpha(alpha)
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
	const double	alpha;

public:
	GeometricEmbeddingModel(int dim, double alpha)
		:dim(dim), alpha(alpha)
	{
		embedding_entity.resize(set_entity.size());
		for_each(embedding_entity.begin(), embedding_entity.end(), [=](vec& elem){elem = randu(dim,1);});

		embedding_relation.reserve(set_relation.size());
		for_each(embedding_relation.begin(), embedding_relation.end(), [=](vec& elem){elem = randu(dim,1);});
	}
};