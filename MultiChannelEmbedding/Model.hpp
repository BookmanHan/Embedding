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
#include <boost/progress.hpp>

using namespace std;
using namespace arma;

class EmbeddingModel
{
protected:
	set<pair<pair<string, string>, string>>	check_data_train;
	vector<pair<pair<string, string>, string>>	data_train;
	vector<pair<pair<string, string>, string>>	data_dev_true;
	vector<pair<pair<string, string>, string>>	data_dev_false;
	vector<pair<pair<string, string>, string>>	data_test_true;
	vector<pair<pair<string, string>, string>>	data_test_false;
	
	set<pair<pair<unsigned, unsigned>, unsigned>>	i_check_data_train;
	vector<pair<pair<unsigned, unsigned>, unsigned>>	i_data_train;
	vector<pair<pair<unsigned, unsigned>, unsigned>>	i_data_dev_true;
	vector<pair<pair<unsigned, unsigned>, unsigned>>	i_data_dev_false;
	vector<pair<pair<unsigned, unsigned>, unsigned>>	i_data_test_true;
	vector<pair<pair<unsigned, unsigned>, unsigned>>	i_data_test_false;

	set<string>	set_entity;
	set<string>	set_relation;
	vector<set<int>>	set_tail;
	vector<string>	number_entity;
	vector<string>	number_relation;
	vector<double>	prob_head;
	vector<double> prob_tail;
	vector<double>		relation_tph;
	vector<double>		relation_hpt;		
	map<string, int>	name_entity;
	map<string, int>	name_relation;
	map<string, int>	count_entity;
	map<string, map<string, vector<string>>>     rel_heads;
	map<string, map<string, vector<string>>>     rel_tails;
	map<string, vector<string>>			gen_head;
	map<string, vector<string>>			gen_tail;

protected:
	const double	alpha;
	unsigned int	epos;
	double			best_result;
	double			best_mean;
	double			best_hitatten;

public:
	EmbeddingModel(double alpha)
		:alpha(alpha), best_result(0), best_mean(1e7), best_hitatten(0)
	{
		epos = 0;
		load_training("D:\\Data\\Wordnet-18\\train.txt");
		load_training("D:\\Data\\Wordnet-18\\dev.txt");

		relation_hpt.resize(set_relation.size());
		relation_tph.resize(set_relation.size());
		for(auto i=set_relation.begin(); i!=set_relation.end(); ++i)
		{
			double sum = 0;
			double total = 0;
			for(auto ds=rel_heads[*i].begin(); ds!=rel_heads[*i].end(); ++ds)
			{
				++ sum;
				total += ds->second.size();
			}
			relation_tph[name_relation[*i]] = total / sum;
		}
		for(auto i=set_relation.begin(); i!=set_relation.end(); ++i)
		{
			double sum = 0;
			double total = 0;
			for(auto ds=rel_tails[*i].begin(); ds!=rel_tails[*i].end(); ++ds)
			{
				++ sum;
				total += ds->second.size();
			}
			relation_hpt[name_relation[*i]] = total / sum;
		}

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

		load_testing("D:\\Data\\Wordnet-18\\dev.txt", data_dev_true, data_dev_false, true);
		load_testing("D:\\Data\\Wordnet-18\\test.txt", data_test_true, data_test_false, true);
		i_load_testing("D:\\Data\\Wordnet-18\\dev.txt", i_data_dev_true, i_data_dev_false, true);
		i_load_testing("D:\\Data\\Wordnet-18\\test.txt", i_data_test_true, i_data_test_false, true);

		cout<<"Entities = "<<set_entity.size()<<endl;

		set_tail.resize(set_relation.size());
		prob_head.resize(set_entity.size());
		prob_tail.resize(set_entity.size());
		for(auto i=i_data_train.begin(); i!=i_data_train.end(); ++i)
		{
			set_tail[i->second].insert(i->first.second);

			++ prob_head[i->first.first];
			++ prob_tail[i->first.second];
		}

		for(auto & elem : prob_head)
		{
			elem /= i_data_train.size();
		}

		for(auto & elem : prob_tail)
		{
			elem /= i_data_train.size();
		}
	}

	void load_training(const string& filename)
	{
		fstream fin(filename);
		while(!fin.eof())
		{
			string head, tail, relation;
			fin>>head>>relation>>tail;
			data_train.push_back(make_pair(make_pair(head,tail),relation));
			
			check_data_train.insert(make_pair(make_pair(head,tail),relation));
			set_entity.insert(head);
			set_entity.insert(tail);
			set_relation.insert(relation);
			++ count_entity[head];
			++ count_entity[tail];

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

			rel_heads[relation][head].push_back(tail);
			rel_tails[relation][tail].push_back(head);
			gen_head[relation].push_back(head);
			gen_tail[relation].push_back(tail);

			i_check_data_train.insert(make_pair(make_pair(name_entity[head], name_entity[tail]), 
				name_relation[relation]));
			i_data_train.push_back(make_pair(make_pair(name_entity[head],name_entity[tail]),
				name_relation[relation]));
		}

		fin.close();
	}

	void load_testing(	const string& filename, 
		vector<pair<pair<string, string>,string>>& vin_true,
		vector<pair<pair<string, string>,string>>& vin_false,
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
					vin_true.push_back(make_pair(make_pair(head,tail),relation));
				else
					vin_false.push_back(make_pair(make_pair(head, tail),relation));
			}
		}
		else
		{
			while(!fin.eof())
			{
				string head, tail, relation;
				pair<pair<string, string>,string>	sample_false;
				fin>>head>>relation>>tail;
				sample_false_triplet(make_pair(make_pair(head, tail), relation), sample_false);

				vin_true.push_back(make_pair(make_pair(head, tail),relation));
				vin_false.push_back(sample_false); 

				check_data_train.insert(make_pair(make_pair(head, tail), relation));
			}
		}

		fin.close();
	}

	void i_load_testing(	const string& filename, 
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
					vin_true.push_back(make_pair(make_pair(name_entity[head], name_entity[tail]),
						name_relation[relation]));
				else
					vin_false.push_back(make_pair(make_pair(name_entity[head], name_entity[tail]),
					name_relation[relation]));
			}
		}
		else
		{
			while(!fin.eof())
			{
				string head, tail, relation;
				pair<pair<unsigned, unsigned>, unsigned>	sample_false;
				fin>>head>>relation>>tail;
				sample_false_triplet(make_pair(make_pair(name_entity[head], name_entity[tail]),
					name_relation[relation]), sample_false);

				vin_true.push_back(make_pair(make_pair(name_entity[head], name_entity[tail]),
					name_relation[relation]));
				vin_false.push_back(sample_false); 

				i_check_data_train.insert(make_pair(make_pair(name_entity[head], name_entity[tail]),
					name_relation[relation]));
			}
		}

		fin.close();
	}

public:
	virtual double prob_triplets(const pair<pair<string, string>,string>& triplet)
	{
		;
	}

	virtual double prob_triplets(const pair<pair<unsigned, unsigned>,unsigned>& triplet)
	{
		return prob_triplets(make_pair(make_pair(number_entity[triplet.first.first],
			number_entity[triplet.first.second]), number_relation[triplet.second]));
	}

	virtual double train_once(	const pair<pair<string, string>,string>& triplet,
		double factor) 
	{
		;
	}

	virtual double train_once(	const pair<pair<unsigned, unsigned>,unsigned>& triplet,
		double factor) 
	{
		train_once(make_pair(make_pair(triplet.first.first,triplet.first.second), triplet.second), alpha);
	}

public:
	double test()
	{
		double real_hit = 0;
		for(auto r=0; r<set_relation.size(); ++r)
		{
			vector<pair<double, bool>>	threshold_dev;
			for(auto i=i_data_dev_true.begin(); i!=i_data_dev_true.end(); ++i)
			{
				if (i->second != r)
					continue;

				threshold_dev.push_back(make_pair(prob_triplets(*i), true));
			}
			for(auto i=i_data_dev_false.begin(); i!=i_data_dev_false.end(); ++i)
			{
				if (i->second != r)
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

			for(auto i=i_data_test_true.begin(); i!=i_data_test_true.end(); ++i)
			{
				if (i->second != r)
					continue;

				if (prob_triplets(*i) > threshold)
					++ real_hit;
			}

			for(auto i=i_data_test_false.begin(); i!=i_data_test_false.end(); ++i)
			{
				if (i->second != r)
					continue;

				if (prob_triplets(*i) <= threshold)
					++ real_hit;
			}
		}

		cout<<epos<<"\t Accuracy = "<<real_hit/(data_test_true.size() + data_test_false.size());
		best_result = max(best_result, real_hit/(data_test_true.size() + data_test_false.size()));
		cout<<", Best = "<<best_result<<endl;

		return real_hit/(data_test_true.size() + data_test_false.size());
	}

	virtual void train(double alpha)
	{
#pragma omp parallel for
		for(auto i=i_data_train.begin(); i!=i_data_train.end(); ++i)
		{
			train_once(*i, alpha);
		}
	}

	void test_hit()
	{
		double mean = 0;
		double hits = 0;
		double total = 0;

#pragma omp parallel for
		for(auto i=i_data_test_true.begin(); i!=i_data_test_true.end(); ++i)
		{
			++ total;

			auto t = *i;
			unsigned rmean = 0;
			double score_i = prob_triplets(*i);
			for(auto j=set_tail[t.second].begin(); j!=set_tail[t.second].end(); ++j)
			{
				t.first.second = *j;
				//if (i_check_data_train.find(t) != i_check_data_train.end())
				//	continue;

				if (score_i < prob_triplets(t))
					++ rmean;
			}

#pragma omp critical
			{
				mean += rmean;
				if (rmean <= 10)
					++ hits;
			}
		}

		cout<<endl;
		cout<<"MEANS = "<<mean/total<<endl;
		cout<<"HITS = "<<hits/total<<endl;

		best_mean = min(best_mean, mean/total);
		best_hitatten = max(best_hitatten, hits/total);

		cout<<"BestMEANS = "<<best_mean<<endl;
		cout<<"BestHITS = "<<best_hitatten<<endl;
	}

public:
	void sample_false_triplet(	const pair<pair<string,string>,string>& origin,
								pair<pair<string,string>,string>& triplet)
	{
		
		double prob = relation_hpt[name_relation[origin.second]]
			/(relation_hpt[name_relation[origin.second]] + relation_tph[name_relation[origin.second]]);
		
		triplet = origin;
		while(true)
		{
			if(rand()%1000 < 1000 * prob)
			{
				triplet.first.second = number_entity[rand()%number_entity.size()];
			}
			else
			{
				triplet.first.first = number_entity[rand()%number_entity.size()];
			}

			if (check_data_train.find(triplet) == check_data_train.end())
				return;
		}
	}

	void sample_false_triplet(	const pair<pair<unsigned,unsigned>,unsigned>& origin,
		pair<pair<unsigned,unsigned>,unsigned>& triplet)
	{

		double prob = relation_hpt[origin.second]
		/(relation_hpt[origin.second] + relation_tph[origin.second]);

		triplet = origin;
		while(true)
		{
			if(rand()%1000 < 1000 * prob)
			{
				triplet.first.second = rand()%number_entity.size();
			}
			else
			{
				triplet.first.first = rand()%number_entity.size();
			}

			if (i_check_data_train.find(triplet) == i_check_data_train.end())
				return;
		}
	}

	void sample_triplet(	
		const pair<pair<string,string>,string>& origin,
		pair<pair<string,string>,string>& triplet,
		bool head, bool relation, bool tail)
	{
		triplet = origin;
		if (relation)
		{
			triplet.second = number_relation[rand()%number_relation.size()];
		}
		if (head)
		{
			triplet.first.first = number_entity[rand()%number_entity.size()];
		}
		if (tail)
		{
			triplet.first.second = number_entity[rand()%number_entity.size()];
		}
	}

	void sample_triplet(	
		const pair<pair<unsigned,unsigned>,unsigned>& origin,
		pair<pair<unsigned,unsigned>,unsigned>& triplet,
		bool head, bool relation, bool tail)
	{
		triplet = origin;
		if (relation)
		{
			triplet.second = rand()%number_relation.size();
		}
		if (head)
		{
			triplet.first.first = rand()%number_entity.size();
		}
		if (tail)
		{
			triplet.first.second = rand()%number_entity.size();
		}
	}

public:
	void search_head(const string& tail, const string& relation)
	{
		vector<pair<double, string>>	runs;
		for(auto i=set_entity.begin(); i!=set_entity.end(); ++i)
		{
			runs.push_back(make_pair(- prob_triplets(make_pair(make_pair(*i, tail), relation)),*i));
		}

		sort(runs.begin(), runs.end());
		for(auto i=0; i<30; ++i)
		{
			cout<<runs[i].second<<endl;
		}
	}

public:
	virtual void run(int max_epos = 500)
	{
		best_result = 0;
		epos = 0;
		while(--max_epos)
		{
			++ epos;
			train(alpha);
			//test();

			cout<<epos<<',';
			if (epos%100 == 0)
			{
				test_hit();
				cout<<endl;
			}
		}
		test_hit();
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

class GeometricEmbeddingModel
	:public EmbeddingModel
{
protected:
	vector<vec>	embedding_entity;
	vector<vec>	embedding_relation;
	const unsigned	dim;

protected:
	enum componet	{componet_head, componet_tail, componet_relation, componet_matr, componet_mata};

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

class GeneralGeometricEmbeddingModel
	:public GeometricEmbeddingModel
{
protected:
	vector<mat>	mat_relation;

public:
	GeneralGeometricEmbeddingModel(int dim, double alpha)
		:GeometricEmbeddingModel(dim, alpha)
	{
		mat_relation.resize(set_relation.size());
		for_each(mat_relation.begin(), mat_relation.end(), [=](mat& elem){elem = eye(dim,dim);});
	}
};
