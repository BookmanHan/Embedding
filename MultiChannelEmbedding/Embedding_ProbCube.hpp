#pragma once
#include "Model.hpp"

class ProbCubeEmbedding
	:public CubeEmbeddingModel
{
public:
	ProbCubeEmbedding(double alpha)
		:CubeEmbeddingModel(alpha)
	{
		;
	}

public:
	virtual double prob_triplets( const pair<pair<string, string>,string>& triplet ) 
	{
		double total_prob = 0;
		for(auto i=0; i<set_relation.size(); ++i)
		{
			total_prob += embedding[i](name_entity[triplet.first.first],
				name_entity[triplet.first.second]);
		}

		return embedding[name_relation[triplet.second]](name_entity[triplet.first.first],
			name_entity[triplet.first.second])/total_prob;
	}

	virtual double train_once( const pair<pair<string, string>,string>& triplet, double factor ) 
	{
		double total_prob_r = 0;
		double total_prob_h = 0;
		double total_prob_t = 0;
		for(auto i=0; i<set_relation.size(); ++i)
		{
			total_prob_r += embedding[i](name_entity[triplet.first.first],
				name_entity[triplet.first.second]);
		}
		for(auto i=0; i<set_entity.size(); ++i)
		{
			total_prob_h += embedding[name_relation[triplet.second]]
				(i, name_entity[triplet.first.second]);
		}
		for(auto i=0; i<set_entity.size(); ++i)
		{
			total_prob_t += embedding[name_relation[triplet.second]]
				(name_entity[triplet.first.first], i);
		}

		embedding[name_relation[triplet.second]](name_entity[triplet.first.first],
			name_entity[triplet.first.second]) += 3 * alpha;

		for(auto i=0; i<set_relation.size(); ++i)
		{
			embedding[i](name_entity[triplet.first.first],
				name_entity[triplet.first.second]) -= 
				alpha *
				embedding[i](name_entity[triplet.first.first], 
				name_entity[triplet.first.second]) / total_prob_r;
		}
		for(auto i=0; i<set_entity.size(); ++i)
		{
			embedding[name_relation[triplet.second]](i, name_entity[triplet.first.second])
				-= alpha * embedding[name_relation[triplet.second]]
					(i, name_entity[triplet.first.second]) / total_prob_h;
		}
		for(auto i=0; i<set_entity.size(); ++i)
		{
			embedding[name_relation[triplet.second]](name_entity[triplet.first.first],i)
				-= alpha * embedding[name_relation[triplet.second]]
					(name_entity[triplet.first.first],i) / total_prob_h;
		}
	}
};

class DeepProbCubeEmbedding
	:public DeepCubeEmbeddingModel
{
public:
	DeepProbCubeEmbedding(int dim, double alpha)
		:DeepCubeEmbeddingModel(dim, alpha)
	{
		;
	}

public:
	virtual double prob_triplets( const pair<pair<string, string>,string>& triplet ) 
	{
		double total_prob = 0;
		for(auto i=0; i<set_relation.size(); ++i)
		{
			total_prob += norm(embedding[i](name_entity[triplet.first.first],
				name_entity[triplet.first.second]));
		}

		return norm(embedding[name_relation[triplet.second]](name_entity[triplet.first.first],
			name_entity[triplet.first.second]))/total_prob;
	}

	virtual double train_once( const pair<pair<string, string>,string>& triplet, double factor ) 
	{
		double total_prob = 0;
		double total_prob_r = 0;
		double total_prob_h = 0;
		double total_prob_t = 0;
		for(auto i=0; i<set_relation.size(); ++i)
		{
			total_prob_r += norm(embedding[i](name_entity[triplet.first.first],
				name_entity[triplet.first.second]));
		}
		for(auto i=0; i<set_entity.size(); ++i)
		{
			total_prob_h += norm(embedding[name_relation[triplet.second]]
			(i, name_entity[triplet.first.second]));
		}
		for(auto i=0; i<set_entity.size(); ++i)
		{
			total_prob_t += norm(embedding[name_relation[triplet.second]]
			(name_entity[triplet.first.first], i));
		}

		embedding[name_relation[triplet.second]](name_entity[triplet.first.first],
			name_entity[triplet.first.second]) += 3 * 
			alpha / pow(norm(embedding[name_relation[triplet.second]]
		(name_entity[triplet.first.first], name_entity[triplet.first.second])),3)
			* embedding[name_relation[triplet.second]](name_entity[triplet.first.first],
			name_entity[triplet.first.second]);

		for(auto i=0; i<set_relation.size(); ++i)
		{
			embedding[i](name_entity[triplet.first.first],
				name_entity[triplet.first.second]) -= 
				alpha / total_prob
				* embedding[i](name_entity[triplet.first.first], name_entity[triplet.first.second]);
		}
		for(auto i=0; i<set_entity.size(); ++i)
		{
			embedding[name_relation[triplet.second]](i, name_entity[triplet.first.second]) 
				-= alpha / total_prob
				* embedding[name_relation[triplet.second]](i, name_entity[triplet.first.second]) ;
		}
		for(auto i=0; i<set_relation.size(); ++i)
		{
			embedding[name_relation[triplet.second]](name_entity[triplet.first.first], i) -= 
				alpha / total_prob
				* embedding[name_relation[triplet.second]](name_entity[triplet.first.first], i);
		}
	}
};