#pragma once
#include "Model.hpp"

#pragma once
#include "Model.hpp"

inline double sign(const double& x)
{
	return x>0?+1:-1;
}

class GeometricEmbeddingHadamard
	:public GeometricEmbeddingModel
{
public:
	GeometricEmbeddingHadamard(int dim, double alpha)
		:GeometricEmbeddingModel(dim, alpha)
	{
		;
	}

public:
	virtual double prob_triplets( const pair<pair<string, string>,string>& triplet )
	{
		double total = 0;;
		for(auto i=0; i<dim; ++i)
		{
			total += embedding_entity[name_entity[triplet.first.first]][i]
			* embedding_entity[name_entity[triplet.first.second]][i]
			* embedding_relation[name_relation[triplet.second]][i];
		}

		return total;
	}

	virtual double train_once( const pair<pair<string, string>,string>& triplet, double factor )
	{
		vec& head = embedding_entity[name_entity[triplet.first.first]];
		vec& tail = embedding_entity[name_entity[triplet.first.second]];
		vec& relation = embedding_relation[name_relation[triplet.second]];

		pair<pair<string, string>,string> triplet_f;
		sample_false_triplet(triplet, triplet_f);
		vec& head_f = embedding_entity[name_entity[triplet_f.first.first]];
		vec& tail_f = embedding_entity[name_entity[triplet_f.first.second]];
		vec& relation_f = embedding_relation[name_relation[triplet_f.second]];

		for(auto i=0; i<dim; ++i)
		{
			head[i] += alpha * tail[i] * relation[i];
			tail[i] += alpha * head[i] * relation[i];
			relation[i] += alpha * head[i] * tail[i];
			head_f[i] += alpha * tail_f[i] * relation_f[i];
			tail_f[i] += alpha * head_f[i] * relation_f[i];
			relation_f[i] += alpha * head_f[i] * tail_f[i];
		}

		normalise(head);
		normalise(tail);
		normalise(relation);
		normalise(head_f);
		normalise(tail_f);
		normalise(relation_f);
	}
};

class TransE
	:public GeometricEmbeddingModel
{
public:
	TransE(int dim, double alpha)
		:GeometricEmbeddingModel(dim, alpha)
	{
		;
	}

public:
	virtual double prob_triplets( const pair<pair<string, string>,string>& triplet )
	{
		double total = 0;;
		for(auto i=0; i<dim; ++i)
		{
			total += fabs(embedding_entity[name_entity[triplet.first.first]][i]
			- embedding_entity[name_entity[triplet.first.second]][i]
			+ embedding_relation[name_relation[triplet.second]][i]);
		}

		return - total;
	}

	virtual double train_once( const pair<pair<string, string>,string>& triplet, double factor )
	{
		vec& head = embedding_entity[name_entity[triplet.first.first]];
		vec& tail = embedding_entity[name_entity[triplet.first.second]];
		vec& relation = embedding_relation[name_relation[triplet.second]];

		pair<pair<string, string>,string> triplet_f;
		sample_false_triplet(triplet, triplet_f);
		vec& head_f = embedding_entity[name_entity[triplet_f.first.first]];
		vec& tail_f = embedding_entity[name_entity[triplet_f.first.second]];
		vec& relation_f = embedding_relation[name_relation[triplet_f.second]];

		for(auto i=0; i<dim; ++i)
		{
			head[i] -= alpha * sign(head[i] + relation[i] - tail[i]);
			tail[i] += alpha * sign(head[i] + relation[i] - tail[i]);
			relation[i] -= alpha * sign(head[i] + relation[i] - tail[i]);
			head_f[i] += alpha * sign(head_f[i] + relation_f[i] - tail_f[i]);
			tail_f[i] -= alpha * sign(head_f[i] + relation_f[i] - tail_f[i]);
			relation_f[i] += alpha * sign(head_f[i] + relation_f[i] - tail_f[i]);
		}

		//normalise(head);
		//normalise(tail);
		//normalise(relation);
		//normalise(head_f);
		//normalise(tail_f);
		//normalise(relation_f);
	}
};