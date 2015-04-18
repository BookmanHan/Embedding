#pragma once
#include "Model.hpp"

#pragma once
#include "Model.hpp"

inline double sign(const double& x)
{
	return x>0?+1:-1;
}

inline vec& sign(vec& x)
{
	for_each(x.begin(), x.end(), [](double& elem){elem=sign(elem);});
	return x;
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
		vec error = embedding_entity[name_entity[triplet.first.first]]
			% embedding_entity[name_entity[triplet.first.second]]
			% embedding_relation[name_relation[triplet.second]];

		return sum(error);
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

		head += alpha * tail % relation;
		tail += alpha * head % relation;
		relation += alpha * head % tail;
		head_f -= alpha * tail_f % relation_f;
		tail_f -= alpha * head_f % relation_f;
		relation_f -= alpha * head_f % tail_f;

		head = normalise(head);
		tail = normalise(tail);
		relation = normalise(relation);
		head_f = normalise(head_f);
		tail_f = normalise(tail_f);
		relation_f = normalise(relation_f);
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
		vec error = embedding_entity[name_entity[triplet.first.first]] 
			+ embedding_relation[name_relation[triplet.second]] 
			- embedding_entity[name_entity[triplet.first.second]];

		return - sum(abs(error));
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

		head -= alpha * sign(head + relation - tail);
		tail += alpha * sign(head + relation - tail);
		relation -= alpha * sign(head + relation - tail);
		head_f += alpha * sign(head_f + relation_f - tail_f);
		tail_f -= alpha * sign(head_f + relation_f - tail_f);
		relation_f += alpha * sign(head_f + relation_f - tail_f);

		head = normalise(head);
		tail = normalise(tail);
		relation = normalise(relation);
		head_f = normalise(head_f);
		tail_f = normalise(tail_f);
		relation_f = normalise(relation_f);
	}
};

class TransG
	:public GeneralGeometricEmbeddingModel
{
public:
	TransG(int dim, double alpha)
		:GeneralGeometricEmbeddingModel(dim, alpha)
	{
		;
	}

public:
	virtual double prob_triplets( const pair<pair<string, string>,string>& triplet )
	{
		vec error = embedding_entity[name_entity[triplet.first.first]] 
		+ embedding_relation[name_relation[triplet.second]] 
		- embedding_entity[name_entity[triplet.first.second]];

		return - as_scalar(abs(error).t() * mat_relation[name_relation[triplet.second]] * abs(error));
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

		head -= alpha * sign(head + relation - tail);
		tail += alpha * sign(head + relation - tail);
		relation -= alpha * sign(head + relation - tail);
		head_f += alpha * sign(head_f + relation_f - tail_f);
		tail_f -= alpha * sign(head_f + relation_f - tail_f);
		relation_f += alpha * sign(head_f + relation_f - tail_f);

		head = normalise(head);
		tail = normalise(tail);
		relation = normalise(relation);
		head_f = normalise(head_f);
		tail_f = normalise(tail_f);
		relation_f = normalise(relation_f);

		mat_relation[name_relation[triplet.second]] -= abs(head + relation - tail) * abs(head + relation - tail).t()
			- abs(head_f + relation_f - tail_f) * abs(head_f + relation_f - tail_f).t();
	}

	virtual void train(double alpha)
	{
		for_each(mat_relation.begin(), mat_relation.end(), [=](mat& elem){elem = eye(dim, dim);});
		EmbeddingModel::train(alpha);
		for_each(mat_relation.begin(), mat_relation.end(), [=](mat& elem){elem = normalise(elem);});
	}

};

class TransMP
	:public GeometricEmbeddingModel
{
public:
	TransMP(int dim, double alpha)
		:GeometricEmbeddingModel(dim, alpha)
	{
		;
	}

public:
	virtual double prob_triplets( const pair<pair<string, string>,string>& triplet )
	{
		vector<double>	scores(set_relation.size());
		double			total_score = 0;
		for(auto i=0; i<set_relation.size(); ++i)
		{
			vec error = embedding_entity[name_entity[triplet.first.first]] 
			+ embedding_relation[i] - embedding_entity[name_entity[triplet.first.second]];

			scores[i] = exp(-sum(abs(error)));
		}
		
		for_each(scores.begin(), scores.end(), [&](double& elem){total_score += elem;});

		return scores[name_relation[triplet.second]]/total_score;
	}

	virtual double train_once( const pair<pair<string, string>,string>& triplet, double factor )
	{
		pair<pair<string, string>,string> triplet_f;
		sample_false_triplet(triplet, triplet_f);

		vec& head = embedding_entity[name_entity[triplet.first.first]];
		vec& tail = embedding_entity[name_entity[triplet.first.second]];
		vec& relation = embedding_relation[name_relation[triplet.second]];
		vec& head_f = embedding_entity[name_entity[triplet_f.first.first]];
		vec& tail_f = embedding_entity[name_entity[triplet_f.first.second]];
		vec& relation_f = embedding_relation[name_relation[triplet_f.second]];

		vector<double>	scores(set_relation.size());
		double			total_score = 0;
		for(auto i=0; i<set_relation.size(); ++i)
		{
			vec error = embedding_entity[name_entity[triplet.first.first]] 
			+ embedding_relation[i] - embedding_entity[name_entity[triplet.first.second]];

			scores[i] = exp(-sum(abs(error)));
		}

		for_each(scores.begin(), scores.end(), [&](double& elem){total_score += elem;});

		head -= alpha * sign(head + relation - tail); 
		tail += alpha * sign(head + relation - tail);
		relation -= alpha * sign(head + relation - tail);
		
		for(auto r=0; r<set_relation.size(); ++r)
		{
			head += alpha * scores[r] / total_score * sign(head + embedding_relation[r] - tail); 
			tail -= alpha * scores[r] / total_score * sign(head + embedding_relation[r] - tail);
			embedding_relation[r] += alpha * scores[r] / total_score * sign(head + embedding_relation[r] - tail);
		}

		head_f += alpha * sign(head_f + relation_f - tail_f); 
		tail_f -= alpha * sign(head_f + relation_f - tail_f);
		relation_f += alpha * sign(head_f + relation_f - tail_f);

		for(auto r=0; r<set_relation.size(); ++r)
		{
			head_f -= alpha * scores[r] / total_score * sign(head_f + embedding_relation[r] - tail_f); 
			tail_f += alpha * scores[r] / total_score * sign(head_f + embedding_relation[r] - tail_f); 
			embedding_relation[r] -= alpha * scores[r] / total_score * sign(head_f + embedding_relation[r] - tail_f); 
		}

		head = normalise(head);
		tail = normalise(tail);
		relation = normalise(relation);
		head_f = normalise(head_f);
		tail_f = normalise(tail_f);
		relation_f = normalise(relation_f);
	}
};
