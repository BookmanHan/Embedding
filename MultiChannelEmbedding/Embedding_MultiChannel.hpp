#pragma once
#include "Model.hpp"

class MultiChannel_Innerproduct
	:public MultiChannelEmbeddingModel
{
public:
	MultiChannel_Innerproduct(int dim, double alpha)
		:MultiChannelEmbeddingModel(dim, alpha)
	{
		;
	}

public:
	virtual double prob_triplets( const pair<pair<string, string>,string>& triplet ) 
	{
		vector<double>	scores(set_relation.size());
		double total_score = 0;
		for(auto i=0; i<set_relation.size(); ++i)
		{
			scores[i] = exp(as_scalar(embeddings[i][name_entity[triplet.first.first]].t()
				*embeddings[i][name_entity[triplet.first.second]]));
		}
		for_each(scores.begin(), scores.end(), [&](double & elem){total_score += elem;});

		return scores[name_relation[triplet.second]]/total_score;
	}

	virtual double train_once(	const pair<pair<string, string>,string>& triplet, 
								double alpha) 
	{
		vector<double>	scores(set_relation.size());
		double total_score = 0;
		for(auto i=0; i<set_relation.size(); ++i)
		{
			scores[i] = exp(as_scalar(embeddings[i][name_entity[triplet.first.first]].t()
				*embeddings[i][name_entity[triplet.first.second]]));
		}
		for_each(scores.begin(), scores.end(), [&](double & elem){total_score += elem;});

		embeddings[name_relation[triplet.second]][name_entity[triplet.first.first]]
			+= alpha * embeddings[name_relation[triplet.second]][name_entity[triplet.first.second]];
		embeddings[name_relation[triplet.second]][name_entity[triplet.first.second]]
			+= alpha * embeddings[name_relation[triplet.second]][name_entity[triplet.first.first]];

		for(auto i=0; i<set_relation.size(); ++i)
		{
			embeddings[i][name_entity[triplet.first.first]] -= 
				alpha * scores[i]/total_score * embeddings[i][name_entity[triplet.first.second]]
				+ alpha * alpha * embeddings[i][name_entity[triplet.first.first]];
			embeddings[i][name_entity[triplet.first.second]]-= 
				alpha * scores[i]/total_score * embeddings[i][name_entity[triplet.first.first]]
				+ alpha * alpha * embeddings[i][name_entity[triplet.first.second]];
		}
	}
};