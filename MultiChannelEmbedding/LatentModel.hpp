#pragma once
#include "Import.hpp"
#include "ModelConfig.hpp"
#include "DataModel.hpp"
#include "Model.hpp"

class LatentModel
	:public Model
{
protected:
	field<vec>		embedding_head;
	field<vec>		embedding_tail;
	vec				embedding_topic;
	const int		n_topic;

protected:
	field<vec>		embedding_head_comp;
	field<vec>		embedding_tail_comp;
	vec				embedding_topic_comp;
	vector<vec>		embedding_relation;

public:
	LatentModel(		
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int topic)
		:Model(dataset, task_type, logging_base_path), n_topic(topic)
	{
			embedding_head = field<vec>(count_relation(), count_entity());
			for(auto i=embedding_head.begin(); i!=embedding_head.end(); ++i)
			{
				*i = randu(n_topic,1) * sqrt(6.0/n_topic);
			}

			embedding_tail = field<vec>(count_relation(), count_entity());
			for(auto i=embedding_tail.begin(); i!=embedding_tail.end(); ++i)
			{
				*i = randu(n_topic,1) * sqrt(6.0/n_topic);
			}

			embedding_topic.resize(n_topic);
			embedding_topic.fill(1.0/n_topic);
			normalize_to_prob();

			embedding_head_comp = field<vec>(count_relation(), count_entity());
			embedding_tail_comp = field<vec>(count_relation(), count_entity());

			embedding_topic_comp.resize(n_topic);
			embedding_relation.resize(count_relation());

			for(auto i=embedding_relation.begin(); i!=embedding_relation.end(); ++i)
			{
				*i = zeros(n_topic, 1);
			}

			for(auto j=0; j<count_relation(); ++j)
			{
				for(auto i=0; i<count_entity(); ++i)
				{
					embedding_relation[j] += embedding_head(j, i) + embedding_tail(j, i);
				}		
			}

			for(auto i=embedding_relation.begin(); i!=embedding_relation.end(); ++i)
			{
				*i /= sum(*i) + 1e-100;
			}
	}

public:
	void normalize_to_prob()
	{
		for(auto i=0; i<n_topic; ++i)
		{
			double total_sum = 0;
			for(auto item=embedding_head.begin(); item!=embedding_head.end(); ++item)
			{
				total_sum += (*item)(i);
			}

			if (total_sum > 1e-100)
			{
				for(auto item=embedding_head.begin(); item!=embedding_head.end(); ++item)
				{
					(*item)(i) /= total_sum;
				}
			}
		}

		for(auto i=0; i<n_topic; ++i)
		{
			double total_sum = 0;
			for(auto item=embedding_tail.begin(); item!=embedding_tail.end(); ++item)
			{
				total_sum += (*item)(i);
			}

			if (total_sum > 1e-100)
			{
				for(auto item=embedding_tail.begin(); item!=embedding_tail.end(); ++item)
				{
					(*item)(i) /= total_sum;
				}
			}
		}

		for(auto item=embedding_head.begin(); item!=embedding_head.end(); ++item)
		{
			*item += 0.01/(count_entity()*count_relation());
		}

		for(auto item=embedding_tail.begin(); item!=embedding_tail.end(); ++item)
		{
			*item += 0.01/(count_entity()*count_relation());
		}

		embedding_topic /= sum(embedding_topic);
	}

	virtual double prob_triplets(const pair<pair<int, int>,int>& triplet)
	{
		return sum(
			embedding_head(triplet.second, triplet.first.first) 
			% embedding_tail(triplet.second, triplet.first.second)
			% embedding_topic / embedding_relation[triplet.second]) + 1e-100;
	}

	virtual void train_triplet(const pair<pair<int, int>,int>& triplet)
	{
		embedding_head_comp(triplet.second, triplet.first.first) += 
			embedding_head(triplet.second, triplet.first.first) 
			% embedding_topic / prob_triplets(triplet);
		embedding_tail_comp(triplet.second, triplet.first.second) += 
			embedding_tail(triplet.second, triplet.first.second)
			% embedding_topic / prob_triplets(triplet);	
		embedding_topic_comp += embedding_topic / prob_triplets(triplet);
	}

	virtual void train(bool last_time = false)
	{
		for(auto i=embedding_head_comp.begin(); i!=embedding_head_comp.end(); ++i)
		{
			*i = zeros(n_topic, 1);
		}

		for(auto i=embedding_tail_comp.begin(); i!=embedding_tail_comp.end(); ++i)
		{
			*i = zeros(n_topic, 1);
		}

		embedding_topic_comp = zeros(n_topic, 1);

		Model::train(last_time);

		embedding_head = embedding_head_comp;
		embedding_tail = embedding_tail_comp;
		embedding_topic = embedding_topic_comp;

		normalize_to_prob();

		for(auto i=embedding_relation.begin(); i!=embedding_relation.end(); ++i)
		{
			*i = zeros(n_topic, 1);
		}

		for(auto j=0; j<count_relation(); ++j)
		{
			for(auto i=0; i<count_entity(); ++i)
			{
				embedding_relation[j] += embedding_tail(j, i) + embedding_head(j, i);
			}		
		}

		for(auto i=embedding_relation.begin(); i!=embedding_relation.end(); ++i)
		{
			*i /= sum(*i) + 1e-100;
		}

		cout<<embedding_topic.t();
	}
};

class PropergationModel
	:public Model
{
protected:
	vector<vec>		relation_in;
	vector<vec>		relation_out;
	vector<vec>		entity;
	const int		dim;
	const double	alpha;
	const double	training_threshold;

public:
	PropergationModel(		
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold)
		:Model(dataset, task_type, logging_base_path), 
		dim(dim), alpha(alpha), training_threshold(training_threshold)
	{
		logging.record()<<"\t[Dimension]\t"<<dim;
		logging.record()<<"\t[Learning Rate]\t"<<alpha;
		logging.record()<<"\t[Training Threshold]\t"<<training_threshold;
		logging.record()<<"\t[Name]\tPropergation Model";

		relation_in.resize(count_relation());
		for_each(relation_in.begin(), relation_in.end(), 
			[&](vec & elem){elem = (2*randu(dim,1)-1)*sqrt(6.0/dim);});

		relation_out.resize(count_relation());
		for_each(relation_out.begin(), relation_out.end(), 
			[&](vec & elem){elem = (2*randu(dim,1)-1)*sqrt(6.0/dim);});

		entity.resize(count_entity());
		for_each(entity.begin(), entity.end(), 
			[&](vec & elem){elem = (2*randu(dim,1)-1)*sqrt(6.0/dim);});
	}

	virtual double prob_triplets(const pair<pair<int, int>,int>& triplet)
	{
		double score = sum(abs(
			entity[triplet.first.first] % relation_in[triplet.second] 
			- entity[triplet.first.second] % relation_out[triplet.second]));
		
		return - score;
	}

	virtual void train_triplet(const pair<pair<int, int>,int>& triplet)
	{
		pair<pair<int, int>,int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > training_threshold)
			return;

		vec factor_vec = - sign(entity[triplet.first.first] % relation_in[triplet.second] 
		- entity[triplet.first.second] % relation_out[triplet.second]);

		entity[triplet.first.first] += 
			alpha * factor_vec % relation_in[triplet.second];
		relation_in[triplet.second] +=
			alpha * factor_vec % entity[triplet.first.first];
		entity[triplet.first.second] -= 
			alpha * factor_vec % relation_out[triplet.second];
		relation_out[triplet.second] -=
			alpha * factor_vec % entity[triplet.first.second];

		factor_vec = - sign(entity[triplet_f.first.first] % relation_in[triplet_f.second] 
		- entity[triplet_f.first.second] % relation_out[triplet_f.second]);

		entity[triplet_f.first.first] -= 
			alpha * factor_vec % relation_in[triplet_f.second];
		relation_in[triplet_f.second] -=
			alpha * factor_vec % entity[triplet_f.first.first];
		entity[triplet_f.first.second] += 
			alpha * factor_vec % relation_out[triplet_f.second];
		relation_out[triplet_f.second] +=
			alpha * factor_vec % entity[triplet_f.first.second];

		if (norm_L2(entity[triplet.first.first]) > 1)
			entity[triplet.first.first] = normalise(entity[triplet.first.first]);
		if (norm_L2(entity[triplet.first.second]) > 1)
			entity[triplet.first.second] = normalise(entity[triplet.first.second]);
		if (norm_L2(entity[triplet_f.first.first]) > 1)
			entity[triplet_f.first.first] = normalise(entity[triplet_f.first.first]);
		if (norm_L2(entity[triplet_f.first.second]) > 1)
			entity[triplet_f.first.second] = normalise(entity[triplet_f.first.second]);
		//if (norm_L2(relation_in[triplet.second]) > 1)
			relation_in[triplet.second] = normalise(relation_in[triplet.second]);
		//if (norm_L2(relation_out[triplet.second]) > 1)
			relation_out[triplet.second] = normalise(relation_out[triplet.second]);
	}
};