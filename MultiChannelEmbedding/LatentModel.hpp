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

inline
vec& max_filter_prob(vec& src, const double filter_factor)
{
	for (auto i = src.begin(); i != src.end(); ++i)
	{
		if (*i < filter_factor)
			*i = filter_factor;
	}

	return src;
}

class TopicE
	:public Model
{
protected:
	vector<vec>	embedding_entity;
	vector<vec> embedding_relation;
	vec			embedding_topic;

protected:
	vector<vec>	accprob_entity;
	vector<vec> accprob_relation;
	vec			accprob_topic;

public:
	const int		dim;
	const double	smoothing;
	const double	balance;
	const double	margin;

public:
	vec alpha;

public:
	TopicE(const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double smoothing_factor,
		double balance,
		double margin)
		:Model(dataset, task_type, logging_base_path),
		dim(dim), smoothing(smoothing_factor), balance(balance), margin(margin)
	{
		logging.record() << "\t[Name]\tTopicE";
		logging.record() << "\t[Dimension]\t" << dim;
		logging.record() << "\t[Smoothing Share]\t" << smoothing_factor;
		logging.record() << "\t[Balance]\t" << balance;
		logging.record() << "\t[Margin]\t" << margin;

		embedding_entity.resize(count_entity());
		for_each(embedding_entity.begin(), embedding_entity.end(), [=](vec& elem){elem = normalise(smoothing + randu(dim), 1); });

		embedding_relation.resize(count_relation());
		for_each(embedding_relation.begin(), embedding_relation.end(), [=](vec& elem){elem = normalise(smoothing + randu(dim), 1); });

		embedding_topic = normalise(smoothing + randu(dim), 1);

		alpha = ones(count_entity());
	}

	TopicE(const DataModel* data_model,
		const TaskType& task_type,
		ModelLogging* logging_p,
		int dim,
		double smoothing_factor,
		double balance,
		double margin)
		:Model(data_model, task_type, logging_p),
		dim(dim), smoothing(smoothing_factor), balance(balance), margin(margin)
	{
		logging.record() << "\t[Name]\tTopicE";
		logging.record() << "\t[Dimension]\t" << dim;
		logging.record() << "\t[Smoothing Share]\t" << smoothing_factor;
		logging.record() << "\t[Balance]\t" << balance;
		logging.record() << "\t[Margin]\t" << margin;

		embedding_entity.resize(count_entity());
		for_each(embedding_entity.begin(), embedding_entity.end(), [=](vec& elem){elem = normalise(smoothing + randu(dim), 1); });

		embedding_relation.resize(count_relation());
		for_each(embedding_relation.begin(), embedding_relation.end(), [=](vec& elem){elem = normalise(smoothing + randu(dim), 1); });

		embedding_topic = normalise(smoothing + randu(dim), 1);
		
		alpha = ones(count_entity());
	}

public:
	virtual double prob_triplets(const pair<pair<int, int>, int>& triplet) override
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		vec& prob = embedding_topic;

		double frac = sum(head % relation % tail / prob);
		double part = sum(head % relation);

		return frac / part;
	}

	void train_derv(const pair<pair<int, int>, int>& triplet, const double alpha)
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		vec& prob = embedding_topic;

		vec& ap_head = accprob_entity[triplet.first.first];
		vec& ap_tail = accprob_entity[triplet.first.second];
		vec& ap_relation = accprob_relation[triplet.second];
		vec& ap_prob = accprob_topic;

		//vec curprob_topic = normalise(tail / prob, 1);
		//vec curprob_topic = normalise(head / prob, 1);
		//vec curprob_topic = normalise(tail / prob, 1) + normalise(head / prob, 1);
		vec curprob_topic = normalise(head % tail % relation / prob);

		ap_head += alpha * curprob_topic;
		ap_tail += alpha * curprob_topic;
		ap_relation += alpha * curprob_topic;
		ap_prob += alpha * curprob_topic;

	}

	virtual void train_triplet(const pair<pair<int, int>, int>& triplet) override
	{
		pair<pair<int, int>, int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);
		
		train_derv(triplet, alpha[triplet.first.second]);

		if (prob_triplets(triplet)/prob_triplets(triplet_f) > margin)
			return;

		train_derv(triplet_f, -balance * alpha[triplet.first.second]);
	}

	virtual void train(bool last_time = false) override
	{
		accprob_entity.resize(count_entity());
		for_each(accprob_entity.begin(), accprob_entity.end(), [=](vec& elem){elem = zeros(dim); });

		accprob_relation.resize(count_relation());
		for_each(accprob_relation.begin(), accprob_relation.end(), [=](vec& elem){elem = zeros(dim); });

		accprob_topic = zeros(dim);

		Model::train(last_time);

		embedding_topic = normalise(accprob_topic - smoothing, 1);

		auto i_entity_embedding = embedding_entity.begin();
		auto i_entity_accprob = accprob_entity.begin();
		while (i_entity_accprob != accprob_entity.end() && i_entity_embedding != embedding_entity.end())
		{
			*i_entity_embedding = normalise(*i_entity_accprob + smoothing, 1);
			++i_entity_accprob;
			++i_entity_embedding;
		}

		auto i_relation_embedding = embedding_relation.begin();
		auto i_relation_accprob = accprob_relation.begin();
		while (i_relation_accprob != accprob_relation.end() && i_relation_embedding != embedding_relation.end())
		{
			*i_relation_embedding = normalise(*i_relation_accprob + smoothing, 1);
			++i_relation_accprob;
			++i_relation_embedding;
		}
	}
};

class MTopicE
	:public Model
{
public:
	vector<TopicE*>	topic_factor;
	vector<vec>		embedding_factor;
	vec				prob_factor;

public:
	vector<vec>		accprob_factor;
	vec				accprob_prob_factor;

public:
	const int		dim;
	const double	smoothing;
	const double	balance;
	const double	margin;
	const int		n_factor;

public:
	MTopicE(const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double smoothing_factor,
		double balance,
		double margin,
		int n_factor)
		:Model(dataset, task_type, logging_base_path),
		dim(dim), smoothing(smoothing_factor), balance(balance), margin(margin), n_factor(n_factor)
	{
		logging.record() << "\t[Name]\tMTopicE";
		logging.record() << "\t[Dimension]\t" << dim;
		logging.record() << "\t[Smoothing Share]\t" << smoothing_factor;
		logging.record() << "\t[Balance]\t" << balance;
		logging.record() << "\t[Margin]\t" << margin;
		logging.record() << "\t[Factor]\t" << n_factor;

		for (auto i = 0; i < n_factor; ++i)
		{
			topic_factor.push_back(new TopicE(&data_model, task_type, &logging, 
				dim, smoothing_factor, balance, margin));
		}

		embedding_factor.resize(count_entity());
		for_each(embedding_factor.begin(), embedding_factor.end(), [=](vec& elem){elem = normalise(smoothing + ones(n_factor), 1); });

		prob_factor = normalise(ones(n_factor), 1);
	}

public:
	virtual double prob_triplets(const pair<pair<int, int>, int>& triplet) override
	{
		vec acc_factor(n_factor);
		auto i_acc_factor = acc_factor.begin();
		for (auto i = topic_factor.begin(); i != topic_factor.end(); ++i)
		{
			*i_acc_factor = ((*i)->prob_triplets(triplet));
			++i_acc_factor;
		}

		return sum(acc_factor % embedding_factor[triplet.first.second] / prob_factor);
	}

	virtual void train_triplet(const pair<pair<int, int>, int>& triplet) override
	{
		vec acc_factor(n_factor);
		auto i_acc_factor = acc_factor.begin();
		for (auto i = topic_factor.begin(); i != topic_factor.end(); ++i)
		{
			*i_acc_factor = ((*i)->prob_triplets(triplet));
			++i_acc_factor;
		}

		accprob_factor[triplet.first.second] += acc_factor;
		accprob_prob_factor += acc_factor;
	}

	virtual void train(bool last_time = false) override
	{
		for (auto i = 0; i != n_factor; ++i)
		{
			for (auto j = 0; j < count_entity(); ++j)
			{
				topic_factor[i]->alpha[j] = embedding_factor[j][i] / prob_factor[i];
			}
			topic_factor[i]->train(last_time);
		}

		accprob_factor.resize(count_entity());
		for_each(accprob_factor.begin(), accprob_factor.end(), [=](vec& elem){elem = zeros(n_factor); });

		accprob_prob_factor = zeros(n_factor);

		Model::train(last_time);

		prob_factor = normalise(accprob_prob_factor + smoothing, 1);
		
		auto i_factor_embedding = embedding_factor.begin();
		auto i_factor_accprob = accprob_factor.begin();
		while (i_factor_accprob != accprob_factor.end() && i_factor_embedding != embedding_factor.end())
		{
			*i_factor_embedding = normalise(*i_factor_accprob + smoothing, 1);
			++i_factor_accprob;
			++i_factor_embedding;
		}
	}
};