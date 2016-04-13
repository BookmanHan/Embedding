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
			[&](vec & elem){elem = ones(dim);  (2 * randu(dim, 1) - 1)*sqrt(6.0 / dim); });

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
		//relation_out[triplet.second] -=
		//	alpha * factor_vec % entity[triplet.first.second];

		factor_vec = - sign(entity[triplet_f.first.first] % relation_in[triplet_f.second] 
		- entity[triplet_f.first.second] % relation_out[triplet_f.second]);

		entity[triplet_f.first.first] -= 
			alpha * factor_vec % relation_in[triplet_f.second];
		relation_in[triplet_f.second] -=
			alpha * factor_vec % entity[triplet_f.first.first];
		entity[triplet_f.first.second] += 
			alpha * factor_vec % relation_out[triplet_f.second];
		//relation_out[triplet_f.second] +=
		//	alpha * factor_vec % entity[triplet_f.first.second];

		if (norm_L2(entity[triplet.first.first]) > 1)
			entity[triplet.first.first] = normalise(entity[triplet.first.first]);
		if (norm_L2(entity[triplet.first.second]) > 1)
			entity[triplet.first.second] = normalise(entity[triplet.first.second]);
		if (norm_L2(entity[triplet_f.first.first]) > 1)
			entity[triplet_f.first.first] = normalise(entity[triplet_f.first.first]);
		if (norm_L2(entity[triplet_f.first.second]) > 1)
			entity[triplet_f.first.second] = normalise(entity[triplet_f.first.second]);

		relation_in[triplet.second] = normalise(relation_in[triplet.second]);
		relation_out[triplet.second] = normalise(relation_out[triplet.second]);
	}
};

class FactorEKL
	:public Model
{
protected:
	vector<vec>	embedding_entity;
	vector<vec>	embedding_relation_head;
	vector<vec>	embedding_relation_tail;

public:
	const double	margin;
	const double	alpha;
	const double	smoothing;
	const int		dim;

public:
	FactorEKL(
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold,
		double smoothing)
		:Model(dataset, task_type, logging_base_path),
		dim(dim), alpha(alpha), margin(training_threshold), smoothing(smoothing)
	{
		logging.record() << "\t[Name]\tFactorE";
		logging.record() << "\t[Dimension]\t" << dim;
		logging.record() << "\t[Learning Rate]\t" << alpha;
		logging.record() << "\t[Training Threshold]\t" << training_threshold;
		logging.record() << "\t[Smoothing]\t" << smoothing;

		embedding_entity.resize(count_entity());
		for_each(embedding_entity.begin(), embedding_entity.end(),
			[=](vec& elem){elem = normalise(randu(dim), 1); });

		embedding_relation_head.resize(count_relation());
		for_each(embedding_relation_head.begin(), embedding_relation_head.end(),
			[=](vec& elem){elem = normalise(randu(dim), 1); });

		embedding_relation_tail.resize(count_relation());
		for_each(embedding_relation_tail.begin(), embedding_relation_tail.end(),
			[=](vec& elem){elem = normalise(randu(dim), 1); });
	}

public:
	virtual double prob_triplets(const pair<pair<int, int>, int>& triplet) override
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation_head = embedding_relation_head[triplet.second];
		vec& relation_tail = embedding_relation_tail[triplet.second];

		vec head_feature = max(head % relation_head, smoothing * ones(dim));
		vec tail_feature = max(tail % relation_tail, smoothing * ones(dim));

		return - sum(head_feature % log(head_feature / tail_feature));
	}

	virtual void train_derv(const pair<pair<int, int>, int>& triplet, const double alpha)
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation_head = embedding_relation_head[triplet.second];
		vec& relation_tail = embedding_relation_tail[triplet.second];

		vec head_feature = max(head % relation_head, smoothing * ones(dim));
		vec tail_feature = max(tail % relation_tail, smoothing * ones(dim));

		head += alpha * relation_head % (1.0 + log(head_feature / tail_feature));
		relation_head += alpha * head % (1.0 + log(head_feature / tail_feature));
		tail -= alpha * relation_tail % head_feature / tail_feature;
		relation_tail -= alpha * tail % head_feature / tail_feature;

		head = normalise(abs(head), 1);
		tail = normalise(abs(tail), 1);
		relation_head = normalise(abs(relation_head), 1);
		relation_tail = normalise(abs(relation_tail), 1);
	}

	virtual void train_triplet(const pair<pair<int, int>, int>& triplet) override
	{
		pair<pair<int, int>, int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (prob_triplets(triplet)/prob_triplets(triplet_f) > exp(margin))
			return;

		train_derv(triplet, -alpha);
		train_derv(triplet_f, +alpha);
	}
};

class FactorE
	:public FactorEKL
{
public:
	FactorE(
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold)
		:FactorEKL(dataset, task_type, logging_base_path, dim, alpha, training_threshold, 0.0)
	{
		;
	}

public:
	virtual double prob_triplets(const pair<pair<int, int>, int>& triplet) override
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation_head = embedding_relation_head[triplet.second];
		vec& relation_tail = embedding_relation_tail[triplet.second];

		vec head_feature = head % relation_head;
		vec tail_feature = tail % relation_tail;

		return sum(head_feature) * sum(tail_feature)
			* exp(-sum(abs(head_feature - tail_feature)));
	}

	virtual void train_derv(const pair<pair<int, int>, int>& triplet, const double alpha) override
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation_head = embedding_relation_head[triplet.second];
		vec& relation_tail = embedding_relation_tail[triplet.second];

		vec head_feature = head % relation_head;
		vec tail_feature = tail % relation_tail;
		vec grad = sign(head_feature - tail_feature);

		head += alpha * grad % relation_head - alpha * relation_head / sum(head_feature);
		relation_head += alpha * grad % head - alpha * head / sum(head_feature);
		tail -= alpha * grad % relation_tail + alpha * relation_tail / sum(tail_feature);
		relation_tail -= alpha * grad % tail + alpha * tail / sum(tail_feature);

		head = normalise(abs(head), 2);
		tail = normalise(abs(tail), 2);
		relation_head = normalise(abs(relation_head), 2);
		relation_tail = normalise(abs(relation_tail), 2);
	}
};

class SFactorE
{
protected:
	vector<vec>	embedding_entity;
	vector<vec>	embedding_relation_head;
	vector<vec>	embedding_relation_tail;

public:
	const int		dim;

public:
	SFactorE(int dim, int entity_count, int relation_count)
		:dim(dim)
	{
		embedding_entity.resize(entity_count);
		for_each(embedding_entity.begin(), embedding_entity.end(),
			[=](vec& elem){elem = normalise(randu(dim), 1); });

		embedding_relation_head.resize(relation_count);
		for_each(embedding_relation_head.begin(), embedding_relation_head.end(),
			[=](vec& elem){elem = normalise(randu(dim), 1); });

		embedding_relation_tail.resize(relation_count);
		for_each(embedding_relation_tail.begin(), embedding_relation_tail.end(),
			[=](vec& elem){elem = normalise(randu(dim), 1); });
	}

	double prob(const pair<pair<int, int>, int>& triplet)
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation_head = embedding_relation_head[triplet.second];
		vec& relation_tail = embedding_relation_tail[triplet.second];

		vec head_feature = head % relation_head;
		vec tail_feature = tail % relation_tail;

		return sum(head_feature) * sum(tail_feature)
			* exp(-sum(abs(head_feature - tail_feature)));
	}

	void train(const pair<pair<int, int>, int>& triplet, const double alpha)
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation_head = embedding_relation_head[triplet.second];
		vec& relation_tail = embedding_relation_tail[triplet.second];

		vec head_feature = head % relation_head;
		vec tail_feature = tail % relation_tail;
		vec grad = sign(head_feature - tail_feature);

		head += alpha * grad % relation_head - alpha * relation_head / sum(head_feature);
		relation_head += alpha * grad % head - alpha * head / sum(head_feature);
		tail -= alpha * grad % relation_tail + alpha * relation_tail / sum(tail_feature);
		relation_tail -= alpha * grad % tail + alpha * tail / sum(tail_feature);

		head = normalise(abs(head), 2);
		tail = normalise(abs(tail), 2);
		relation_head = normalise(abs(relation_head), 2);
		relation_tail = normalise(abs(relation_tail), 2);
	}
};

class MFactorE
	:public Model
{
protected:
	vector<SFactorE*>	factors;
	vector<vec>			relation_space;

public:
	const double	margin;
	const double	alpha;
	const int		dim;
	const int		n_factor;

public:
	MFactorE(
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold,
		int n_factor)
		:Model(dataset, task_type, logging_base_path),
		dim(dim), alpha(alpha), margin(training_threshold), n_factor(n_factor)
	{
		logging.record() << "\t[Name]\tFactorE";
		logging.record() << "\t[Dimension]\t" << dim;
		logging.record() << "\t[Learning Rate]\t" << alpha;
		logging.record() << "\t[Training Threshold]\t" << training_threshold;
		logging.record() << "\t[Factor Number]\t" << n_factor;

		relation_space.resize(count_relation());
		for (vec& elem : relation_space)
		{
			elem = normalise(randu(n_factor));
		}

		for (auto i = 0; i < n_factor; ++i)
		{
			factors.push_back(new SFactorE(dim, count_entity(), count_relation()));
		}
	}

	vec get_error_vec(const pair<pair<int, int>, int>& triplet)
	{
		vec score(n_factor);
		auto i_score = score.begin();
		for (auto factor : factors)
		{
			*i_score++ = factor->prob(triplet);
		}

		return score;
	}

	virtual double prob_triplets(const pair<pair<int, int>, int>& triplet) override
	{
		return sum(get_error_vec(triplet) % relation_space[triplet.second]);
	}

	virtual void train_triplet(const pair<pair<int, int>, int>& triplet) override
	{
		pair<pair<int, int>, int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (prob_triplets(triplet) / prob_triplets(triplet_f) > n_factor * exp(margin))
			return;

		auto i = relation_space[triplet.second].begin();
		for (auto factor : factors)
		{
			factor->train(triplet, -alpha * (*i));
			factor->train(triplet_f, +alpha * (*i));

			++i;
		}

		relation_space[triplet.second] += alpha * get_error_vec(triplet);
		relation_space[triplet.second] -= alpha * get_error_vec(triplet_f);

		relation_space[triplet.second] = normalise(abs(relation_space[triplet.second]), 2);
	}
};